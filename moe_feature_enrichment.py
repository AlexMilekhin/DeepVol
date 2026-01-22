"""
Feature enrichment module: Add regime indicators to existing SVI data.
Run AFTER historical_pipeline.py to avoid re-extracting SVI parameters.
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegimeFeatureExtractor:
    """Extract regime features from existing SVI time series."""
    
    def __init__(self, lookback_percentile: int = 60, min_periods: int = 20):
        self.lookback = lookback_percentile
        self.min_periods = min_periods
    
    def _expanding_then_rolling_rank(self, series: pd.Series) -> pd.Series:
        """
        Compute rolling percentile rank with expanding window for early periods.
        Avoids all-NaN results when history is shorter than lookback.
        """
        result = pd.Series(index=series.index, dtype=float)
        values = series.values
        
        for i in range(len(series)):
            if i < self.min_periods - 1:
                result.iloc[i] = np.nan
            elif i < self.lookback:
                # expanding window (all data up to this point)
                window = values[:i + 1]
                result.iloc[i] = (window < values[i]).sum() / len(window)
            else:
                # rolling window
                window = values[i - self.lookback + 1:i + 1]
                result.iloc[i] = (window < values[i]).sum() / len(window)
        
        return result
    
    def _expanding_then_rolling_zscore(self, series: pd.Series) -> pd.Series:
        """
        Compute rolling z-score with expanding window for early periods.
        """
        result = pd.Series(index=series.index, dtype=float)
        values = series.values
        
        for i in range(len(series)):
            if i < self.min_periods - 1:
                result.iloc[i] = np.nan
            elif i < self.lookback:
                # Use expanding window
                window = values[:i + 1]
                mean = np.mean(window)
                std = np.std(window) + 1e-8
                result.iloc[i] = (values[i] - mean) / std
            else:
                # Use rolling window
                window = values[i - self.lookback + 1:i + 1]
                mean = np.mean(window)
                std = np.std(window) + 1e-8
                result.iloc[i] = (values[i] - mean) / std
        
        return result
    
    def extract_regime_features(self, df_svi: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime features to SVI DataFrame.
        """
        logger.info(f"Extracting regime features from {len(df_svi)} records...")
        
        df = df_svi.copy()
        df = df.sort_values(['asof', 'expiry']).reset_index(drop=True)
        
        
        # For each date, get a representative IV (e.g., shortest expiry or average)
        
        daily_iv = df.groupby('asof').agg({
            'ATM_IV': 'mean',
            'ATM_skew': 'mean',
        }).sort_index()
        
        # Compute regime features on the DAILY series (not per-expiry)
        daily_iv['IV_percentile'] = self._expanding_then_rolling_rank(daily_iv['ATM_IV'])
        daily_iv['ATM_IV_zscore'] = self._expanding_then_rolling_zscore(daily_iv['ATM_IV'])
        daily_iv['skew_percentile'] = self._expanding_then_rolling_rank(daily_iv['ATM_skew'])
        daily_iv['ATM_skew_zscore'] = self._expanding_then_rolling_zscore(daily_iv['ATM_skew'])
        
        # Merge back to main df
        daily_features = ['IV_percentile', 'ATM_IV_zscore', 'skew_percentile', 'ATM_skew_zscore']
        df = df.merge(
            daily_iv[daily_features].reset_index(),
            on='asof',
            how='left'
        )
        
        
        # Discrete regime (for interpretation)
        df['IV_regime'] = pd.cut(
            df['ATM_IV'],
            bins=[0, 0.15, 0.25, 0.40, 1.0],
            labels=['low', 'normal', 'elevated', 'high']
        )
        
        # Volatility Momentum (per expiry)
        logger.info("  Computing IV momentum...")
        for lag in [5, 10, 20]:
            df[f'IV_change_{lag}d'] = df.groupby('expiry')['ATM_IV'].diff(lag)
        
        df['IV_momentum'] = df.groupby('expiry')['IV_change_5d'].transform(
            lambda x: x.rolling(5, min_periods=2).mean()
        )
        
        # Acceleration (second derivative)
        df['IV_acceleration'] = df.groupby('expiry')['IV_change_5d'].diff()
        
        # Skew regime
        df['skew_regime'] = pd.cut(
            df['ATM_skew'],
            bins=[-np.inf, -0.15, -0.05, 0.05, np.inf],
            labels=['extreme_put_skew', 'high_put_skew', 'neutral', 'call_skew']
        )
        
        # Curvature percentile - compute on daily aggregate
        daily_curv = df.groupby('asof')['ATM_curvature'].mean()
        daily_curv_pct = self._expanding_then_rolling_rank(daily_curv)
        curv_map = pd.Series(daily_curv_pct.values, index=daily_curv.index)
        df['curv_percentile'] = df['asof'].map(curv_map)
        
     
        logger.info("  Computing term structure indicators...")
        
        def compute_term_slope(group):
            if len(group) < 2:
                return pd.Series(0.0, index=group.index)
            try:
                slope = np.polyfit(group['T'], group['ATM_IV'], 1)[0]
                return pd.Series(slope, index=group.index)
            except:
                return pd.Series(0.0, index=group.index)
        
        df['term_slope'] = df.groupby('asof', group_keys=False).apply(compute_term_slope)
        
        df['term_structure'] = pd.cut(
            df['term_slope'],
            bins=[-np.inf, -0.05, 0.05, np.inf],
            labels=['inverted', 'flat', 'upward']
        )
        
        
        logger.info("  Computing vol-of-vol...")
        daily_iv['IV_change_5d'] = daily_iv['ATM_IV'].diff(5)
        daily_iv['IV_vol'] = daily_iv['IV_change_5d'].rolling(20, min_periods=5).std()
        df = df.merge(
            daily_iv[['IV_vol']].reset_index(),
            on='asof',
            how='left'
        )
        
        
        if 'ticker' in df.columns:
            logger.info("  Computing cross-sectional features...")
            df['IV_cross_sectional_rank'] = df.groupby(['asof', 'T'])['ATM_IV'].rank(pct=True)
        
        # Handle NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(0)
        
        logger.info(f"✅ Added {len(df.columns) - len(df_svi.columns)} new features")
        
        return df
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for regime features."""
        regime_features = [
            'IV_percentile', 'IV_momentum', 'IV_acceleration',
            'skew_percentile', 'curv_percentile', 'term_slope',
            'IV_vol', 'ATM_IV_zscore', 'ATM_skew_zscore'
        ]
        
        available = [f for f in regime_features if f in df.columns]
        summary = df[available].describe()
        
        return summary

def enrich_historical_svi(
    input_path: str,
    output_path: Optional[str] = None,
    lookback: int = 60
) -> pd.DataFrame:
    """
    Main function: Load SVI parquet, add regime features, save enriched version.
    
    Args:
        input_path: Path to historical SVI parquet (from historical_pipeline.py)
        output_path: Path for enriched output (defaults to input_path with '_enriched' suffix)
        lookback: Lookback window for percentile calculations
        
    Returns:
        Enriched DataFrame
    """
    logger.info("=" * 70)
    logger.info("FEATURE ENRICHMENT PIPELINE")
    logger.info("=" * 70)
    
    # Load existing SVI data
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"\n📂 Loading: {input_path}")
    df_svi = pd.read_parquet(input_path)
    logger.info(f"   Loaded: {len(df_svi)} records, {df_svi['asof'].nunique()} dates")
    
    # Convert dates
    df_svi['asof'] = pd.to_datetime(df_svi['asof'])
    df_svi['expiry'] = pd.to_datetime(df_svi['expiry'])
    
    # Extract features
    extractor = RegimeFeatureExtractor(lookback_percentile=lookback)
    df_enriched = extractor.extract_regime_features(df_svi)
    
    # Summary
    logger.info("\n📊 Feature Summary:")
    summary = extractor.get_feature_summary(df_enriched)
    print(summary)
    
    # Save
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_enriched.parquet"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df_enriched.to_parquet(output_path, index=False)
    logger.info(f"\n💾 Saved enriched data: {output_path}")
    logger.info(f"   Original columns: {len(df_svi.columns)}")
    logger.info(f"   Enriched columns: {len(df_enriched.columns)}")
    logger.info(f"   New features: {len(df_enriched.columns) - len(df_svi.columns)}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ ENRICHMENT COMPLETE")
    logger.info("=" * 70)
    
    return df_enriched


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enrich SVI data with regime features')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to historical SVI parquet')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: input_enriched.parquet)')
    parser.add_argument('--lookback', type=int, default=60,
                       help='Lookback window for percentiles (default: 60)')
    
    args = parser.parse_args()
    
    enrich_historical_svi(
        input_path=args.input,
        output_path=args.output,
        lookback=args.lookback
    )