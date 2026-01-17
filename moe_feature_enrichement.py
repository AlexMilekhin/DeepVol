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
        """
        Args:
            lookback_percentile: Window for percentile calculations (trading days)
            min_periods: Minimum periods required for rolling calculations
        """
        self.lookback = lookback_percentile
        self.min_periods = min_periods
    
    def extract_regime_features(self, df_svi: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime features to SVI DataFrame.
        
        Args:
            df_svi: DataFrame with columns [asof, expiry, T, ATM_IV, ATM_skew, ATM_curvature]
            
        Returns:
            Enriched DataFrame with regime features
        """
        logger.info(f"Extracting regime features from {len(df_svi)} records...")
        
        df = df_svi.copy()
        df = df.sort_values(['expiry', 'asof']).reset_index(drop=True)
        
        # 1. Volatility Level Regime
        logger.info("  Computing IV regime indicators...")
        df['IV_percentile'] = df.groupby('expiry')['ATM_IV'].transform(
            lambda x: x.rolling(self.lookback, min_periods=self.min_periods).rank(pct=True)
        )
        
        # Discrete regime (for interpretation)
        df['IV_regime'] = pd.cut(
            df['ATM_IV'],
            bins=[0, 0.15, 0.25, 0.40, 1.0],
            labels=['low', 'normal', 'elevated', 'high']
        )
        
        # 2. Volatility Momentum (trending vs mean-reverting)
        logger.info("  Computing IV momentum...")
        for lag in [5, 10, 20]:
            df[f'IV_change_{lag}d'] = df.groupby('expiry')['ATM_IV'].diff(lag)
        
        df['IV_momentum'] = df.groupby('expiry')['IV_change_5d'].transform(
            lambda x: x.rolling(5, min_periods=2).mean()
        )
        
        # Acceleration (second derivative)
        df['IV_acceleration'] = df.groupby('expiry')['IV_change_5d'].diff()
        
        # 3. Skew Regime (risk aversion / put demand)
        logger.info("  Computing skew regime...")
        df['skew_percentile'] = df.groupby('expiry')['ATM_skew'].transform(
            lambda x: x.rolling(self.lookback, min_periods=self.min_periods).rank(pct=True)
        )
        
        # Absolute skew level (more negative = more risk aversion)
        df['skew_regime'] = pd.cut(
            df['ATM_skew'],
            bins=[-np.inf, -0.15, -0.05, 0.05, np.inf],
            labels=['extreme_put_skew', 'high_put_skew', 'neutral', 'call_skew']
        )
        
        # 4. Curvature Regime (smile convexity)
        logger.info("  Computing curvature indicators...")
        df['curv_percentile'] = df.groupby('expiry')['ATM_curvature'].transform(
            lambda x: x.rolling(self.lookback, min_periods=self.min_periods).rank(pct=True)
        )
        
        # 5. Term Structure Features (requires multiple expiries per date)
        logger.info("  Computing term structure indicators...")
        
        # For each date, fit linear slope across expiries
        def compute_term_slope(group):
            if len(group) < 2:
                return pd.Series(0.0, index=group.index)
            try:
                # Slope of IV vs T
                slope = np.polyfit(group['T'], group['ATM_IV'], 1)[0]
                return pd.Series(slope, index=group.index)
            except:
                return pd.Series(0.0, index=group.index)
        
        df['term_slope'] = df.groupby('asof', group_keys=False).apply(compute_term_slope)
        
        # Term structure regime
        df['term_structure'] = pd.cut(
            df['term_slope'],
            bins=[-np.inf, -0.05, 0.05, np.inf],
            labels=['inverted', 'flat', 'upward']
        )
        
        # 6. Volatility of Volatility (realized vol of IV changes)
        logger.info("  Computing vol-of-vol...")
        df['IV_vol'] = df.groupby('expiry')['IV_change_5d'].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
        
        # 7. Temporal Distance Features (for attention weighting)
        logger.info("  Computing temporal features...")
        
        # Days since extreme IV
        def days_since_extreme(series, threshold=0.9):
            """Days since IV was in top 10%"""
            extreme_dates = series[series > series.quantile(threshold)].index
            if len(extreme_dates) == 0:
                return pd.Series(999, index=series.index)
            
            result = pd.Series(index=series.index, dtype=float)
            for idx in series.index:
                past_extremes = extreme_dates[extreme_dates <= idx]
                if len(past_extremes) > 0:
                    result[idx] = idx - past_extremes[-1]
                else:
                    result[idx] = 999
            return result
        
        df['days_since_high_iv'] = df.groupby('expiry')['ATM_IV'].transform(
            lambda x: days_since_extreme(x, 0.9)
        )
        
        # 8. Cross-sectional Features (when available)
        # Note: These only work if you have multiple tickers
        if 'ticker' in df.columns:
            logger.info("  Computing cross-sectional features...")
            
            # IV rank vs other tickers on same date
            df['IV_cross_sectional_rank'] = df.groupby(['asof', 'T'])['ATM_IV'].rank(pct=True)
            
            # Correlation with market (requires multiple tickers)
            # Skip for now - needs separate implementation
        
        # 9. Statistical Features for MoE Gating
        logger.info("  Computing statistical indicators...")
        
        # Rolling statistics
        for col in ['ATM_IV', 'ATM_skew']:
            df[f'{col}_zscore'] = df.groupby('expiry')[col].transform(
                lambda x: (x - x.rolling(self.lookback, min_periods=self.min_periods).mean()) / 
                         (x.rolling(self.lookback, min_periods=self.min_periods).std() + 1e-8)
            )
        
        # Handle NaN values (fill with neutral values)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(0)
        
        logger.info(f"✅ Added {len(df.columns) - len(df_svi.columns)} new features")
        
        return df
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for regime features."""
        regime_features = [
            'IV_percentile', 'IV_momentum', 'IV_acceleration',
            'skew_percentile', 'curv_percentile', 'term_slope',
            'IV_vol', 'days_since_high_iv',
            'ATM_IV_zscore', 'ATM_skew_zscore'
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