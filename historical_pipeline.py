"""
Historical Volatility Surface Pipeline

Processes historical options data through the existing volatility surface models.
Uses the same src.models components as main.py but iterates over historical dates.

Usage:
    python historical_pipeline.py --file {file_name} --ticker {TICKER} --test
    python historical_pipeline.py --file {file_name} --ticker {TICKER} --output-dir ./results
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from existing codebase
try:
    from src.models.bs_engine import ImpliedVolatilityCalculator
    from src.models.svi_fit import SVIFitter, SSVIFitter
    from src.features import RNDFeatureExtractor
    IMPORTS_AVAILABLE = True
    logger.info("Successfully imported src.models components")
except ImportError as e:
    logger.warning(f"Could not import src.models: {e}")
    IMPORTS_AVAILABLE = False


@dataclass
class ColumnMapping:
    """Maps historical data columns to standard pipeline column names."""
    date: str = 'date'
    expiry: str = 'expiration'
    strike: str = 'strike'
    option_type: str = 'call_put'
    bid: str = 'bid'
    ask: str = 'ask'
    volume: str = 'vol'
    
    @classmethod
    def auto_detect(cls, df: pd.DataFrame) -> 'ColumnMapping':
        """Auto-detect column mapping from dataframe."""
        mapping = cls()
        
        for col in ['date', 'quote_date', 'trade_date', 'asof']:
            if col in df.columns:
                mapping.date = col
                break
        
        for col in ['expiration', 'expiry', 'expiry_date']:
            if col in df.columns:
                mapping.expiry = col
                break
        
        for col in ['strike', 'strike_price', 'K']:
            if col in df.columns:
                mapping.strike = col
                break
        
        for col in ['call_put', 'option_type', 'type', 'cp_flag']:
            if col in df.columns:
                mapping.option_type = col
                break
        
        for col in ['vol', 'volume']:
            if col in df.columns:
                mapping.volume = col
                break
        
        return mapping


class HistoricalDataLoader:
    """Loads historical options data and fetches underlying prices/rates from Yahoo."""
    
    def __init__(self, filepath: str, ticker: str):
        self.filepath = Path(filepath)
        self.ticker = ticker
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading historical data from {filepath}...")
        self.df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(self.df):,} records")
        
        self.mapping = ColumnMapping.auto_detect(self.df)
        logger.info(f"Column mapping: date={self.mapping.date}, expiry={self.mapping.expiry}")
        
        # Convert date column
        self.df['_date'] = pd.to_datetime(self.df[self.mapping.date])
        
        # Fetch underlying prices and rates
        self._fetch_underlying_prices()
        self._fetch_historical_rates()
        
        # Standardize columns
        self._standardize_columns()
        
        # Cache unique dates
        self._dates = sorted(self.df['asof'].dt.date.unique())
        logger.info(f"Found {len(self._dates)} unique trading days "
                   f"from {self._dates[0]} to {self._dates[-1]}")
    
    def _fetch_underlying_prices(self):
        """Fetch historical underlying prices from Yahoo Finance."""
        import yfinance as yf
        
        min_date = self.df['_date'].min() - timedelta(days=5)
        max_date = self.df['_date'].max() + timedelta(days=5)
        
        logger.info(f"Fetching {self.ticker} prices from Yahoo Finance...")
        
        hist = yf.Ticker(self.ticker).history(
            start=min_date.strftime('%Y-%m-%d'),
            end=max_date.strftime('%Y-%m-%d')
        )
        
        if len(hist) == 0:
            raise ValueError(f"No price data returned for {self.ticker}")
        
        logger.info(f"Fetched {len(hist)} days of price data")
        
        hist = hist.reset_index()
        hist['_date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None).dt.normalize()
        hist = hist[['_date', 'Close']].rename(columns={'Close': 'underlying_price'})
        
        self.df['_date'] = pd.to_datetime(self.df['_date']).dt.normalize()
        self.df = self.df.merge(hist, on='_date', how='left')
        
        # Forward/back fill for weekends
        missing = self.df['underlying_price'].isna().sum()
        if missing > 0:
            logger.warning(f"{missing:,} records missing price - applying fill...")
            self.df = self.df.sort_values('_date')
            self.df['underlying_price'] = self.df['underlying_price'].ffill().bfill()
        
        logger.info(f"Underlying prices: ${self.df['underlying_price'].min():.2f} - "
                   f"${self.df['underlying_price'].max():.2f}")
    
    def _fetch_historical_rates(self):
        """Fetch historical risk-free rates from Yahoo Finance."""
        import yfinance as yf
        
        min_date = self.df['_date'].min() - timedelta(days=10)
        max_date = self.df['_date'].max() + timedelta(days=5)
        
        logger.info(f"Fetching historical risk-free rates (^IRX)...")
        
        try:
            irx = yf.Ticker("^IRX").history(
                start=min_date.strftime('%Y-%m-%d'),
                end=max_date.strftime('%Y-%m-%d')
            )
            
            if len(irx) == 0:
                logger.warning("No ^IRX data, using default rate")
                self._rates_df = None
                return
            
            irx = irx.reset_index()
            irx['_date'] = pd.to_datetime(irx['Date']).dt.tz_localize(None).dt.normalize()
            irx['rate'] = irx['Close'] / 100.0
            self._rates_df = irx[['_date', 'rate']].sort_values('_date')
            
            logger.info(f"Rates: {self._rates_df['rate'].min()*100:.2f}% - "
                       f"{self._rates_df['rate'].max()*100:.2f}%")
        except Exception as e:
            logger.warning(f"Failed to fetch rates: {e}")
            self._rates_df = None
    
    def _standardize_columns(self):
        """Standardize column names."""
        self.df['asof'] = pd.to_datetime(self.df[self.mapping.date])
        self.df['expiry'] = pd.to_datetime(self.df[self.mapping.expiry])
        self.df['strike'] = pd.to_numeric(self.df[self.mapping.strike], errors='coerce')
        
        type_col = self.df[self.mapping.option_type].astype(str).str.lower()
        self.df['type'] = np.where(type_col.str.contains('c'), 'call',
                          np.where(type_col.str.contains('p'), 'put', type_col))
        
        if self.mapping.bid in self.df.columns:
            self.df['bid'] = pd.to_numeric(self.df[self.mapping.bid], errors='coerce')
        if self.mapping.ask in self.df.columns:
            self.df['ask'] = pd.to_numeric(self.df[self.mapping.ask], errors='coerce')
        
        if 'mid' not in self.df.columns and 'bid' in self.df.columns:
            self.df['mid'] = (self.df['bid'] + self.df['ask']) / 2
        
        self.df['S'] = pd.to_numeric(self.df['underlying_price'], errors='coerce')
        
        if self.mapping.volume in self.df.columns:
            self.df['volume'] = pd.to_numeric(self.df[self.mapping.volume], errors='coerce')
        
        # Calculate T from asof date (not today!)
        self.df['T'] = (self.df['expiry'] - self.df['asof']).dt.days / 365.0
        self.df = self.df[self.df['T'] > 0].copy()
        
        if 'S' in self.df.columns:
            self.df['k'] = np.log(self.df['strike'] / self.df['S'])
        
        if '_date' in self.df.columns:
            self.df.drop(columns=['_date'], inplace=True)
    
    def get_dates(self) -> List[date]:
        return self._dates.copy()
    
    def get_snapshot(self, asof_date: date) -> pd.DataFrame:
        mask = self.df['asof'].dt.date == asof_date
        return self.df.loc[mask].copy()
    
    def get_spot_price(self, asof_date: date) -> Optional[float]:
        snapshot = self.get_snapshot(asof_date)
        if 'S' in snapshot.columns and snapshot['S'].notna().any():
            return float(snapshot['S'].iloc[0])
        return None
    
    def get_rate(self, asof_date: date) -> float:
        if self._rates_df is None:
            return 0.045
        target = pd.Timestamp(asof_date).normalize()
        mask = self._rates_df['_date'] <= target
        if mask.any():
            return float(self._rates_df.loc[mask, 'rate'].iloc[-1])
        return float(self._rates_df['rate'].iloc[0])
    
    def summary(self) -> Dict[str, Any]:
        return {
            'total_records': len(self.df),
            'unique_dates': len(self._dates),
            'date_range': (self._dates[0], self._dates[-1]),
            'spot_range': (self.df['S'].min(), self.df['S'].max()),
            'strike_range': (self.df['strike'].min(), self.df['strike'].max()),
            'call_count': (self.df['type'] == 'call').sum(),
            'put_count': (self.df['type'] == 'put').sum(),
        }


class HistoricalVolatilitySurfacePipeline:
    """Pipeline for processing historical options data."""
    
    def __init__(self, filepath: str, ticker: str, output_dir: str = "historical_results",
                 min_options_per_expiry: int = 5):
        self.ticker = ticker
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.min_options = min_options_per_expiry
        
        self.data_loader = HistoricalDataLoader(filepath, ticker)
        
        if IMPORTS_AVAILABLE:
            self.iv_calculator = ImpliedVolatilityCalculator(ticker=ticker)
            self.svi_fitter = SVIFitter()
            self.ssvi_fitter = SSVIFitter()
            self.rnd_extractor = RNDFeatureExtractor()
        else:
            raise RuntimeError("src.models not available - install package first")
        
        self.svi_results = []
        self.ssvi_results = []
        self.rnd_results = []
        self.errors = []
    
    def process_single_date(self, asof_date: date, debug: bool = False) -> Dict[str, Any]:
        """Process a single date through the full pipeline."""
        result = {'asof_date': asof_date, 'svi': [], 'ssvi': [], 'rnd': [], 
                  'error': None, 'n_options': 0, 'n_expiries': 0}
        
        try:
            snapshot = self.data_loader.get_snapshot(asof_date)
            result['n_options'] = len(snapshot)
            
            if len(snapshot) < 10:
                result['error'] = f"Insufficient data: {len(snapshot)} options"
                return result
            
            S = self.data_loader.get_spot_price(asof_date)
            if S is None or not np.isfinite(S):
                result['error'] = "No spot price"
                return result
            
            rate = self.data_loader.get_rate(asof_date)
            
            if debug:
                print(f"      [DEBUG] Options: {len(snapshot)}, Spot: ${S:.2f}, Rate: {rate*100:.2f}%")
            
            clean_df = self.iv_calculator.recompute_clean_iv_surface(
                snapshot, r=rate, asof_date=asof_date
            )
            
            if debug:
                print(f"      [DEBUG] After cleaning: {len(clean_df)} options")
            
            if len(clean_df) < 10:
                result['error'] = f"Insufficient clean data: {len(clean_df)} options"
                return result
            
            # Fit SVI
            svi_rows = self._fit_svi(clean_df, asof_date)
            result['svi'] = svi_rows
            result['n_expiries'] = len(svi_rows)
            
            # Fit SSVI
            if len(svi_rows) >= 2:
                ssvi_rows = self._fit_ssvi(clean_df, asof_date, S, rate)
                result['ssvi'] = ssvi_rows
                
                if ssvi_rows:
                    rnd_rows = self._extract_rnd(ssvi_rows, S, rate, asof_date)
                    result['rnd'] = rnd_rows
            
        except Exception as e:
            result['error'] = str(e)
            if debug:
                import traceback
                print(f"      [DEBUG] Error: {traceback.format_exc()}")
        
        return result
    
    def _fit_svi(self, clean_df: pd.DataFrame, asof_date: date) -> List[Dict]:
        rows = []
        for expiry, df_exp in clean_df.groupby("expiry"):
            if len(df_exp) < self.min_options:
                continue
            
            params, _ = self.svi_fitter.fit(df_exp)
            if params is not None:
                T = float(df_exp["T"].iloc[0])
                iv_atm, skew_atm, curv_atm = SVIFitter.atm_metrics_from_params(params, T)
                rows.append({
                    'asof': asof_date, 'expiry': expiry, 'T': T,
                    'ATM_IV': iv_atm, 'ATM_skew': skew_atm, 'ATM_curvature': curv_atm,
                    'a': params.a, 'b': params.b, 'rho': params.rho,
                    'm': params.m, 'sigma': params.sigma, 'n_options': len(df_exp),
                })
        return rows
    
    def _fit_ssvi(self, clean_df: pd.DataFrame, asof_date: date, S: float, rate: float) -> List[Dict]:
        rows = []
        try:
            clean_df = clean_df.copy()
            forwards, thetas = {}, {}
            
            for exp, g in clean_df.groupby('expiry'):
                if len(g) < self.min_options:
                    continue
                F = self.iv_calculator.infer_forward_from_parity(g, rate)
                if np.isfinite(F):
                    forwards[exp] = F
                    thetas[exp] = self.ssvi_fitter.estimate_theta_atm(g, F)
            
            if len(thetas) < 2:
                return rows
            
            thetas = self.ssvi_fitter.enforce_monotone_theta(thetas, pd.Timestamp(asof_date))
            
            clean_df['F'] = clean_df['expiry'].map(forwards)
            clean_df['k_fwd'] = np.log(clean_df['strike'] / clean_df['F'])
            if 'mid_iv' not in clean_df.columns:
                clean_df['mid_iv'] = clean_df['iv_clean']
            
            slice_params = self.ssvi_fitter.fit_global(clean_df, thetas, forwards, pd.Timestamp(asof_date))
            
            for exp, (params, cost) in slice_params.items():
                rows.append({
                    'asof': asof_date, 'expiry': exp,
                    'T': float(clean_df[clean_df['expiry'] == exp]['T'].iloc[0]),
                    'theta': thetas[exp], 'ssvi_rho': params.rho,
                    'ssvi_eta': params.eta, 'ssvi_p': params.gamma, 'fit_cost': cost,
                })
        except Exception as e:
            logger.warning(f"SSVI fitting failed for {asof_date}: {e}")
        return rows
    
    def _extract_rnd(self, ssvi_rows: List[Dict], S: float, rate: float, asof_date: date) -> List[Dict]:
        rows = []
        try:
            ssvi_df = pd.DataFrame(ssvi_rows)
            rnd_df = self.rnd_extractor.extract_features(ssvi_df, S, rate)
            for _, row in rnd_df.iterrows():
                rows.append({
                    'asof': asof_date, 'expiry': row['expiry'], 'T': row['T'],
                    'F': row['F'], 'rnd_mean': row['rnd_mean'], 'rnd_vol': row['rnd_vol'],
                    'rnd_skew': row['rnd_skew'], 'rnd_kurtosis': row['rnd_kurtosis'],
                    'martingale_error': row['martingale_error'],
                })
        except Exception as e:
            logger.warning(f"RND extraction failed: {e}")
        return rows
    
    def run_test(self, n_dates: int = 5) -> bool:
        """Test pipeline on a small subset."""
        print("\n" + "=" * 70)
        print(" HISTORICAL PIPELINE TEST")
        print("=" * 70)
        
        summary = self.data_loader.summary()
        print(f"\n✅ Data loaded: {summary['total_records']:,} records, "
              f"{summary['unique_dates']} dates")
        print(f"   Spot: ${summary['spot_range'][0]:.2f} - ${summary['spot_range'][1]:.2f}")
        print(f"   Strikes: ${summary['strike_range'][0]:.2f} - ${summary['strike_range'][1]:.2f}")
        
        dates = self.data_loader.get_dates()[:n_dates]
        print(f"\n🔬 Testing on {len(dates)} dates...")
        print("-" * 70)
        
        success = 0
        for i, d in enumerate(dates):
            print(f"\n   [{i+1}/{len(dates)}] {d}...", end=" ")
            result = self.process_single_date(d, debug=True)
            
            if result['error']:
                print(f"❌ {result['error']}")
            else:
                print(f"✅ {result['n_expiries']} expiries")
                success += 1
                self.svi_results.extend(result['svi'])
                self.ssvi_results.extend(result['ssvi'])
                self.rnd_results.extend(result['rnd'])
        
        print(f"\n" + "-" * 70)
        print(f"   Results: {success}/{len(dates)} successful")
        print(f"   SVI fits: {len(self.svi_results)}, SSVI: {len(self.ssvi_results)}, RND: {len(self.rnd_results)}")
        print("=" * 70)
        
        return success > 0
    
    def run_full(self, start_date: Optional[date] = None, end_date: Optional[date] = None,
                 checkpoint: int = 50):
        """Run full historical pipeline."""
        dates = self.data_loader.get_dates()
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        
        logger.info(f"Processing {len(dates)} dates...")
        
        for i, d in enumerate(dates):
            if (i + 1) % 20 == 0:
                logger.info(f"Progress: {i+1}/{len(dates)} ({d})")
            
            result = self.process_single_date(d)
            self.svi_results.extend(result['svi'])
            self.ssvi_results.extend(result['ssvi'])
            self.rnd_results.extend(result['rnd'])
            if result['error']:
                self.errors.append({'asof': d, 'error': result['error']})
            
            if (i + 1) % checkpoint == 0:
                self._save_checkpoint(i + 1)
        
        self._save_final()
        self._print_summary(len(dates))
    
    def _save_checkpoint(self, n: int):
        if self.svi_results:
            pd.DataFrame(self.svi_results).to_parquet(
                self.output_dir / f"svi_checkpoint_{n}.parquet")
    
    def _save_final(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.svi_results:
            pd.DataFrame(self.svi_results).to_parquet(
                self.output_dir / f"historical_svi_{self.ticker}_{ts}.parquet")
        if self.ssvi_results:
            pd.DataFrame(self.ssvi_results).to_parquet(
                self.output_dir / f"historical_ssvi_{self.ticker}_{ts}.parquet")
        if self.rnd_results:
            pd.DataFrame(self.rnd_results).to_parquet(
                self.output_dir / f"historical_rnd_{self.ticker}_{ts}.parquet")
        if self.errors:
            pd.DataFrame(self.errors).to_parquet(
                self.output_dir / f"errors_{self.ticker}_{ts}.parquet")
    
    def _print_summary(self, total: int):
        success = total - len(self.errors)
        print(f"\n{'='*70}")
        print(f" SUMMARY: {success}/{total} dates ({success/total*100:.1f}%)")
        print(f" SVI: {len(self.svi_results)}, SSVI: {len(self.ssvi_results)}, RND: {len(self.rnd_results)}")
        print(f" Output: {self.output_dir}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Historical Volatility Surface Pipeline')
    parser.add_argument('--file', type=str, required=True, help='Parquet file path')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol')
    parser.add_argument('--output-dir', type=str, default='historical_results')
    parser.add_argument('--test', action='store_true', help='Test mode (5 dates)')
    parser.add_argument('--test-dates', type=int, default=5)
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    
    args = parser.parse_args()
    
    pipeline = HistoricalVolatilitySurfacePipeline(
        filepath=args.file, ticker=args.ticker, output_dir=args.output_dir
    )
    
    if args.test:
        pipeline.run_test(n_dates=args.test_dates)
    else:
        start = datetime.strptime(args.start_date, '%Y-%m-%d').date() if args.start_date else None
        end = datetime.strptime(args.end_date, '%Y-%m-%d').date() if args.end_date else None
        pipeline.run_full(start_date=start, end_date=end)


if __name__ == "__main__":
    main()