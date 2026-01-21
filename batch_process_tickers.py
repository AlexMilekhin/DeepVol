"""
Batch processing script for historical options pipeline
Processes multiple tickers sequentially with error handling and logging
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

TICKERS = [
    "HUM", "ACN", "V", "LRCX", "BLK", "MA", "NOW", "INTU", "MSFT", "NVDA",
    "AVGO", "BIIB", "REGN", "DE", "DIA", "UNP", "STZ", "URI", "SPGI", "CHTR",
    "LLY", "VRTX", "UNH", "CMG", "MCD", "AMZN", "BA", "MCK", "ALGN", "GOOG",
    "AAPL", "NOC", "ADSK", "CI", "CAT", "DPZ", "SPY", "RH", "FDX", "NSC",
    "ADP", "CRM", "XOP", "LOW", "GD", "ANET", "XLY", "XLK", "HON", "KRE",
    "BRK.B", "KLAC", "TSCO", "TMO", "XLF", "XLE", "HSY", "IBM", "JPM", "XBI",
    "DG", "AXP", "ADI", "MAR", "BAC", "ILMN", "XRT", "TXN", "PYPL", "TTWO",
    "TSLA", "UPS", "TGT", "PNC", "PEP", "GM", "UAL", "JNJ", "EOG", "XLV",
    "XHB", "AMAT", "WMT", "XLI", "FSLR", "FCX", "ABBV"
]

class BatchProcessor:
    def __init__(self, output_dir="./results", log_dir="./batch_logs"):
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"batch_run_{timestamp}.log"
        self.status_file = self.log_dir / f"batch_status_{timestamp}.json"
        
        self.results = {
            "start_time": datetime.now().isoformat(),
            "tickers": {},
            "summary": {
                "total": len(TICKERS),
                "completed": 0,
                "failed": 0,
                "skipped": 0
            }
        }
    
    def log(self, message):
        """Write to both console and log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, "a", encoding="UTF-8", errors="replace") as f:
            f.write(log_message + "\n")
    
    def save_status(self):
        """Save current status to JSON file"""
        with open(self.status_file, "w", encoding="UTF-8", errors="replace") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    def process_ticker(self, ticker):
        """Process a single ticker"""
        self.log(f"\n{'='*60}")
        self.log(f"Processing {ticker} ({self.results['summary']['completed'] + 1}/{len(TICKERS)})")
        self.log(f"{'='*60}")
        
        # Construct input file path
        input_file = Path("data_loading") / f"{ticker.lower()}_full_history.parquet"
        
        # Check if input file exists
        if not input_file.exists():
            self.log(f"WARNING: Input file not found: {input_file}")
            self.log(f"Skipping {ticker}")
            self.results["tickers"][ticker] = {
                "status": "skipped",
                "reason": "input_file_not_found",
                "input_file": str(input_file)
            }
            self.results["summary"]["skipped"] += 1
            self.save_status()
            return False
        
        # Construct command
        cmd = [
            "python",
            "historical_pipeline.py",
            "--file", str(input_file),
            "--ticker", ticker,
            "--output-dir", str(self.output_dir)
        ]
        
        self.log(f"Command: {' '.join(cmd)}")
        
        start_time = datetime.now()
        
        try:
            # Run the pipeline
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result.returncode == 0:
                self.log(f"[OK] SUCCESS: {ticker} completed in {duration:.1f}s")
                self.results["tickers"][ticker] = {
                    "status": "success",
                    "duration_seconds": duration,
                    "end_time": end_time.isoformat()
                }
                self.results["summary"]["completed"] += 1
                self.save_status()
                return True
            else:
                self.log(f"[FAIL] FAILED: {ticker} (exit code {result.returncode})")
                self.log(f"STDERR: {result.stderr[:500]}")  
                self.results["tickers"][ticker] = {
                    "status": "failed",
                    "duration_seconds": duration,
                    "exit_code": result.returncode,
                    "error": result.stderr[:1000] 
                }
                self.results["summary"]["failed"] += 1
                self.save_status()
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"[FAIL] TIMEOUT: {ticker} exceeded 1 hour limit")
            self.results["tickers"][ticker] = {
                "status": "timeout",
                "duration_seconds": 3600
            }
            self.results["summary"]["failed"] += 1
            self.save_status()
            return False
            
        except Exception as e:
            self.log(f"[FAIL] ERROR: {ticker} - {str(e)}")
            self.results["tickers"][ticker] = {
                "status": "error",
                "error": str(e)
            }
            self.results["summary"]["failed"] += 1
            self.save_status()
            return False
    
    def run(self):
        """Run the batch processing"""
        self.log("="*60)
        self.log("BATCH PROCESSING STARTED")
        self.log(f"Total tickers to process: {len(TICKERS)}")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Log file: {self.log_file}")
        self.log(f"Status file: {self.status_file}")
        self.log("="*60)
        
        for ticker in TICKERS:
            self.process_ticker(ticker)
        
        # Final summary
        self.results["end_time"] = datetime.now().isoformat()
        total_duration = (
            datetime.fromisoformat(self.results["end_time"]) - 
            datetime.fromisoformat(self.results["start_time"])
        ).total_seconds()
        
        self.log("\n" + "="*60)
        self.log("BATCH PROCESSING COMPLETED")
        self.log("="*60)
        self.log(f"Total time: {total_duration/3600:.2f} hours")
        self.log(f"Completed: {self.results['summary']['completed']}")
        self.log(f"Failed: {self.results['summary']['failed']}")
        self.log(f"Skipped: {self.results['summary']['skipped']}")
        self.log(f"Success rate: {self.results['summary']['completed']/len(TICKERS)*100:.1f}%")
        self.log("="*60)
        
        self.save_status()
        
        return self.results["summary"]["failed"] == 0


if __name__ == "__main__":
    processor = BatchProcessor()
    success = processor.run()
    sys.exit(0 if success else 1)