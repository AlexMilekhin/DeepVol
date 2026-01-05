import pyarrow.parquet as pq
import pandas as pd
import numpy as np

def analyze_data(filepath, title, show_all_rows=False):
    print("=" * 80)
    print(f"{title}")
    print("=" * 80)
    
    table = pq.read_table(filepath)
    df = table.to_pandas()
    
    print(f"\nFile: {filepath}")
    print(f"Number of rows: {table.num_rows:,}")
    print(f"Number of columns: {table.num_columns}")
    print(f"\nColumn names: {list(table.column_names)}")
    
    print("\n" + "-" * 80)
    print("DATA SAMPLE (first 10 rows):")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(df.head(10).to_string())
    
    if len(df) > 10:
        print(f"\n... ({len(df) - 10} more rows)")
    
    print("\n" + "-" * 80)
    print("STATISTICAL SUMMARY:")
    print("-" * 80)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    
    print("\n" + "-" * 80)
    print("DATA QUALITY:")
    print("-" * 80)
    print(f"Total null values: {df.isnull().sum().sum()}")
    print(f"Rows with nulls: {df.isnull().any(axis=1).sum()}")
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("\nNull counts by column:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
    
    print("\n" + "-" * 80)
    print("KEY INSIGHTS:")
    print("-" * 80)
    
    # Analyze based on file type
    if 'options' in filepath.lower() or 'combined' in filepath.lower():
        if 'expiry' in df.columns:
            print(f"Unique expiries: {df['expiry'].nunique()}")
            print(f"Expiry range: {df['expiry'].min()} to {df['expiry'].max()}")
        if 'strike' in df.columns or 'K' in df.columns:
            strike_col = 'strike' if 'strike' in df.columns else 'K'
            print(f"Strike range: ${df[strike_col].min():.2f} to ${df[strike_col].max():.2f}")
        if 'optionType' in df.columns or 'type' in df.columns:
            type_col = 'optionType' if 'optionType' in df.columns else 'type'
            if type_col in df.columns:
                print(f"Call options: {(df[type_col] == 'call').sum()}")
                print(f"Put options: {(df[type_col] == 'put').sum()}")
        if 'iv_clean' in df.columns or 'mid_iv' in df.columns:
            iv_col = 'iv_clean' if 'iv_clean' in df.columns else 'mid_iv'
            valid_iv = df[iv_col].dropna()
            if len(valid_iv) > 0:
                print(f"Valid IV values: {len(valid_iv)} ({len(valid_iv)/len(df)*100:.1f}%)")
                print(f"IV range: {valid_iv.min():.4f} to {valid_iv.max():.4f} ({valid_iv.min()*100:.2f}% to {valid_iv.max()*100:.2f}%)")
                print(f"Mean IV: {valid_iv.mean():.4f} ({valid_iv.mean()*100:.2f}%)")
    
    elif 'hist' in filepath.lower() or 'price' in filepath.lower():
        if 'Close' in df.columns:
            print(f"Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
            print(f"Latest price: ${df['Close'].iloc[-1]:.2f}")
            if len(df) > 1:
                returns = df['Close'].pct_change().dropna()
                print(f"Mean daily return: {returns.mean()*100:.4f}%")
                print(f"Volatility (std): {returns.std()*100:.4f}%")
        if df.index.dtype == 'datetime64[ns]' or 'Date' in df.columns:
            date_col = df.index if df.index.dtype == 'datetime64[ns]' else df['Date']
            print(f"Date range: {date_col.min()} to {date_col.max()}")
            print(f"Trading days: {len(df)}")
    
    print("\n" + "-" * 80)
    print("COLUMN DETAILS:")
    print("-" * 80)
    for i, col_name in enumerate(table.column_names):
        col = table.column(col_name)
        field = table.schema.field(i)
        print(f"\n{col_name}:")
        print(f"  Type: {field.type}")
        print(f"  Nullable: {field.nullable}")
        print(f"  Null count: {col.null_count}")
        if col_name in df.columns:
            if df[col_name].dtype in ['float64', 'int64']:
                valid_vals = df[col_name].dropna()
                if len(valid_vals) > 0:
                    print(f"  Min: {valid_vals.min():.6f}")
                    print(f"  Max: {valid_vals.max():.6f}")
                    print(f"  Mean: {valid_vals.mean():.6f}")
                    print(f"  Std: {valid_vals.std():.6f}")
            elif df[col_name].dtype == 'object':
                unique_count = df[col_name].nunique()
                print(f"  Unique values: {unique_count}")
                if unique_count <= 10:
                    print(f"  Values: {df[col_name].unique().tolist()}")
    
    return df

# Analyze data files
print("\n")
combined_df = analyze_data('data/combined_options_data.parquet', 'COMBINED OPTIONS DATA ANALYSIS')

print("\n\n")

cleaned_df = analyze_data('data/cleaned_options_data.parquet', 'CLEANED OPTIONS DATA ANALYSIS')


print("\n\n")
print("=" * 80)
print("CROSS-FILE ANALYSIS")
print("=" * 80)
print(f"\nCombined options: {len(combined_df):,} rows")
print(f"Cleaned options: {len(cleaned_df):,} rows")
print(f"Data reduction: {len(combined_df) - len(cleaned_df):,} rows removed ({((len(combined_df) - len(cleaned_df))/len(combined_df)*100):.1f}%)")

if 'iv_clean' in cleaned_df.columns:
    valid_iv_pct = (cleaned_df['iv_clean'].notna().sum() / len(cleaned_df)) * 100
    print(f"Valid IV percentage in cleaned data: {valid_iv_pct:.1f}%")

if 'Close' in hist_df.columns:
    latest_price = hist_df['Close'].iloc[-1]
    print(f"\nLatest QQQ price: ${latest_price:.2f}")
    
    if 'strike' in cleaned_df.columns or 'K' in cleaned_df.columns:
        strike_col = 'strike' if 'strike' in cleaned_df.columns else 'K'
        moneyness = cleaned_df[strike_col] / latest_price
        print(f"Strike moneyness range: {moneyness.min():.3f} to {moneyness.max():.3f}")
        print(f"ATM options (0.95-1.05): {(moneyness.between(0.95, 1.05)).sum()}")
