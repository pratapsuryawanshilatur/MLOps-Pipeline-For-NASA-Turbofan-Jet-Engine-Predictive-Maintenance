import pandas as pd
import numpy as np
import os

def load_data(file_path, column_names=None):
    """
    Load NASA turbofan data from text file
    """
    print(f"Loading data from {file_path}")
    
    # Read the space-separated text file
    df = pd.read_csv(file_path, sep='\\s+', header=None)
    
    # Add column names if provided
    if column_names and len(column_names) == df.shape[1]:
        df.columns = column_names
    
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df

def create_columns():
    """
    Create meaningful column names for the NASA dataset
    """
    columns = ['engine_id', 'time_cycles']
    
    # Operational settings
    columns.extend([f'op_setting_{i}' for i in range(1, 4)])
    
    # Sensor measurements
    columns.extend([f'sensor_{i:02d}' for i in range(1, 22)])
    
    return columns

def calculate_rul(df, group_by_column='engine_id'):
    """
    Calculate Remaining Useful Life for each engine
    RUL = maximum cycles - current cycle for each engine
    """
    print("Calculating RUL...")
    
    # Group by engine and find max cycle for each
    max_cycle = df.groupby(group_by_column)['time_cycles'].transform('max')
    
    # RUL = max cycle - current cycle
    df['RUL'] = max_cycle - df['time_cycles']
    
    print(f"RUL range: {df['RUL'].min()} to {df['RUL'].max()} cycles")
    return df

def create_binary_target(df, threshold=30):
    """
    Create a binary classification target:
    1 if engine will fail within next 'threshold' cycles
    0 otherwise
    """
    print(f"Creating binary target (failure within {threshold} cycles)")
    df['failure_soon'] = (df['RUL'] <= threshold).astype(int)
    
    # Check class distribution
    failure_count = df['failure_soon'].sum()
    print(f"Engines about to fail: {failure_count} ({failure_count/len(df)*100:.2f}%)")
    
    return df

def save_processed_data(df, output_path):
    """
    Save processed data to Parquet format (efficient storage)
    """
    print(f"Saving processed data to {output_path}")
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    
    # Also save a sample for testing
    sample_path = output_path.replace('.parquet', '_sample.parquet')
    df.sample(min(1000, len(df))).to_parquet(sample_path, index=False)
    print(f"Saved sample to {sample_path}")

def main():
    """
    Main function to process the data
    """
    print("Starting NASA Turbofan Data Processing...")
    
    # Define column names
    columns = create_columns()
    
    # For now, use a sample - we'll make this configurable later
    # Copy your train_FD001.txt to data/raw/train_FD001.txt first
    raw_data_path = "data/raw/train_FD001.txt"
    
    if not os.path.exists(raw_data_path):
        print(f"ERROR: Please copy train_FD001.txt to {raw_data_path}")
        print("The data should be in your downloaded zip file")
        return
    
    # Load the data
    df = load_data(raw_data_path, columns)
    
    # Calculate RUL
    df = calculate_rul(df)
    
    # Create binary target (will fail in next 30 cycles?)
    df = create_binary_target(df, threshold=30)
    
    # Save processed data
    output_path = "data/processed/turbofan_train_FD001.parquet"
    save_processed_data(df, output_path)
    
    print("Data processing completed successfully!")

if __name__ == "__main__":
    main()