import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import gc
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the preprocessed data
DATA_FILE = Path("data/processed/flights_clean.csv")
OUTPUT_DIR = Path("analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Memory-efficient settings
CHUNK_SIZE = 50000  # Process 50k rows at a time to manage 16GB RAM
MAX_SAMPLE_SIZE = 500000  # Maximum sample size for visualization

def get_data_info():
    """Get basic info about the dataset without loading it all"""
    print("Analyzing dataset structure...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Preprocessed data file not found: {DATA_FILE}")

    # Read just the first chunk to get column info
    sample_chunk = pd.read_csv(DATA_FILE, chunksize=1000).__next__()
    columns = sample_chunk.columns.tolist()

    # Count total rows
    total_rows = sum(1 for line in open(DATA_FILE, encoding='utf-8')) - 1  # subtract header

    print(f"Dataset columns: {columns}")
    print(f"Total rows: {total_rows:,}")
    print(f"Estimated memory requirement: {total_rows * len(columns) * 8 / 1024**3:.1f} GB")

    return total_rows, columns


def load_data(sample_size=None, random_sample=True):
    """Load preprocessed flight data with optional sampling for memory efficiency.

    Args:
        sample_size: Number of rows to load (None for all data)
        random_sample: If True, randomly sample chunks; if False, load first N rows

    Returns:
        pd.DataFrame with loaded data
    """
    print(f"Loading data from {DATA_FILE}...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessed data file not found: {DATA_FILE}\n"
            "Please run preprocess.py first to generate the preprocessed data."
        )

    if sample_size is None:
        # Load all data
        print("Loading entire dataset...")
        df = pd.read_csv(DATA_FILE, low_memory=False)
        print(f"Loaded {len(df):,} rows")
        return df

    # Load a sample
    total_rows, _ = get_data_info()

    if sample_size >= total_rows:
        print(f"Requested sample size ({sample_size:,}) >= total rows ({total_rows:,}), loading all data")
        df = pd.read_csv(DATA_FILE, low_memory=False)
        return df

    if random_sample:
        # Random sampling across chunks
        print(f"Randomly sampling {sample_size:,} rows from {total_rows:,} total rows...")

        num_chunks = int(np.ceil(total_rows / CHUNK_SIZE))
        chunks_needed = max(1, int(np.ceil(sample_size / CHUNK_SIZE)))

        # Randomly select which chunks to load
        selected_chunks = sorted(np.random.choice(num_chunks,
                                                 min(chunks_needed, num_chunks),
                                                 replace=False))

        dfs = []
        loaded_rows = 0

        for i, chunk in enumerate(pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE, low_memory=False)):
            if i in selected_chunks:
                dfs.append(chunk)
                loaded_rows += len(chunk)

                if loaded_rows >= sample_size:
                    break

        df = pd.concat(dfs, ignore_index=True)

        # If we loaded too many rows, sample down to exact size
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        print(f"Loaded {len(df):,} rows")
        return df

    else:
        # Sequential sampling (first N rows)
        print(f"Loading first {sample_size:,} rows...")
        df = pd.read_csv(DATA_FILE, nrows=sample_size, low_memory=False)
        print(f"Loaded {len(df):,} rows")
        return df

def process_chunks(chunk_processor, aggregator=None, sample_size=MAX_SAMPLE_SIZE):
    """Process data in chunks to manage memory"""
    print(f"Processing data in chunks of {CHUNK_SIZE:,} rows...")
    
    # If sampling, determine which chunks to process
    total_rows, _ = get_data_info()
    
    if sample_size and sample_size < total_rows:
        # Randomly sample chunks to process
        num_chunks = int(np.ceil(total_rows / CHUNK_SIZE))
        chunks_needed = int(np.ceil(sample_size / CHUNK_SIZE))
        selected_chunks = sorted(np.random.choice(num_chunks, 
                                               min(chunks_needed, num_chunks), 
                                               replace=False))
        print(f"Sampling {chunks_needed} chunks out of {num_chunks} total chunks")
    else:
        selected_chunks = None
    
    results = []
    processed_rows = 0
    
    for i, chunk in enumerate(pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE)):
        if selected_chunks is not None and i not in selected_chunks:
            continue
            
        # Process this chunk
        result = chunk_processor(chunk)
        if result is not None:
            results.append(result)
        
        processed_rows += len(chunk)
        if processed_rows % (CHUNK_SIZE * 10) == 0:
            print(f"  Processed {processed_rows:,} rows...")
        
        # Force garbage collection to free memory
        del chunk
        gc.collect()
        
        # Stop if we've reached our sample size
        if sample_size and processed_rows >= sample_size:
            break
    
    print(f"Finished processing {processed_rows:,} rows")
    
    # Aggregate results if aggregator is provided
    if aggregator and results:
        return aggregator(results)
    return results

def delay_overview():
    """High-level delay statistics using chunked processing"""
    print("\n" + "="*60)
    print("FLIGHT DELAY OVERVIEW")
    print("="*60)
    
    def process_chunk(chunk):
        """Process a single chunk for delay overview"""
        stats = {
            'total_flights': len(chunk),
            'delayed_flights': chunk['arr_delayed_15'].sum() if 'arr_delayed_15' in chunk.columns else 0,
            'total_delay_minutes': chunk['arr_delay_min'].sum() if 'arr_delay_min' in chunk.columns else 0,
            'delayed_only_sum': chunk[chunk['arr_delay_min'] > 0]['arr_delay_min'].sum() if 'arr_delay_min' in chunk.columns else 0,
            'delayed_only_count': (chunk['arr_delay_min'] > 0).sum() if 'arr_delay_min' in chunk.columns else 0,
            'max_delay': chunk['arr_delay_min'].max() if 'arr_delay_min' in chunk.columns else 0,
            'cancelled': chunk['is_cancelled'].sum() if 'is_cancelled' in chunk.columns else 0
        }
        return stats
    
    def aggregate_results(results):
        """Aggregate chunk results"""
        total_stats = {
            'total_flights': sum(r['total_flights'] for r in results),
            'delayed_flights': sum(r['delayed_flights'] for r in results),
            'total_delay_minutes': sum(r['total_delay_minutes'] for r in results),
            'delayed_only_sum': sum(r['delayed_only_sum'] for r in results),
            'delayed_only_count': sum(r['delayed_only_count'] for r in results),
            'max_delay': max(r['max_delay'] for r in results if r['max_delay'] > 0),
            'cancelled': sum(r['cancelled'] for r in results)
        }
        return total_stats
    
    # Process chunks and aggregate
    stats = process_chunks(process_chunk, aggregate_results)
    
    delay_rate = (stats['delayed_flights'] / stats['total_flights']) * 100 if stats['total_flights'] > 0 else 0
    avg_delay = stats['delayed_only_sum'] / stats['delayed_only_count'] if stats['delayed_only_count'] > 0 else 0
    cancel_rate = (stats['cancelled'] / stats['total_flights']) * 100 if stats['total_flights'] > 0 else 0
    
    print(f"Total flights analyzed: {stats['total_flights']:,}")
    print(f"Flights delayed >15 min: {stats['delayed_flights']:,}")
    print(f"Overall delay rate: {delay_rate:.1f}%")
    
    if avg_delay > 0:
        print(f"Average delay (when delayed): {avg_delay:.1f} minutes")
        print(f"Maximum delay recorded: {stats['max_delay']:.0f} minutes")
    
    if stats['cancelled'] > 0:
        print(f"Cancelled flights: {stats['cancelled']:,} ({cancel_rate:.2f}%)")

def delay_reasons_analysis():
    """Analyze primary delay causes using chunked processing"""
    print("\n" + "="*60)
    print("DELAY REASONS ANALYSIS")
    print("="*60)
    
    def process_chunk(chunk):
        """Process delay reasons for a chunk"""
        if 'primary_delay_reason' not in chunk.columns:
            return None
        
        delayed_only = chunk[chunk['primary_delay_reason'] != 'on_time']
        if len(delayed_only) == 0:
            return None
            
        return delayed_only['primary_delay_reason'].value_counts()
    
    def aggregate_results(results):
        """Aggregate reason counts"""
        total_counts = pd.Series(dtype=int)
        for result in results:
            if result is not None:
                total_counts = total_counts.add(result, fill_value=0)
        return total_counts
    
    reason_counts = process_chunks(process_chunk, aggregate_results, sample_size=MAX_SAMPLE_SIZE)
    
    if len(reason_counts) > 0:
        print("Top delay reasons:")
        total_delayed = reason_counts.sum()
        
        for reason, count in reason_counts.head(10).items():
            pct = (count / total_delayed) * 100
            print(f"  {reason.replace('_', ' ').title()}: {count:,} ({pct:.1f}%)")
        
        # Visualize delay reasons
        plt.figure(figsize=(12, 6))
        reason_counts.head(8).plot(kind='bar')
        plt.title('Primary Flight Delay Reasons')
        plt.xlabel('Delay Reason')
        plt.ylabel('Number of Flights')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'delay_reasons.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory

def delay_by_time_patterns(df):
    """Analyze delay patterns by time of day, day of week, month"""
    print("\n" + "="*60)
    print("DELAY TIME PATTERNS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Delay by hour of day
    if 'dep_hour' in df.columns and 'arr_delayed_15' in df.columns:
        hourly_delays = df.groupby('dep_hour')['arr_delayed_15'].agg(['count', 'sum', 'mean']).reset_index()
        hourly_delays['delay_rate'] = hourly_delays['mean'] * 100
        
        axes[0,0].plot(hourly_delays['dep_hour'], hourly_delays['delay_rate'], marker='o')
        axes[0,0].set_title('Delay Rate by Departure Hour')
        axes[0,0].set_xlabel('Departure Hour')
        axes[0,0].set_ylabel('Delay Rate (%)')
        axes[0,0].grid(True, alpha=0.3)
    
    # Delay by day of week
    if 'dow' in df.columns and 'arr_delayed_15' in df.columns:
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_delays = df.groupby('dow')['arr_delayed_15'].mean() * 100
        
        axes[0,1].bar(range(7), dow_delays.values)
        axes[0,1].set_title('Delay Rate by Day of Week')
        axes[0,1].set_xlabel('Day of Week')
        axes[0,1].set_ylabel('Delay Rate (%)')
        axes[0,1].set_xticks(range(7))
        axes[0,1].set_xticklabels(dow_names)
    
    # Delay by month
    if 'month' in df.columns and 'arr_delayed_15' in df.columns:
        monthly_delays = df.groupby('month')['arr_delayed_15'].mean() * 100
        
        axes[1,0].plot(monthly_delays.index, monthly_delays.values, marker='o')
        axes[1,0].set_title('Delay Rate by Month')
        axes[1,0].set_xlabel('Month')
        axes[1,0].set_ylabel('Delay Rate (%)')
        axes[1,0].set_xticks(range(1, 13))
        axes[1,0].grid(True, alpha=0.3)
    
    # Delay magnitude distribution
    if 'delay_magnitude' in df.columns:
        delay_dist = df['delay_magnitude'].value_counts()
        axes[1,1].pie(delay_dist.values, labels=delay_dist.index, autopct='%1.1f%%')
        axes[1,1].set_title('Distribution of Delay Magnitudes')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'delay_time_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

def airline_delay_comparison(df):
    """Compare delay rates across airlines"""
    print("\n" + "="*60)
    print("AIRLINE DELAY COMPARISON")
    print("="*60)
    
    if 'OP_CARRIER' in df.columns and 'arr_delayed_15' in df.columns:
        # Get airlines with sufficient flights for meaningful analysis
        airline_stats = df.groupby('OP_CARRIER').agg({
            'arr_delayed_15': ['count', 'sum', 'mean'],
            'arr_delay_min': 'mean'
        }).round(2)
        
        airline_stats.columns = ['total_flights', 'delayed_flights', 'delay_rate', 'avg_delay_mins']
        airline_stats['delay_rate'] *= 100
        
        # Filter airlines with >1000 flights
        significant_airlines = airline_stats[airline_stats['total_flights'] >= 1000]
        significant_airlines = significant_airlines.sort_values('delay_rate', ascending=False)
        
        print("Top 10 airlines by delay rate (min 1000 flights):")
        for carrier, row in significant_airlines.head(10).iterrows():
            print(f"  {carrier}: {row['delay_rate']:.1f}% ({row['total_flights']:,} flights)")
        
        # Visualize
        plt.figure(figsize=(14, 8))
        top_carriers = significant_airlines.head(15)
        plt.barh(range(len(top_carriers)), top_carriers['delay_rate'])
        plt.yticks(range(len(top_carriers)), top_carriers.index)
        plt.xlabel('Delay Rate (%)')
        plt.title('Airline Delay Rates (>1000 flights)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'airline_delay_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def airport_delay_analysis(df):
    """Analyze delays by airport"""
    print("\n" + "="*60)
    print("AIRPORT DELAY ANALYSIS")
    print("="*60)
    
    if 'ORIGIN' in df.columns and 'arr_delayed_15' in df.columns:
        # Origin airport delays
        origin_delays = df.groupby('ORIGIN').agg({
            'arr_delayed_15': ['count', 'mean'],
            'dep_delayed_15': 'mean' if 'dep_delayed_15' in df.columns else 'count'
        }).round(3)
        
        origin_delays.columns = ['total_flights', 'arrival_delay_rate', 'departure_delay_rate']
        origin_delays['arrival_delay_rate'] *= 100
        if 'dep_delayed_15' in df.columns:
            origin_delays['departure_delay_rate'] *= 100
        
        # Top airports by volume
        busy_airports = origin_delays[origin_delays['total_flights'] >= 5000].sort_values('total_flights', ascending=False)
        
        print("Top 10 busiest airports - delay rates:")
        for airport, row in busy_airports.head(10).iterrows():
            print(f"  {airport}: {row['arrival_delay_rate']:.1f}% arrival delays ({row['total_flights']:,} flights)")

def seasonal_delay_trends(df):
    """Analyze delay trends over years and seasons"""
    print("\n" + "="*60)
    print("SEASONAL DELAY TRENDS")
    print("="*60)
    
    if 'year' in df.columns and 'arr_delayed_15' in df.columns:
        yearly_trends = df.groupby('year')['arr_delayed_15'].mean() * 100
        
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_trends.index, yearly_trends.values, marker='o', linewidth=2)
        plt.title('Flight Delay Trends Over Years')
        plt.xlabel('Year')
        plt.ylabel('Delay Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'yearly_delay_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Yearly delay trends:")
        for year, rate in yearly_trends.items():
            print(f"  {year}: {rate:.1f}%")

def generate_summary_report(df):
    """Generate comprehensive delay analysis report"""
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)
    
    report = []
    report.append("# Flight Delay Analysis Report")
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    
    # Add key findings
    if 'arr_delayed_15' in df.columns:
        delay_rate = (df['arr_delayed_15'].sum() / len(df)) * 100
        report.append(f"## Key Findings")
        report.append(f"- Overall delay rate: {delay_rate:.1f}%")
        
        if 'primary_delay_reason' in df.columns:
            top_reason = df[df['primary_delay_reason'] != 'on_time']['primary_delay_reason'].mode()
            if len(top_reason) > 0:
                report.append(f"- Most common delay cause: {top_reason.iloc[0].replace('_', ' ').title()}")
    
    # Save report
    report_path = OUTPUT_DIR / 'delay_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to: {report_path}")

def main():
    """Run complete delay analysis"""
    print("🛫 FLIGHT DELAY ANALYSIS STARTING...")

    # Check if preprocessed data exists
    if not DATA_FILE.exists():
        print(f"\n❌ Error: Preprocessed data file not found: {DATA_FILE}")
        print("Please run preprocess.py first to generate the preprocessed data.")
        return

    # Run chunk-based analyses (don't require loading all data)
    delay_overview()
    delay_reasons_analysis()

    # Load sample for detailed analyses (memory-efficient)
    print("\nLoading sample data for detailed analyses...")
    df = load_data(sample_size=100000)  # Adjust or remove for full dataset

    # Run analyses that need loaded data
    delay_by_time_patterns(df)
    airline_delay_comparison(df)
    airport_delay_analysis(df)
    seasonal_delay_trends(df)
    generate_summary_report(df)

    print("\n🎯 Analysis complete! Check the 'analysis' folder for visualizations.")
    print(f"Generated files:")
    for file in OUTPUT_DIR.glob("*.png"):
        print(f"  - {file.name}")
    if (OUTPUT_DIR / "delay_analysis_report.md").exists():
        print(f"  - delay_analysis_report.md")

if __name__ == "__main__":
    main()