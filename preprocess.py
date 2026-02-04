import math
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("Original dataset")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "flights_clean.csv"
QUALITY_REPORT_FILE = OUTPUT_DIR / "data_quality_report.txt"

# Keep ALL features needed to explain and predict delays - comprehensive delay analysis.
BASE_COLS: List[str] = [
    "YEAR",
    "QUARTER",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER",
    "OP_CARRIER_FL_NUM",
    "TAIL_NUM",
    "ORIGIN",
    "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_ABR",
    "DEST",
    "DEST_CITY_NAME",
    "DEST_STATE_ABR",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "DEP_DELAY_NEW",
    "DEP_DEL15",
    "DEP_DELAY_GROUP",
    "DEP_TIME_BLK",
    "TAXI_OUT",
    "WHEELS_OFF",
    "WHEELS_ON",
    "TAXI_IN",
    "CRS_ARR_TIME",
    "ARR_TIME",
    "ARR_DELAY",
    "ARR_DELAY_NEW",
    "ARR_DEL15",
    "ARR_DELAY_GROUP",
    "ARR_TIME_BLK",
    "CANCELLED",
    "CANCELLATION_CODE",
    "DIVERTED",
    "CRS_ELAPSED_TIME",
    "ACTUAL_ELAPSED_TIME",
    "AIR_TIME",
    "FLIGHTS",
    "DISTANCE",
    "DISTANCE_GROUP",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
    # Additional delay analysis columns
    "FIRST_DEP_TIME",
    "TOTAL_ADD_GTIME",
    "LONGEST_ADD_GTIME",
]

# Light dtypes to keep memory under control; missing columns are handled gracefully.
DTYPES: Dict[str, str] = {
    "YEAR": "float64",
    "QUARTER": "float64", 
    "MONTH": "float64",
    "DAY_OF_MONTH": "float64",
    "DAY_OF_WEEK": "float64",
    "OP_UNIQUE_CARRIER": "string",
    "OP_CARRIER": "string",
    "OP_CARRIER_FL_NUM": "float64",
    "TAIL_NUM": "string",
    "ORIGIN": "string",
    "ORIGIN_CITY_NAME": "string",
    "DEST": "string",
    "DEST_CITY_NAME": "string",
    "CRS_DEP_TIME": "float64",
    "DEP_TIME": "float64",
    "DEP_DELAY": "float64",
    "DEP_DELAY_NEW": "float64",
    "DEP_DEL15": "float64",
    "DEP_TIME_BLK": "string",
    "TAXI_OUT": "float64",
    "WHEELS_OFF": "float64",
    "WHEELS_ON": "float64",
    "TAXI_IN": "float64",
    "CRS_ARR_TIME": "float64",
    "ARR_TIME": "float64",
    "ARR_DELAY": "float64",
    "ARR_DELAY_NEW": "float64",
    "ARR_DEL15": "float64",
    "ARR_TIME_BLK": "string",
    "CANCELLED": "float64",
    "DIVERTED": "float64",
    "CRS_ELAPSED_TIME": "float64",
    "ACTUAL_ELAPSED_TIME": "float64",
    "AIR_TIME": "float64",
    "FLIGHTS": "float64",
    "DISTANCE": "float64",
    "DISTANCE_GROUP": "float64",
    "CARRIER_DELAY": "float64",
    "WEATHER_DELAY": "float64",
    "NAS_DELAY": "float64",
    "SECURITY_DELAY": "float64",
    "LATE_AIRCRAFT_DELAY": "float64",
    "ORIGIN_STATE_ABR": "string",
    "DEST_STATE_ABR": "string",
    "DEP_DELAY_GROUP": "float64",
    "ARR_DELAY_GROUP": "float64",
    "CANCELLATION_CODE": "string",
    "FIRST_DEP_TIME": "float64",
    "TOTAL_ADD_GTIME": "float64",
    "LONGEST_ADD_GTIME": "float64",
}


def hhmm_to_minutes(hhmm: pd.Series) -> pd.Series:
    """Convert HHMM integers to minutes since midnight.

    Handles missing values and validates time format.
    Returns float64 series with NaN for invalid/missing values.
    """
    def _convert(val: Optional[float]) -> Optional[float]:
        if pd.isna(val):
            return math.nan
        try:
            v = int(val)
            hh, mm = divmod(v, 100)
            # Validate time ranges
            if hh < 0 or hh > 23 or mm < 0 or mm > 59:
                return math.nan
            return float(hh * 60 + mm)
        except (ValueError, TypeError):
            return math.nan

    return hhmm.map(_convert).astype("float64")


def validate_and_clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Validate data quality and clean common issues.

    Returns:
        Tuple of (cleaned_dataframe, quality_metrics)
    """
    quality_metrics = {
        'rows_input': len(df),
        'rows_output': 0,
        'missing_critical_fields': 0,
        'invalid_delays': 0,
        'invalid_times': 0,
        'invalid_distances': 0,
        'rows_dropped': 0
    }

    initial_rows = len(df)

    # Remove rows with missing critical identifiers
    critical_cols = ['FL_DATE', 'ORIGIN', 'DEST']
    missing_critical = df[critical_cols].isna().any(axis=1)
    quality_metrics['missing_critical_fields'] = missing_critical.sum()
    df = df[~missing_critical].copy()

    # Validate and clean delay values
    delay_cols = ['ARR_DELAY', 'DEP_DELAY']
    for col in delay_cols:
        if col in df.columns:
            # Flag extreme outliers (>24 hours delay seems unrealistic for analysis)
            invalid_delays = (df[col].notna()) & ((df[col] < -180) | (df[col] > 1440))
            quality_metrics['invalid_delays'] += invalid_delays.sum()
            # Keep but can be filtered later based on analysis needs

    # Validate distance values
    if 'DISTANCE' in df.columns:
        invalid_dist = (df['DISTANCE'].notna()) & ((df['DISTANCE'] <= 0) | (df['DISTANCE'] > 5000))
        quality_metrics['invalid_distances'] = invalid_dist.sum()
        df.loc[invalid_dist, 'DISTANCE'] = np.nan

    # Validate time values (should be 0-2359)
    time_cols = ['CRS_DEP_TIME', 'DEP_TIME', 'CRS_ARR_TIME', 'ARR_TIME']
    for col in time_cols:
        if col in df.columns:
            invalid_times = (df[col].notna()) & ((df[col] < 0) | (df[col] > 2359))
            quality_metrics['invalid_times'] += invalid_times.sum()
            df.loc[invalid_times, col] = np.nan

    quality_metrics['rows_output'] = len(df)
    quality_metrics['rows_dropped'] = initial_rows - len(df)

    return df, quality_metrics


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for analysis and modeling.

    Handles missing values gracefully and adds comprehensive delay analysis features.
    """
    # Basic temporal encodings
    if "FL_DATE" in df.columns:
        # Parse date - try common formats first for speed
        try:
            # Try ISO format first (fastest)
            df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format='ISO8601', errors='coerce')
            if df["FL_DATE"].isna().all():
                # Fall back to infer format
                df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
        except Exception:
            # Final fallback
            df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")

        # Extract temporal features
        df["year"] = df["FL_DATE"].dt.year.astype("float64")
        df["month"] = df["FL_DATE"].dt.month.astype("float64")
        df["day"] = df["FL_DATE"].dt.day.astype("float64")
        df["dow"] = df["FL_DATE"].dt.dayofweek.astype("float64")

    # Time blocks to minutes for potential modeling.
    for col in ["CRS_DEP_TIME", "DEP_TIME", "CRS_ARR_TIME", "ARR_TIME", "WHEELS_OFF", "WHEELS_ON"]:
        if col in df.columns:
            df[f"{col.lower()}_mins"] = hhmm_to_minutes(df[col])

    # Label: arrival delay minutes and binary >15 threshold.
    if "ARR_DELAY" in df.columns:
        df["arr_delay_min"] = df["ARR_DELAY"].astype("float64")
        df["arr_delayed_15"] = (df["arr_delay_min"] >= 15).astype("float64")

    # Label: departure delay minutes and binary >15.
    if "DEP_DELAY" in df.columns:
        df["dep_delay_min"] = df["DEP_DELAY"].astype("float64")
        df["dep_delayed_15"] = (df["dep_delay_min"] >= 15).astype("float64")

    # COMPREHENSIVE DELAY ANALYSIS
    
    # 1. Cancellation reasons
    if "CANCELLATION_CODE" in df.columns:
        df["cancellation_reason"] = df["CANCELLATION_CODE"].map({
            "A": "carrier",
            "B": "weather", 
            "C": "national_air_system",
            "D": "security"
        }).fillna("not_cancelled")
    
    # 2. Time-based delay patterns
    if "CRS_DEP_TIME" in df.columns:
        df["dep_hour"] = (df["CRS_DEP_TIME"] // 100).astype("float64")
        df["dep_time_category"] = pd.cut(df["dep_hour"], 
                                        bins=[-1, 6, 10, 14, 18, 24], 
                                        labels=["early_morning", "morning", "afternoon", "evening", "night"])
    
    # 3. Delay magnitude categories
    if "ARR_DELAY" in df.columns:
        df["delay_magnitude"] = pd.cut(df["ARR_DELAY"],
                                     bins=[-float('inf'), -15, 0, 15, 60, 180, float('inf')],
                                     labels=["early", "on_time", "minor_delay", "moderate_delay", "major_delay", "severe_delay"])
    
    # 4. Primary and secondary delay reasons
    reason_cols = [c for c in [
        "CARRIER_DELAY",
        "WEATHER_DELAY",
        "NAS_DELAY",
        "SECURITY_DELAY",
        "LATE_AIRCRAFT_DELAY",
    ] if c in df.columns]
    
    if reason_cols:
        def analyze_delay_reasons(row) -> tuple:
            vals = {c: row[c] for c in reason_cols if not pd.isna(row[c]) and row[c] > 0}
            if not vals:
                return "on_time", "none", 0, 0
            
            sorted_reasons = sorted(vals.items(), key=lambda x: x[1], reverse=True)
            primary = sorted_reasons[0][0].replace("_DELAY", "").lower()
            primary_mins = sorted_reasons[0][1]
            
            secondary = "none"
            secondary_mins = 0
            if len(sorted_reasons) > 1:
                secondary = sorted_reasons[1][0].replace("_DELAY", "").lower()
                secondary_mins = sorted_reasons[1][1]
                
            return primary, secondary, primary_mins, secondary_mins

        delay_analysis = df.apply(analyze_delay_reasons, axis=1, result_type="expand")
        df["primary_delay_reason"] = delay_analysis[0]
        df["secondary_delay_reason"] = delay_analysis[1]
        df["primary_delay_minutes"] = delay_analysis[2]
        df["secondary_delay_minutes"] = delay_analysis[3]
        
        # Total delay time from all causes
        df["total_delay_minutes"] = df[reason_cols].fillna(0).sum(axis=1)
    
    # 5. Operational efficiency metrics
    if "TAXI_OUT" in df.columns and "TAXI_IN" in df.columns:
        df["total_taxi_time"] = df["TAXI_OUT"].fillna(0) + df["TAXI_IN"].fillna(0)
    
    if "CRS_ELAPSED_TIME" in df.columns and "ACTUAL_ELAPSED_TIME" in df.columns:
        df["schedule_vs_actual_diff"] = df["ACTUAL_ELAPSED_TIME"] - df["CRS_ELAPSED_TIME"]
    
    # 6. Airport and route characteristics for delay analysis
    if "ORIGIN" in df.columns and "DEST" in df.columns:
        df["route"] = df["ORIGIN"].astype(str) + "-" + df["DEST"].astype(str)
        df["is_hub_route"] = df["ORIGIN"].isin(["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO"]) | \
                             df["DEST"].isin(["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO"])

    # Keep cancelled/diverted flights for comprehensive delay analysis
    if "CANCELLED" in df.columns:
        df["is_cancelled"] = (df["CANCELLED"] == 1).astype("float64")
    if "DIVERTED" in df.columns:
        df["is_diverted"] = (df["DIVERTED"] == 1).astype("float64")

    return df


def detect_header(csv_path: Path) -> bool:
    """Detect if CSV file has a header row.

    Returns:
        True if file has header, False otherwise
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    # Check if first line contains expected column names
    # Newer files have 'YEAR' as first column, older files start with year value like '2012'
    return 'YEAR' in first_line.upper() or 'FL_DATE' in first_line.upper()


def process_file(csv_path: Path, chunksize: int = 200_000) -> Iterable[Tuple[pd.DataFrame, Dict]]:
    """Process a CSV file in chunks with validation and feature engineering.

    Handles both files with headers (2019+) and without headers (2012-2018).

    Yields:
        Tuple of (processed_dataframe, quality_metrics) for each chunk
    """
    try:
        # Detect if file has header
        has_header = detect_header(csv_path)

        logger.info(f"Processing {csv_path.name} (header={'yes' if has_header else 'no'})...")

        # Read sample to determine available columns
        if has_header:
            sample_df = pd.read_csv(csv_path, nrows=5)
            available_cols = [c for c in BASE_COLS if c in sample_df.columns]
        else:
            # For files without headers, read with column names
            sample_df = pd.read_csv(csv_path, nrows=5, header=None, names=BASE_COLS)
            # Check which columns have any non-null data
            available_cols = [col for col in BASE_COLS if col in sample_df.columns]

        logger.info(f"  {len(available_cols)}/{len(BASE_COLS)} columns available")

        # Use safer dtypes to avoid parsing errors
        dtype_map = {k: v for k, v in DTYPES.items() if k in available_cols and k != "FL_DATE"}

        # Build read_csv arguments
        read_kwargs = {
            'dtype': dtype_map,
            'chunksize': chunksize,
            'low_memory': False,
            'on_bad_lines': 'warn'
        }

        if has_header:
            read_kwargs['usecols'] = available_cols
        else:
            # No header - specify names and use only available columns
            read_kwargs['header'] = None
            read_kwargs['names'] = BASE_COLS
            # Only select columns that exist (based on column count)
            with open(csv_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                actual_col_count = len(first_line.split(','))
            read_kwargs['usecols'] = list(range(min(actual_col_count, len(BASE_COLS))))

        # Process file in chunks
        for i, chunk in enumerate(pd.read_csv(csv_path, **read_kwargs)):
            try:
                # For headerless files, assign proper column names
                if not has_header and chunk.shape[1] < len(BASE_COLS):
                    # Trim BASE_COLS to match actual columns
                    chunk.columns = BASE_COLS[:chunk.shape[1]]

                # Validate and clean data
                chunk_clean, quality_metrics = validate_and_clean_data(chunk)

                # Add features
                chunk_enriched = add_features(chunk_clean)

                yield chunk_enriched, quality_metrics

            except Exception as e:
                logger.error(f"Error processing chunk {i} from {csv_path.name}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error reading file {csv_path.name}: {e}")
        raise


def run() -> None:
    """Main preprocessing pipeline with quality reporting."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found under {DATA_DIR}")

    logger.info(f"Found {len(csv_files)} CSV files to process")

    # Clean up existing output
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        logger.info(f"Removed existing output file: {OUTPUT_FILE}")

    # Track overall quality metrics
    total_quality_metrics = {
        'files_processed': 0,
        'files_failed': 0,
        'total_rows_input': 0,
        'total_rows_output': 0,
        'total_rows_dropped': 0,
        'total_missing_critical': 0,
        'total_invalid_delays': 0,
        'total_invalid_times': 0,
        'total_invalid_distances': 0,
        'chunks_processed': 0
    }

    # Process each file
    for csv_path in csv_files:
        try:
            logger.info(f"Processing {csv_path.name}...")
            file_chunks = 0

            for i, (df, quality_metrics) in enumerate(process_file(csv_path)):
                # Write to output
                mode = "a"
                header = not OUTPUT_FILE.exists()
                df.to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)

                # Update metrics
                total_quality_metrics['total_rows_input'] += quality_metrics['rows_input']
                total_quality_metrics['total_rows_output'] += quality_metrics['rows_output']
                total_quality_metrics['total_rows_dropped'] += quality_metrics['rows_dropped']
                total_quality_metrics['total_missing_critical'] += quality_metrics['missing_critical_fields']
                total_quality_metrics['total_invalid_delays'] += quality_metrics['invalid_delays']
                total_quality_metrics['total_invalid_times'] += quality_metrics['invalid_times']
                total_quality_metrics['total_invalid_distances'] += quality_metrics['invalid_distances']
                total_quality_metrics['chunks_processed'] += 1

                file_chunks += 1

                if (i + 1) % 10 == 0:
                    logger.info(f"  Wrote chunk {i + 1} for {csv_path.name}")

            total_quality_metrics['files_processed'] += 1
            logger.info(f"Completed {csv_path.name}: {file_chunks} chunks processed")

        except Exception as e:
            logger.error(f"Failed to process {csv_path.name}: {e}")
            total_quality_metrics['files_failed'] += 1
            continue

    # Generate quality report
    generate_quality_report(total_quality_metrics)

    logger.info(f"Finished! Output written to {OUTPUT_FILE}")
    logger.info(f"Quality report written to {QUALITY_REPORT_FILE}")


def generate_quality_report(metrics: Dict) -> None:
    """Generate a comprehensive data quality report."""
    report_lines = [
        "="*60,
        "DATA QUALITY REPORT",
        "="*60,
        "",
        "PROCESSING SUMMARY:",
        f"  Files processed: {metrics['files_processed']}",
        f"  Files failed: {metrics['files_failed']}",
        f"  Chunks processed: {metrics['chunks_processed']}",
        "",
        "DATA VOLUME:",
        f"  Total rows input: {metrics['total_rows_input']:,}",
        f"  Total rows output: {metrics['total_rows_output']:,}",
        f"  Total rows dropped: {metrics['total_rows_dropped']:,}",
        f"  Retention rate: {(metrics['total_rows_output']/metrics['total_rows_input']*100):.2f}%",
        "",
        "DATA QUALITY ISSUES:",
        f"  Missing critical fields: {metrics['total_missing_critical']:,}",
        f"  Invalid delay values: {metrics['total_invalid_delays']:,}",
        f"  Invalid time values: {metrics['total_invalid_times']:,}",
        f"  Invalid distances: {metrics['total_invalid_distances']:,}",
        "",
        "="*60,
        f"Report generated: {pd.Timestamp.now()}",
        "="*60
    ]

    report_text = "\n".join(report_lines)

    # Write to file
    with open(QUALITY_REPORT_FILE, 'w') as f:
        f.write(report_text)

    # Also print to console
    logger.info("\n" + report_text)


if __name__ == "__main__":
    run()
