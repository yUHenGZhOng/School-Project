
import pandas as pd
import argparse
import os
import re

def clean_emotion_data(input_path, output_path, core_group_size, min_participation_rate):
    """
    Cleans the raw emotion analysis data based on participation rates.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the cleaned CSV file.
        core_group_size (int): The expected number of core participants.
        min_participation_rate (float): The minimum participation rate for a person to be considered a core member.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    # 2. Extract Theoretical Max Samples from filename
    filename = os.path.basename(input_path)
    match = re.search(r'_(\d+)-(\d+)_(\d+\.\d+)%_(\d+)s\.csv', filename)
    
    if not match:
        print(f"Error: Could not extract analysis metadata from filename: {filename}")
        print("Filename must match the pattern like '..._actual-theoretical_rate%_interval_s.csv'")
        return
        
    theoretical_max_samples = int(match.group(2))

    # 3. First Pass: Identify Core Candidates
    person_counts = df['person_id'].value_counts()
    core_candidates = person_counts.nlargest(core_group_size).index.tolist()

    # 4. Second Pass: Apply Dynamic Threshold
    min_appearance_threshold = theoretical_max_samples * min_participation_rate
    
    valid_ids = [
        pid for pid in core_candidates 
        if person_counts.get(pid, 0) >= min_appearance_threshold
    ]

    # 5. Filter Data
    original_rows = len(df)
    cleaned_df = df[df['person_id'].isin(valid_ids)]
    cleaned_rows = len(cleaned_df)

    # 6. Save Result
    cleaned_df.to_csv(output_path, index=False)

    # 7. Print Report
    all_ids = set(person_counts.index)
    removed_ids = list(all_ids - set(valid_ids))

    print("--- Data Cleaning Report ---")
    print(f"Input File: {filename}")
    print(f"Theoretical Max Samples Extracted: {theoretical_max_samples}")
    print(f"Minimum Appearance Threshold: {int(min_appearance_threshold)} (Rate: {min_participation_rate})")
    print("-" * 20)
    print(f"Original Data Rows: {original_rows}")
    print(f"Cleaned Data Rows: {cleaned_rows}")
    print(f"Rows Removed: {original_rows - cleaned_rows}")
    print("-" * 20)
    print(f"Identified Core Participants (IDs): {sorted(valid_ids)}")
    print(f"Removed Noise IDs: {sorted(removed_ids)}")
    print("----------------------------")
    print(f"Cleaned data successfully saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Smartly clean raw emotion analysis data.")
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help='Path to the input raw CSV file.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help='Full path for the cleaned output CSV file (e.g., output/cleaned/4_cleaned.csv).'
    )
    parser.add_argument(
        '--core_group_size', 
        type=int, 
        default=6, 
        help='The expected number of core participants. Defaults to 6.'
    )
    parser.add_argument(
        '--min_participation_rate', 
        type=float, 
        default=0.5,
        help='Minimum participation rate (0.0 to 1.0) for a core member. Defaults to 0.5.'
    )

    args = parser.parse_args()

    # Ensure the output directory exists before processing
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    clean_emotion_data(
        args.input,
        args.output,
        args.core_group_size,
        args.min_participation_rate
    )

if __name__ == '__main__':
    main()
