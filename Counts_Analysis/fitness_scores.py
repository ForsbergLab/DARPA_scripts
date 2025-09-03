


# USAGE: python3 fitness_scores.py <control_file_path> <experimental_file_path> <output_file_path>
# Fitness Score = log2((experimental_count/experimental_total) / (control_count/control_total))


import pandas as pd
import numpy as np
import argparse
import sys



def calculate_fitness_scores(control_file, experimental_file, output_file):
    """
    Calculate fitness scores from two CSV files with ID, Sequence, and Count columns.

    Fitness Score = log2((experimental_count/experimental_total) / (control_count/control_total))

    Args:
        control_file (str): Path to the control CSV file
        experimental_file (str): Path to the experimental CSV file
        output_file (str): Path for the output CSV file
    """

    try:
        # Read the CSV files
        print(f"Reading {control_file}...")
        control_data = pd.read_csv(control_file)

        print(f"Reading {experimental_file}...")
        experimental_data = pd.read_csv(experimental_file)

        # Validate required columns exist
        required_cols = ['ID', 'Sequence', 'Count']
        for df, name in [(control_data, 'control'), (experimental_data, 'experimental')]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {name} file: {missing_cols}")

        # Calculate total counts for each file
        control_total = control_data['Count'].sum()
        experimental_total = experimental_data['Count'].sum()

        print(f"Control total count: {control_total}")
        print(f"Experimental total count: {experimental_total}")

        # Merge the dataframes on ID to align the data
        merged = pd.merge(control_data[['ID', 'Sequence', 'Count']],
                          experimental_data[['ID', 'Count']],
                          on='ID',
                          suffixes=('_control', '_experimental'),
                          how='inner')  # Only keep IDs present in both files

        # Calculate normalized frequencies
        merged['freq_control'] = merged['Count_control'] / control_total
        merged['freq_experimental'] = merged['Count_experimental'] / experimental_total

        # Calculate fitness scores: log2(freq_experimental / freq_control)
        # Set fitness score to N/A if either count is zero
        merged['Fitness_Score'] = np.where(
            (merged['Count_control'] > 0) & (merged['Count_experimental'] > 0),
            np.log2(merged['freq_experimental'] / merged['freq_control']),
            np.nan
        )

        valid_data = merged.copy()

        # Create output dataframe
        output_data = valid_data[
            ['ID', 'Sequence', 'Count_control', 'Count_experimental', 'freq_control', 'freq_experimental',
             'Fitness_Score']].copy()
        output_data.columns = ['ID', 'Sequence', 'Control Count', 'Experimental Count', 'Control Relative Abundance',
                               'Experimental Relative Abundance', 'Fitness Score']

        # Sort by fitness score in descending order (highest fitness first)
        # NaN values will be placed at the end
        output_data = output_data.sort_values('Fitness Score', ascending=False, na_position='last').reset_index(
            drop=True)

        # Save to output file
        output_data.to_csv(output_file, index=False)

        print(f"\nResults saved to {output_file}")
        print(f"Processed {len(output_data)} sequences")

        # Count valid vs N/A fitness scores
        valid_scores = output_data['Fitness Score'].notna().sum()
        na_scores = output_data['Fitness Score'].isna().sum()
        print(f"Valid fitness scores: {valid_scores}")
        print(f"N/A fitness scores (zero counts): {na_scores}")

        if valid_scores > 0:
            print(
                f"Fitness scores range: {output_data['Fitness Score'].min():.3f} to {output_data['Fitness Score'].max():.3f}")

        # Show some example results
        print(f"\nFirst 5 results:")
        print(output_data.head())

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Calculate fitness scores from two CSV files')
    parser.add_argument('control_file', help='Path to the control CSV file')
    parser.add_argument('experimental_file', help='Path to the experimental CSV file')
    parser.add_argument('output_file', help='Path for the output CSV file')

    args = parser.parse_args()

    calculate_fitness_scores(args.control_file, args.experimental_file, args.output_file)


if __name__ == "__main__":
    main()
