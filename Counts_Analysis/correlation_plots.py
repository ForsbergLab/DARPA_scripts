#!/usr/bin/env python3
"""
CSV Sequence Analysis Generator
Analyzes sequence data with control vs experimental comparisons.
Can create correlation plots and fitness score distributions.
"""

# USAGE: python3 correlation_plots.py fitness <fitness_scores_file_path>
#       -o <output_file_path_svg> --x-label " X LABEL" --y-label "Y LABEL"
#       --title "TITLE"

# Option correlation will only generate a correlation plot, option fitness
# will generate correlation plot with a distribution plot under it.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import seaborn as sns


def load_and_process_data(csv_file):
    """Load CSV and process the sequence data."""
    try:
        # Try to read the CSV with different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not read CSV file with any standard encoding")

        # Check for required columns (case-insensitive)
        df.columns = df.columns.str.strip()  # Remove whitespace

        required_columns = {
            'id': None,
            'sequence': None,
            'control_count': None,
            'experimental_count': None,
            'control_rel_abundance': None,
            'experimental_rel_abundance': None,
            'fitness_score': None
        }

        # Find columns with case-insensitive matching
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if col_lower in ['id', 'sequence_identifier']:
                required_columns['id'] = col
            elif col_lower in ['sequence', 'sequence_data']:
                required_columns['sequence'] = col
            elif col_lower in ['control_count', 'control_raw_count']:
                required_columns['control_count'] = col
            elif col_lower in ['experimental_count', 'experimental_raw_count']:
                required_columns['experimental_count'] = col
            elif col_lower in ['control_relative_abundance', 'control_rel_abundance']:
                required_columns['control_rel_abundance'] = col
            elif col_lower in ['experimental_relative_abundance', 'experimental_rel_abundance']:
                required_columns['experimental_rel_abundance'] = col
            elif col_lower in ['fitness_score', 'fitness']:
                required_columns['fitness_score'] = col

        # Check if all required columns were found
        missing_columns = [key for key, value in required_columns.items() if value is None]
        if missing_columns:
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Select and rename columns
        column_mapping = {v: k for k, v in required_columns.items()}
        df_clean = df[list(column_mapping.keys())].copy()
        df_clean = df_clean.rename(columns=column_mapping)

        # Remove rows with missing ID or sequence data
        df_clean = df_clean.dropna(subset=['id', 'sequence'])

        # Ensure numeric columns are numeric
        numeric_cols = ['control_count', 'experimental_count',
                        'control_rel_abundance', 'experimental_rel_abundance', 'fitness_score']
        for col in numeric_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # Remove rows where numeric conversion failed
        df_clean = df_clean.dropna(subset=numeric_cols)

        # Set fitness score to NaN for sequences with zero counts, but keep the rows
        zero_count_mask = (df_clean['control_count'] == 0) | (df_clean['experimental_count'] == 0)
        df_clean.loc[zero_count_mask, 'fitness_score'] = np.nan

        # Filter out negative counts (but allow zero counts)
        df_clean = df_clean[(df_clean['control_count'] >= 0) & (df_clean['experimental_count'] >= 0)]

        # Convert ID to string
        df_clean['id'] = df_clean['id'].astype(str)

        print(f"Loaded {len(df_clean)} valid records from {csv_file}")
        return df_clean

    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)


def create_correlation_plot(df, output_file, log_scale=False, x_label=None, y_label=None, title=None):
    """Create correlation plot between control and experimental relative abundances."""
    print(f"Creating correlation plot for relative abundances")
    print("-" * 60)

    # Separate sequences with valid fitness scores from those with NaN
    valid_fitness = df[df['fitness_score'].notna()]
    invalid_fitness = df[df['fitness_score'].isna()]

    print(f"Total sequences: {len(df)}")
    print(f"Sequences with valid fitness scores: {len(valid_fitness)}")
    print(f"Sequences with zero counts (fitness = N/A): {len(invalid_fitness)}")

    if len(valid_fitness) > 0:
        print(
            f"Control abundance range: {df['control_rel_abundance'].min():.6f} - {df['control_rel_abundance'].max():.6f}")
        print(
            f"Experimental abundance range: {df['experimental_rel_abundance'].min():.6f} - {df['experimental_rel_abundance'].max():.6f}")

        # Calculate correlations only for sequences with valid fitness scores
        pearson_corr, pearson_p = pearsonr(valid_fitness['control_rel_abundance'],
                                           valid_fitness['experimental_rel_abundance'])
        spearman_corr, spearman_p = spearmanr(valid_fitness['control_rel_abundance'],
                                              valid_fitness['experimental_rel_abundance'])

        print(f"\nCorrelation Analysis (valid fitness scores only):")
        print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
        print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
    else:
        pearson_corr = spearman_corr = np.nan
        print("No sequences with valid fitness scores for correlation analysis")

    # Create the correlation plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use log scale if requested
    if log_scale:
        x_vals = np.log10(df['control_rel_abundance'] + 1e-10)  # Add small value to avoid log(0)
        y_vals = np.log10(df['experimental_rel_abundance'] + 1e-10)

        default_x_label = 'Log₁₀(Control Relative Abundance)'
        default_y_label = 'Log₁₀(Experimental Relative Abundance)'
    else:
        x_vals = df['control_rel_abundance']
        y_vals = df['experimental_rel_abundance']

        default_x_label = 'Control Relative Abundance'
        default_y_label = 'Experimental Relative Abundance'

    # Use custom labels if provided, otherwise use defaults
    xlabel = x_label if x_label else default_x_label
    ylabel = y_label if y_label else default_y_label

    # Plot sequences with valid fitness scores
    if len(valid_fitness) > 0:
        if log_scale:
            x_valid = np.log10(valid_fitness['control_rel_abundance'] + 1e-10)
            y_valid = np.log10(valid_fitness['experimental_rel_abundance'] + 1e-10)
        else:
            x_valid = valid_fitness['control_rel_abundance']
            y_valid = valid_fitness['experimental_rel_abundance']

        scatter_valid = ax.scatter(x_valid, y_valid, alpha=0.6, s=50, c=valid_fitness['fitness_score'],
                                   cmap='RdBu_r', edgecolors='black', linewidth=0.5, label='Valid fitness')

        # Add colorbar with proper formatting
        cbar = plt.colorbar(scatter_valid, ax=ax)
        cbar.set_label('Fitness Score', rotation=270, labelpad=20)

        # Get the formatter and disable scientific notation and offset
        formatter = cbar.ax.yaxis.get_major_formatter()
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        cbar.ax.yaxis.set_major_formatter(formatter)

        # Add trend line for valid data only
        z = np.polyfit(x_valid, y_valid, 1)
        p = np.poly1d(z)
        ax.plot(x_valid, p(x_valid), "r--", alpha=0.8, linewidth=2, label=f'Trend line (slope: {z[0]:.3f})')

        # Label the top 3 sequences with highest fitness scores with enhanced collision avoidance
        top_3_fitness = valid_fitness.nlargest(3, 'fitness_score')
        placed_labels = []  # Store successfully placed label info

        for _, row in top_3_fitness.iterrows():
            if log_scale:
                x_pos = np.log10(row['control_rel_abundance'] + 1e-10)
                y_pos = np.log10(row['experimental_rel_abundance'] + 1e-10)
            else:
                x_pos = row['control_rel_abundance']
                y_pos = row['experimental_rel_abundance']

            # Calculate plot dimensions for relative positioning
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

            # Try different positions in order of preference
            position_attempts = [
                (5, 5, 'left'),  # Top right (preferred)
                (5, -5, 'left'),  # Bottom right
                (-5, 5, 'right'),  # Top left
                (-5, -5, 'right'),  # Bottom left
                (10, 5, 'left'),  # Further right
                (5, 10, 'left'),  # Higher up right
                (-10, 5, 'right'),  # Further left
                (5, -10, 'left'),  # Lower right
                (-5, 10, 'right'),  # Higher up left
                (-5, -10, 'right'),  # Lower left
                (15, 0, 'left'),  # Far right
                (-15, 0, 'right'),  # Far left
            ]

            label_placed = False
            for offset_x, offset_y, ha_align in position_attempts:
                # Calculate actual label position in data coordinates
                label_x = x_pos + (offset_x * x_range / 100)
                label_y = y_pos + (offset_y * y_range / 100)

                # Check for overlap with existing labels
                overlap = False
                for existing_label in placed_labels:
                    existing_x, existing_y = existing_label
                    # Use larger margins for better spacing
                    x_margin = x_range * 0.12  # 12% of x-range
                    y_margin = y_range * 0.08  # 8% of y-range

                    if (abs(label_x - existing_x) < x_margin and
                            abs(label_y - existing_y) < y_margin):
                        overlap = True
                        break

                if not overlap:
                    # Place the label
                    ax.annotate(f"{row['id']}\n({row['fitness_score']:.2f})",
                                xy=(x_pos, y_pos),
                                xytext=(offset_x, offset_y), textcoords='offset points',
                                fontsize=10, ha=ha_align, va='bottom',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

                    # Record this label's position
                    placed_labels.append((label_x, label_y))
                    label_placed = True
                    break

            # If no position worked, place it anyway with a warning
            if not label_placed:
                ax.annotate(f"{row['id']}\n({row['fitness_score']:.2f})",
                            xy=(x_pos, y_pos),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=10, ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                print(f"Warning: Could not find non-overlapping position for label {row['id']}")

    # Plot sequences with invalid fitness scores (zero counts) in black
    if len(invalid_fitness) > 0:
        if log_scale:
            x_invalid = np.log10(invalid_fitness['control_rel_abundance'] + 1e-10)
            y_invalid = np.log10(invalid_fitness['experimental_rel_abundance'] + 1e-10)
        else:
            x_invalid = invalid_fitness['control_rel_abundance']
            y_invalid = invalid_fitness['experimental_rel_abundance']

        ax.scatter(x_invalid, y_invalid, alpha=0.6, s=30, c='black',
                   edgecolors='white', linewidth=0.5, label='Zero counts (fitness = N/A)')

    # Add diagonal reference line (perfect correlation)
    if len(df) > 0:
        min_val = min(min(x_vals), min(y_vals))
        max_val = max(max(x_vals), max(y_vals))
        ax.plot([min_val, max_val], [min_val, max_val], 'k:', alpha=0.5, linewidth=1, label='Perfect correlation')

    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    # Generate title
    if title:
        plot_title = title
    else:
        plot_title = f'{xlabel} vs {ylabel}'

    ax.set_title(plot_title, fontsize=14, fontweight='bold', pad=20)

    # Add correlation info as text box
    if len(valid_fitness) > 0:
        textstr = f'Pearson r = {pearson_corr:.3f}\nSpearman ρ = {spearman_corr:.3f}\nn = {len(valid_fitness)} valid sequences\n{len(invalid_fitness)} with zero counts'
    else:
        textstr = f'No valid fitness scores\n{len(df)} total sequences\n{len(invalid_fitness)} with zero counts'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    # Equal aspect ratio if not log scale
    if not log_scale:
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save the plot with error handling
    try:
        plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"\nCorrelation plot saved as: {output_file}")
    except Exception as e:
        print(f"Warning: Could not save SVG file ({e}). Trying PNG instead...")
        png_file = output_file.replace('.svg', '.png')
        plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Plot saved as PNG: {png_file}")

    plt.show()

    return pearson_corr, spearman_corr


def create_fitness_distribution(df, output_file, x_label=None, y_label=None, title=None):
    """Create fitness score distribution plot."""
    print(f"Creating fitness score distribution")
    print("-" * 60)

    # Separate valid and invalid fitness scores
    valid_fitness = df[df['fitness_score'].notna()]
    invalid_fitness = df[df['fitness_score'].isna()]

    print(f"Total sequences: {len(df)}")
    print(f"Sequences with valid fitness scores: {len(valid_fitness)}")
    print(f"Sequences with zero counts (fitness = N/A): {len(invalid_fitness)}")

    if len(valid_fitness) == 0:
        print("No sequences with valid fitness scores to analyze!")
        return pd.Series(dtype=float)

    print(
        f"Fitness score range: {valid_fitness['fitness_score'].min():.3f} - {valid_fitness['fitness_score'].max():.3f}")
    print(f"Mean fitness score: {valid_fitness['fitness_score'].mean():.3f}")
    print(f"Median fitness score: {valid_fitness['fitness_score'].median():.3f}")

    # Count positive, negative, and neutral fitness scores
    positive = (valid_fitness['fitness_score'] > 0.1).sum()
    negative = (valid_fitness['fitness_score'] < -0.1).sum()
    neutral = ((valid_fitness['fitness_score'] >= -0.1) & (valid_fitness['fitness_score'] <= 0.1)).sum()

    print(f"Sequences with positive fitness (>0.1): {positive}")
    print(f"Sequences with negative fitness (<-0.1): {negative}")
    print(f"Sequences with neutral fitness (-0.1 to 0.1): {neutral}")

    # Create the distribution plot
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Histogram of fitness scores (only valid ones)
    ax1.hist(valid_fitness['fitness_score'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral (0)')
    ax1.axvline(x=valid_fitness['fitness_score'].mean(), color='orange', linestyle='-', alpha=0.7,
                label=f'Mean ({valid_fitness["fitness_score"].mean():.3f})')

    ax1.set_xlabel(x_label if x_label else 'Fitness Score (log2 ratio)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(y_label if y_label else 'Frequency', fontsize=12, fontweight='bold')
    title_text = title if title else f'Distribution of Fitness Scores (n={len(valid_fitness)})'
    if len(invalid_fitness) > 0:
        title_text += f'\n{len(invalid_fitness)} sequences excluded (zero counts)'
    ax1.set_title(title_text, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Scatter plot: Control vs Experimental abundance colored by fitness (log scale)
    # Plot valid fitness scores with log scale
    x_vals_log = np.log10(valid_fitness['control_rel_abundance'] + 1e-10)
    y_vals_log = np.log10(valid_fitness['experimental_rel_abundance'] + 1e-10)

    scatter_valid = ax2.scatter(x_vals_log, y_vals_log,
                                alpha=0.6, s=50, c=valid_fitness['fitness_score'], cmap='RdBu_r',
                                edgecolors='black', linewidth=0.3, label='Valid fitness')

    # Plot invalid fitness scores (zero counts) in black with log scale
    if len(invalid_fitness) > 0:
        x_invalid_log = np.log10(invalid_fitness['control_rel_abundance'] + 1e-10)
        y_invalid_log = np.log10(invalid_fitness['experimental_rel_abundance'] + 1e-10)
        ax2.scatter(x_invalid_log, y_invalid_log,
                    alpha=0.6, s=30, c='black', edgecolors='white', linewidth=0.3,
                    label='Zero counts (fitness = N/A)')

    # Add diagonal line for reference (log scale)
    min_val_log = min(x_vals_log.min(), y_vals_log.min())
    max_val_log = max(x_vals_log.max(), y_vals_log.max())
    ax2.plot([min_val_log, max_val_log], [min_val_log, max_val_log], 'k--', alpha=0.5, label='Equal abundance')

    ax2.set_xlabel('Log₁₀(Control Relative Abundance)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Log₁₀(Experimental Relative Abundance)', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Abundances Colored by Fitness Score', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add colorbar for fitness scores (only for valid data)
    cbar = plt.colorbar(scatter_valid, ax=ax2)
    cbar.set_label('Fitness Score', rotation=270, labelpad=20)

    plt.tight_layout()

    # Save the plot with error handling
    try:
        plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"\nFitness distribution plot saved as: {output_file}")
    except Exception as e:
        print(f"Warning: Could not save SVG file ({e}). Trying PNG instead...")
        png_file = output_file.replace('.svg', '.png')
        plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Plot saved as PNG: {png_file}")

    plt.show()

    return valid_fitness['fitness_score'].describe()


def main():
    parser = argparse.ArgumentParser(description='Analyze sequence data with control vs experimental comparisons')

    # Subcommands for different types of analysis
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Correlation plot command
    corr_parser = subparsers.add_parser('correlation', help='Generate correlation plot of relative abundances')
    corr_parser.add_argument('csv_file', help='Path to CSV file with sequence data')
    corr_parser.add_argument('-o', '--output', help='Output SVG file name', default='abundance_correlation.svg')
    corr_parser.add_argument('--log-scale', action='store_true', help='Use logarithmic scale for both axes')
    corr_parser.add_argument('--x-label', type=str, help='Custom label for X-axis')
    corr_parser.add_argument('--y-label', type=str, help='Custom label for Y-axis')
    corr_parser.add_argument('--title', type=str, help='Custom title for the plot')

    # Fitness distribution command
    fitness_parser = subparsers.add_parser('fitness', help='Generate fitness score distribution plot')
    fitness_parser.add_argument('csv_file', help='Path to CSV file with sequence data')
    fitness_parser.add_argument('-o', '--output', help='Output SVG file name', default='fitness_distribution.svg')
    fitness_parser.add_argument('--x-label', type=str, help='Custom label for X-axis')
    fitness_parser.add_argument('--y-label', type=str, help='Custom label for Y-axis')
    fitness_parser.add_argument('--title', type=str, help='Custom title for the plot')

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Validate input file
    if not Path(args.csv_file).exists():
        print(f"Error: File {args.csv_file} does not exist")
        sys.exit(1)

    # Load data
    print(f"Processing: {args.csv_file}")
    print("-" * 50)
    df = load_and_process_data(args.csv_file)

    if args.command == 'correlation':
        # Create correlation plot
        pearson_corr, spearman_corr = create_correlation_plot(
            df, args.output, args.log_scale, args.x_label, args.y_label, args.title
        )

        print("-" * 50)
        print("Summary Statistics:")
        print(f"  Total sequences analyzed: {len(df)}")
        print(f"  Pearson correlation: {pearson_corr:.4f}")
        print(f"  Spearman correlation: {spearman_corr:.4f}")

    elif args.command == 'fitness':
        # Create fitness distribution plot
        fitness_stats = create_fitness_distribution(
            df, args.output, args.x_label, args.y_label, args.title
        )

        print("-" * 50)
        print("Fitness Score Statistics:")
        print(fitness_stats)


if __name__ == "__main__":
    main()
