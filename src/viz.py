import ast
import os

import joblib
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mstats, trim_mean


def compute_label_balance_from_str(conf_matrix):
    conf_matrix = conf_matrix.replace('array', '').strip('[]')
    matrix = np.array(ast.literal_eval(conf_matrix))
    return compute_label_balance(matrix)

def compute_label_balance(matrix):
    yes_count = matrix[0, 1] + matrix[1, 1]  # FP + TP
    no_count = matrix[0, 0] + matrix[1, 0]  # TN + FN
    total_count = matrix.sum()
    yes_ratio = (yes_count / total_count)
    no_ratio = (no_count / total_count)
    return min(yes_ratio, no_ratio) * 2

def add_label_balance_to_csv(csv_path='data/prompt_test_results.csv'):
    df = pd.read_csv(csv_path)
    df['Label Balance'] = df['Confusion Matrix'].apply(compute_label_balance_from_str)
    df.to_csv(csv_path, index=False)

def plot_prompt_metrics_with_annotations(csv_path='data/experiment_results/prompt_metrics.csv'):
    df = pd.read_csv(csv_path)
    
    norm = mcolors.LogNorm(vmin=df['None Count'].min() + 1, vmax=df['None Count'].max())
    colormap = plt.cm.get_cmap('RdYlGn_r')  # Red to green, reversed

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    sc = plt.scatter(df['Accuracy'], df['Label Balance'], c=df['None Count'], cmap=colormap, norm=norm, s=300, edgecolors='black', linewidths=0.5, zorder=3)

    plt.xlabel('Accuracy')
    plt.ylabel('Label Balance Score')
    
    cbar = plt.colorbar(sc, pad=0.05)
    cbar.set_label('Number of Failed Predictions')
    cbar.ax.yaxis.set_label_position('left')

    ax.set_axisbelow(True)
    plt.grid(True)
    plt.show()

def load_tfidf_matrix_and_vectorizer(tfidf_path, vectorizer_path):
    if os.path.exists(tfidf_path) and os.path.exists(vectorizer_path):
        print("Loading TF-IDF matrix and vectorizer from disk...")
        tfidf_matrix = joblib.load(tfidf_path)
        vectorizer = joblib.load(vectorizer_path)
        return tfidf_matrix, vectorizer
    else:
        raise FileNotFoundError("TF-IDF matrix or vectorizer file not found.")

def calculate_cumulative_frequencies(tfidf_matrix):
    freqs = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    sorted_indices = np.argsort(freqs)[::-1]
    sorted_freqs = freqs[sorted_indices]
    cumulative_freqs = np.cumsum(sorted_freqs)
    normalized_cumulative_freqs = cumulative_freqs / cumulative_freqs[-1]
    return normalized_cumulative_freqs

def plot_cumulative_frequency(normalized_cumulative_freqs):
    num_features = len(normalized_cumulative_freqs)
    feature_usage_counts = np.arange(1, num_features + 1)
    cumulative_freq_percentages = normalized_cumulative_freqs * 100

    plt.figure(figsize=(12, 8))
    plt.plot(feature_usage_counts, cumulative_freq_percentages, color='royalblue')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Frequency Percentage')
    plt.title('TF-IDF Feature Contribution Analysis')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set y-axis ticks in increments of 10%
    plt.yticks(np.arange(0, 110, step=10))

    # Set x-axis ticks with intervals of 5 million and starting from 0
    max_features = feature_usage_counts[-1]
    x_ticks = np.linspace(0, max_features, num=10, dtype=int)
    x_tick_labels = [f'{x // 1000000}M' for x in x_ticks]
    x_tick_labels[0] = '0'  # Start from 0 instead of 0M
    plt.xticks(x_ticks, x_tick_labels)
    
    # Mark at every 10% cumulative frequency percentage with a dot and a label
    for percent in range(10, 101, 10):
        idx = np.searchsorted(cumulative_freq_percentages, percent)
        if idx < num_features:
            plt.plot(feature_usage_counts[idx], cumulative_freq_percentages[idx], 'o', color='royalblue')
            if percent == 100:
                plt.text(feature_usage_counts[idx], cumulative_freq_percentages[idx] - 3, 
                         f'{feature_usage_counts[idx]:,}', horizontalalignment='center', fontsize=10)
            else:
                plt.text(feature_usage_counts[idx] + 100000, cumulative_freq_percentages[idx] - 3, 
                         f'{feature_usage_counts[idx]:,}', horizontalalignment='left', fontsize=10)
    
    plt.show()

def cumulative_sum_plotter():
    # Paths to the saved TF-IDF matrix and vectorizer
    tfidf_path = 'data/saved_models/tfidf/xgboost/tfidf_matrix.pkl'
    vectorizer_path = 'data/saved_models/tfidf/xgboost/tfidf_vectorizer.pkl'

    # Load the TF-IDF matrix and vectorizer
    tfidf_matrix, vectorizer = load_tfidf_matrix_and_vectorizer(tfidf_path, vectorizer_path)

    # Print the shape of the TF-IDF matrix
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Calculate cumulative frequencies
    normalized_cumulative_freqs = calculate_cumulative_frequencies(tfidf_matrix)

    # Plot the cumulative frequency percentage
    plot_cumulative_frequency(normalized_cumulative_freqs)

def load_features(file_path):
    """Load feature data from a .pkl file."""
    return pd.read_pickle(file_path)

def calculate_means(features_df):
    """Calculate mean of each feature."""
    return features_df.mean()

def generate_summary_table(output_csv_path='data/summary_statistics.csv'):
    # Load main dataset
    main_features = load_features('data/saved_data/full/features.pkl')
    main_labels = pd.read_pickle('data/saved_data/full/labels.pkl')

    # Ensure the features and labels have the same length
    assert len(main_features) == len(main_labels), "Features and labels length mismatch"

    # Split main dataset into human and AI texts based on the labels (0 for human, 1 for AI)
    human_features = main_features[main_labels == 0]
    ai_features = main_features[main_labels == 1]

    # Load evasive texts features
    control_features = load_features('data/evasive_texts/feature/control/full_features.pkl')
    basic_features = load_features('data/evasive_texts/feature/basic/full_features.pkl')
    advanced_features = load_features('data/evasive_texts/feature/advanced/full_features.pkl')
    
    # Calculate means for all datasets
    human_means = calculate_means(human_features)
    ai_means = calculate_means(ai_features)
    control_means = calculate_means(control_features)
    basic_means = calculate_means(basic_features)
    advanced_means = calculate_means(advanced_features)

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'human_texts': human_means,
        'ai_texts': ai_means,
        'control_evasive': control_means,
        'basic_evasive': basic_means,
        'advanced_evasive': advanced_means
    })

    # Round the means to 3 decimal places
    summary_df = summary_df.round(3)

    # Remove rows with 'tfidf' or 'sbert' in the index
    summary_df = summary_df[~summary_df.index.str.contains('tfidf|sbert', case=False)]

    # Calculate the relative difference
    summary_df['relative_diff'] = (summary_df['ai_texts'] - summary_df['human_texts']) / (summary_df['ai_texts'] + summary_df['human_texts']).abs()

    # Sort the DataFrame by the absolute value of the relative difference
    summary_df = summary_df.sort_values(by='relative_diff', key=lambda x: x.abs(), ascending=False)
    summary_df = summary_df.drop(columns=["relative_diff"])
    # Save the summary DataFrame to a CSV file
    summary_df.to_csv(output_csv_path, index=True)

    # Print the summary DataFrame
    print(summary_df)

    return summary_df

def plot_relative_differences(summary_df, output_plot_path='data/figures/relative_diff_plot.png'):
    # Create a horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, len(summary_df) * 0.3))  # Adjust the figure size as needed
    colors = ['firebrick' if diff > 0 else 'royalblue' for diff in summary_df['relative_diff']]
    bars = ax.barh(summary_df.index, summary_df['relative_diff'].abs(), color=colors)
    ax.set_xlabel('Relative Difference')
    ax.set_title('Relative Difference Between Human and AI Text Features')

    # Adjust ylim to remove whitespace
    ax.set_ylim(-0.5, len(summary_df) - 0.5)
    ax.invert_yaxis()  # Invert y-axis to have the largest differences on top

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightcoral', edgecolor='firebrick', label='AI > Human'),
                       Patch(facecolor='lightblue', edgecolor='royalblue', label='Human > AI')]
    ax.legend(handles=legend_elements, loc='lower right')

    # Adjust layout to remove whitespace
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(output_plot_path, bbox_inches='tight')
    plt.show()

def calculate_statistics(data):
    """Calculate various statistical measures for a given dataset."""
    stats = {}
    stats['Mean'] = data.mean()
    stats['Median'] = data.median()
    stats['Standard Deviation'] = data.std()
    stats['IQR'] = data.quantile(0.75) - data.quantile(0.25)
    stats['Trimmed Mean (10%)'] = trim_mean(data, 0.1)
    stats['Winsorized Mean (10%)'] = mstats.winsorize(data, limits=[0.1, 0.1]).mean()
    return stats

def generate_perplexity_table(output_csv_path='data/summary_statistics.csv'):
    # Load main dataset
    main_features = load_features('data/saved_data/full/features.pkl')
    main_labels = pd.read_pickle('data/saved_data/full/labels.pkl')

    # Ensure the features and labels have the same length
    assert len(main_features) == len(main_labels), "Features and labels length mismatch"

    # Split main dataset into human and AI texts based on the labels (0 for human, 1 for AI)
    human_features = main_features[main_labels == 0]
    ai_features = main_features[main_labels == 1]

    # Load evasive texts features
    control_features = load_features('data/evasive_texts/feature/control/full_features.pkl')
    basic_features = load_features('data/evasive_texts/feature/basic/full_features.pkl')
    advanced_features = load_features('data/evasive_texts/feature/advanced/full_features.pkl')
    
    # Calculate statistics for all datasets
    human_stats = calculate_statistics(human_features['max_perplexity'])
    ai_stats = calculate_statistics(ai_features['max_perplexity'])
    control_stats = calculate_statistics(control_features['max_perplexity'])
    basic_stats = calculate_statistics(basic_features['max_perplexity'])
    advanced_stats = calculate_statistics(advanced_features['max_perplexity'])

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'Human Texts': human_stats,
        'AI Texts': ai_stats,
        'Control Evasive': control_stats,
        'Basic Evasive': basic_stats,
        'Advanced Evasive': advanced_stats
    })

    # Round the values to 3 decimal places
    summary_df = summary_df.round(3)

    # Set display options to avoid scientific notation
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    # Save the summary DataFrame to a CSV file
    summary_df.to_csv(output_csv_path, index=True)

    # Print the summary DataFrame
    print(summary_df)

    return summary_df
