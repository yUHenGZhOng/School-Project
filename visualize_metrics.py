
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import itertools
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        exit()

def plot_convergence_metrics(df, emotion_cols):
    """Calculates and visualizes group emotional convergence metrics."""
    print("Calculating convergence metrics...")
    person_ids = df['person_id'].unique()
    distance_matrix = pd.DataFrame(np.nan, index=person_ids, columns=person_ids)

    # Heatmap Calculation
    # First, ensure there are no duplicate person_id entries for any frame
    df_unique = df.drop_duplicates(subset=['frame_number', 'person_id'])

    for id1, id2 in itertools.combinations(person_ids, 2):
        # Isolate data for the two individuals
        df1 = df_unique[df_unique['person_id'] == id1][['frame_number'] + emotion_cols]
        df2 = df_unique[df_unique['person_id'] == id2][['frame_number'] + emotion_cols]

        # Merge to find common frames and ensure alignment
        merged_df = pd.merge(df1, df2, on='frame_number', suffixes=['_1', '_2'])

        if merged_df.empty:
            continue

        emotion_cols_1 = [col + '_1' for col in emotion_cols]
        emotion_cols_2 = [col + '_2' for col in emotion_cols]

        # Calculate Euclidean distance for each common frame
        distances = np.linalg.norm(merged_df[emotion_cols_1].values - merged_df[emotion_cols_2].values, axis=1)
        
        avg_distance = distances.mean()
        distance_matrix.loc[id1, id2] = avg_distance
        distance_matrix.loc[id2, id1] = avg_distance

    plt.figure(figsize=(12, 10))
    sns.heatmap(distance_matrix, annot=True, fmt=".2f", cmap="viridis_r") # _r reverses the colormap
    plt.title('Average Emotional Distance Between Individuals (Emotional Allies)')
    plt.xlabel('Person ID')
    plt.ylabel('Person ID')
    plt.savefig('emotional_allies_heatmap.png')
    plt.close()
    print("Saved emotional_allies_heatmap.png")

    # Line Plot Calculation (Affective Dispersion)
    dispersion_scores = []
    for frame, group in df.groupby('frame_number'):
        if len(group) > 1:
            distances = pdist(group[emotion_cols].values, 'euclidean')
            dispersion_scores.append({'frame_number': frame, 'dispersion': np.mean(distances)})
    
    dispersion_df = pd.DataFrame(dispersion_scores)
    
    plt.figure(figsize=(15, 7))
    sns.lineplot(data=dispersion_df, x='frame_number', y='dispersion')
    plt.title('Affective Dispersion Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Average Emotional Distance (Dispersion)')
    plt.savefig('affective_dispersion_timeseries.png')
    plt.close()
    print("Saved affective_dispersion_timeseries.png")

def plot_valence_metrics(df):
    """Calculates and visualizes group affective valence."""
    print("Calculating valence metrics...")
    valence_data = []
    negative_emotions = ['angry', 'sad', 'fear', 'disgust']

    for frame, group in df.groupby('frame_number'):
        positive_valence = group['happy'].mean()
        negative_valence = group[negative_emotions].sum(axis=1).mean()
        valence_data.append({
            'frame_number': frame,
            'positive_valence': positive_valence,
            'negative_valence': negative_valence
        })
    
    valence_df = pd.DataFrame(valence_data)
    
    plt.figure(figsize=(15, 7))
    sns.lineplot(data=valence_df, x='frame_number', y='positive_valence', label='Positive Valence', color='green')
    sns.lineplot(data=valence_df, x='frame_number', y='negative_valence', label='Negative Valence', color='red')
    plt.title('Group Affective Valence Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Average Emotion Probability')
    plt.legend()
    plt.savefig('affective_valence_timeseries.png')
    plt.close()
    print("Saved affective_valence_timeseries.png")


def plot_dynamics_metrics(df, emotion_cols):
    """Calculates and visualizes group emotional dynamics."""
    print("Calculating dynamics metrics...")
    dominant_emotions = []
    for frame, group in df.groupby('frame_number'):
        group_avg_emotion = group[emotion_cols].mean()
        dominant_emotion = group_avg_emotion.idxmax()
        dominant_emotions.append({'frame_number': frame, 'dominant_emotion': dominant_emotion})
    
    dynamics_df = pd.DataFrame(dominant_emotions)

    # Visualization
    emotion_color_map = {
        'happy': 'green',
        'sad': 'blue',
        'angry': 'red',
        'surprise': 'purple',
        'fear': 'gray',
        'disgust': 'brown',
        'neutral': 'lightgray'
    }
    
    fig, ax = plt.subplots(figsize=(20, 2))
    ax.set_yticks([])
    ax.set_xlabel('Frame Number')
    ax.set_title('Group Dominant Emotion Over Time')

    for i in range(len(dynamics_df) - 1):
        start_frame = dynamics_df.iloc[i]['frame_number']
        end_frame = dynamics_df.iloc[i+1]['frame_number']
        emotion = dynamics_df.iloc[i]['dominant_emotion']
        ax.axvspan(start_frame, end_frame, color=emotion_color_map[emotion], alpha=0.7)

    # Add legend
    legend_patches = [plt.Rectangle((0,0),1,1, color=color) for color in emotion_color_map.values()]
    ax.legend(legend_patches, emotion_color_map.keys(), bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('affective_dynamics_carpetplot.png', bbox_inches='tight')
    plt.close()
    print("Saved affective_dynamics_carpetplot.png")


def main():
    """Main function to run the visualization script."""
    parser = argparse.ArgumentParser(description="Visualize metrics from cleaned emotion analysis data.")
    parser.add_argument(
        '--input', 
        type=str, 
        required=True, 
        help='Path to the cleaned CSV data file.'
    )
    args = parser.parse_args()

    df = load_data(args.input)
    
    # Ensure person_id is treated as a categorical or object type for correct grouping
    df['person_id'] = df['person_id'].astype(str)

    emotion_cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    plot_convergence_metrics(df, emotion_cols)
    plot_valence_metrics(df)
    plot_dynamics_metrics(df, emotion_cols)

    print("\nAll visualizations have been successfully generated.")

if __name__ == '__main__':
    main()
