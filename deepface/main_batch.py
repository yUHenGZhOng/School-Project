
import cv2
import pandas as pd
import argparse
import os
import shutil
from processing_engine import EmotionProcessor

def main(video_path, output_csv, interval):
    """
    Main function to process the video and save results.
    """
    # 0. Clear and recreate the output directory for face crops
    face_save_path = 'output/faces'
    if os.path.exists(face_save_path):
        shutil.rmtree(face_save_path)
    os.makedirs(face_save_path)
    print(f"Cleared and recreated directory: {face_save_path}")

    # 1. Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    # 2. Initialize processor and data list
    processor = EmotionProcessor()
    all_results = []
    
    # 3. Open video and get properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate frame interval for sampling
    frame_interval = int(round(interval * fps))
    if frame_interval < 1:
        frame_interval = 1

    print(f"Video FPS: {fps:.2f}, Processing every {frame_interval} frames (approx. every {interval} seconds).")

    frame_number = 0
    print("Starting video processing...")

    # 4. Loop through video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        
        # Print progress, but only update on non-processed frames to keep it clean
        if frame_number % frame_interval != 0:
            print(f"Processing frame {frame_number} / {total_frames}", end='\r')
            continue

        print(f"Analyzing frame {frame_number} / {total_frames}...")

        # 5. Process the frame, passing the frame number
        results_for_frame = processor.process_frame(frame, frame_number)
        
        # 6. Append results to the main list
        if results_for_frame:
            all_results.extend(results_for_frame)

    cap.release()
    # Print a newline character to move to the next line after the progress indicator
    print("\nVideo processing finished.")

    # 7. Save results to CSV
    if all_results:
        print("Saving data to CSV...")
        df = pd.DataFrame(all_results)
        
        # Flatten the 'emotions' dictionary into separate columns
        emotion_df = df['emotions'].apply(pd.Series)
        
        # Concatenate the emotion columns with the main dataframe
        df = pd.concat([df.drop(columns=['emotions']), emotion_df], axis=1)

        # Reorder columns for better readability
        base_cols = ['frame_number', 'person_id', 'bbox', 'dominant_emotion']
        emotion_cols = list(emotion_df.columns)
        final_cols = base_cols + emotion_cols
        df = df[final_cols]

        df.to_csv(output_csv, index=False)
        print(f"Successfully saved analysis data to '{output_csv}'")
    else:
        print("No faces were detected or tracked in the sampled frames. No data to save.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze emotions in a video file and save the results to a CSV.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', type=str, default='output.csv', help='Path for the output CSV file. Default is output.csv')
    parser.add_argument('--interval', type=float, default=1.0, help='Interval in seconds for frame analysis. Default is 1.0s.')
    
    args = parser.parse_args()
    
    main(args.video, args.output, args.interval)
