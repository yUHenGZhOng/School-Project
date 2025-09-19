
import cv2
import pandas as pd
import argparse
import os
import shutil
from processing_engine import EmotionProcessor

def main(video_path, output_dir, interval):
    """
    Main function to process the video and save results.
    """
    # 0. Derive video-specific paths
    video_name_stem = os.path.splitext(os.path.basename(video_path))[0]
    face_save_path = os.path.join('output/faces', video_name_stem)
    output_csv_path = os.path.join(output_dir, f"{video_name_stem}_analysis.csv")

    # Clear and recreate the specific output directory for this video's face crops
    if os.path.exists(face_save_path):
        shutil.rmtree(face_save_path)
    os.makedirs(face_save_path)
    print(f"Cleared and recreated directory: {face_save_path}")

    # 1. Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    # 2. Initialize processor with the correct path for saving faces
    processor = EmotionProcessor(face_save_path=face_save_path)
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
        
        # Skip frames that are not on the sampling interval
        if frame_number % frame_interval != 0:
            continue

        # 5. Process the frame
        results_for_frame = processor.process_frame(frame, frame_number)
        
        # 6. Print the consolidated log message
        num_faces = len(results_for_frame)
        print(f"正在分析第 {frame_number} 帧，检测到 {num_faces} 个人脸")

        # 7. Add timestamp and append results to the main list
        if results_for_frame:
            timestamp_sec = round(frame_number / fps, 2)
            for r in results_for_frame:
                r['timestamp'] = timestamp_sec
            all_results.extend(results_for_frame)

    cap.release()
    print("\nVideo processing finished.")

    # 7. Analyze results and save to a dynamically named CSV
    if all_results:
        print("Analyzing results to generate filename...")
        df = pd.DataFrame(all_results)

        # --- Dynamic Filename Generation ---
        # a. Calculate theoretical max samples
        video_duration_sec = total_frames / fps
        theoretical_max_samples = int(video_duration_sec / interval)

        # b. Find actual max samples for any single person
        if 'person_id' in df.columns:
            person_counts = df['person_id'].value_counts()
            actual_max_samples = person_counts.max() if not person_counts.empty else 0
        else:
            actual_max_samples = 0

        # c. Calculate coverage percentage
        coverage_percentage = (actual_max_samples / theoretical_max_samples * 100) if theoretical_max_samples > 0 else 0

        # d. Construct the new filename
        filename_stats = f"{actual_max_samples}-{theoretical_max_samples}_{coverage_percentage:.1f}%_{int(interval)}s"
        output_csv_path = os.path.join(output_dir, f"{video_name_stem}_analysis_{filename_stats}.csv")
        # --- End of Dynamic Filename Generation ---

        print(f"Saving data to CSV: {output_csv_path}")
        
        # Flatten the 'emotions' dictionary into separate columns
        emotion_df = df['emotions'].apply(pd.Series)
        
        # Concatenate the emotion columns with the main dataframe
        df = pd.concat([df.drop(columns=['emotions']), emotion_df], axis=1)

        # Reorder columns for better readability
        base_cols = ['frame_number', 'timestamp', 'person_id', 'bbox', 'dominant_emotion']
        emotion_cols = [col for col in emotion_df.columns if col in df.columns]
        final_cols = base_cols + emotion_cols
        df = df[final_cols]

        # --- Append Summary --- 
        summary_df = person_counts.reset_index()
        summary_df.columns = ['person_id', 'total_samples']
        summary_df.loc[len(summary_df)] = ['---', '---'] # Separator
        summary_df = summary_df.rename(columns={'person_id': 'Summary: person_id', 'total_samples': 'count'})

        # Convert main df to string to append summary, avoiding dtype conflicts
        df = df.astype(str)
        summary_df = summary_df.astype(str)

        # Align columns for concatenation
        summary_df_aligned = pd.DataFrame(columns=df.columns)
        summary_df_aligned[['Summary: person_id', 'count']] = summary_df

        final_df = pd.concat([df, summary_df_aligned], ignore_index=True)
        # --- End of Append Summary ---

        final_df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved analysis data to '{output_csv_path}'")
    else:
        print("No faces were detected or tracked in the sampled frames. No data to save.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze emotions in a video file and save the results to a CSV.')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', type=str, default='output', help='Path to the output directory for the CSV file. Default is \'output\'.')
    parser.add_argument('--interval', type=float, default=1.0, help='Interval in seconds for frame analysis. Default is 1.0s.')
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    main(args.video, args.output, args.interval)
