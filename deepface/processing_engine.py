import cv2
import numpy as np
import os
from scipy.spatial.distance import cosine
from deepface import DeepFace

# The IOU function is no longer needed as we are not matching tracker boxes.

class EmotionProcessor:
    def __init__(self, reid_threshold=0.32, face_save_path='output/faces'):
        """
        Initializes the EmotionProcessor.
        - reid_threshold: Cosine distance threshold for re-identification.
        - face_save_path: The directory where face crops will be saved.
        """
        self.face_database = {}  # Stores {person_id: [list of embeddings]}
        self.next_person_id = 0
        self.reid_threshold = reid_threshold
        self.face_save_path = face_save_path

    def _reidentify_and_update_database(self, face_embedding):
        """
        Finds an existing person_id for the face embedding or creates a new one.
        Returns the person_id.
        """
        if not self.face_database:
            new_person_id = self.next_person_id
            self.face_database[new_person_id] = [face_embedding]
            self.next_person_id += 1
            return new_person_id

        best_match_id = -1
        min_dist = self.reid_threshold

        for person_id, embeddings in self.face_database.items():
            # Compare against the average embedding for this person for stability
            avg_embedding = np.mean(embeddings, axis=0)
            dist = cosine(face_embedding, avg_embedding)
            if dist < min_dist:
                min_dist = dist
                best_match_id = person_id
        
        if best_match_id != -1:
            self.face_database[best_match_id].append(face_embedding)
            return best_match_id
        else:
            new_person_id = self.next_person_id
            self.face_database[new_person_id] = [face_embedding]
            self.next_person_id += 1
            return new_person_id

    def process_frame(self, frame: np.ndarray, frame_number: int):
        """
        Processes a single frame to detect, re-identify, and analyze faces.
        This function is designed to be 'silent' on success, only printing errors.
        """
        results_data = []

        # Step 1: Detect all faces and their emotions in one go
        try:
            # The frame read by cv2 is BGR, convert to RGB for deepface functions
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_faces = DeepFace.analyze(
                frame_rgb,
                actions=['emotion'],
                detector_backend='retinaface',
                enforce_detection=False
            )
        except Exception as e:
            print(f"An error occurred during DeepFace.analyze on frame {frame_number}: {e}")
            all_faces = []

        if not all_faces or not isinstance(all_faces, list) or len(all_faces) == 0:
            return []

        # Step 2: Process each detected face directly
        for face_data in all_faces:
            if not isinstance(face_data, dict) or 'region' not in face_data:
                continue

            r = face_data['region']
            x, y, w, h = r['x'], r['y'], r['w'], r['h']
            
            # Use the original BGR frame for cropping with cv2, as imwrite expects BGR.
            face_crop_bgr = frame[y:y+h, x:x+w]

            if face_crop_bgr.size == 0:
                print(f"Warning: Created an empty face crop for a face in frame {frame_number}. Skipping.")
                continue

            person_id = -1

            # Step 3: Get embedding and re-identify
            try:
                # DeepFace.represent also expects an RGB image.
                embedding_obj = DeepFace.represent(
                    face_crop_bgr, # Pass BGR, DeepFace handles conversion internally
                    model_name="Facenet512",
                    enforce_detection=False, # The crop is already a face
                    detector_backend='skip'
                )
                if embedding_obj and len(embedding_obj) > 0:
                    embedding = embedding_obj[0]["embedding"]
                    person_id = self._reidentify_and_update_database(embedding)
            except Exception as e:
                print(f"Could not get embedding for a face in frame {frame_number}: {e}")

            if person_id == -1:
                continue

            # Save the face crop (using the BGR crop)
            try:
                face_filename = os.path.join(self.face_save_path, f"{person_id}_{frame_number}_{x}_{y}.jpg")
                cv2.imwrite(face_filename, face_crop_bgr)
            except Exception as e:
                print(f"Could not save face for person {person_id} at frame {frame_number}: {e}")

            # Step 4: Append results for this face
            results_data.append({
                'frame_number': frame_number,
                'person_id': person_id,
                'bbox': [x, y, w, h],
                'dominant_emotion': face_data['dominant_emotion'],
                'emotions': face_data['emotion']
            })

        return results_data