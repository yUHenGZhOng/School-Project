
import cv2
import numpy as np
import os
from scipy.spatial.distance import cosine
from deepface import DeepFace
from sort_local import Sort

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou

class EmotionProcessor:
    def __init__(self, reid_threshold=0.4):
        """
        Initializes the EmotionProcessor.
        - reid_threshold: Cosine distance threshold for re-identification.
        """
        self.tracker = Sort()
        self.face_database = {}  # Stores {person_id: [list of embeddings]}
        self.next_person_id = 0
        self.reid_threshold = reid_threshold
        self.track_to_person = {}  # Maps temporary track_id from SORT to our permanent person_id

        # Setup directory for saving face crops
        self.face_save_path = 'output/faces'
        os.makedirs(self.face_save_path, exist_ok=True)

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
        Processes a single frame to detect, track, re-identify, and analyze faces.
        """
        results_data = []
        
        # Step 1: Detect all faces and their emotions in one go
        try:
            # Use a more reliable detector and get region and emotion
            all_faces = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                detector_backend='retinaface', 
                enforce_detection=True
            )
        except Exception:
            # No faces detected in the frame
            all_faces = []

        # Prepare detections for SORT
        detections_for_sort = []
        for face in all_faces:
            r = face['region']
            # Format for SORT: [x1, y1, x2, y2, score]
            detections_for_sort.append([r['x'], r['y'], r['x'] + r['w'], r['y'] + r['h'], face.get('confidence', 0.99)])

        detections_np = np.array(detections_for_sort) if detections_for_sort else np.empty((0, 5))
        
        # Step 2: Update motion tracker
        tracked_objects = self.tracker.update(detections_np)
        
        current_track_ids = set()

        # Step 3: Process each tracked object
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            current_track_ids.add(track_id)
            
            person_id = -1
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # Check if this track is already associated with a person
            if track_id in self.track_to_person:
                person_id = self.track_to_person[track_id]
            else:
                # New track, perform re-identification
                try:
                    # Get face embedding for re-identification
                    embedding_obj = DeepFace.represent(
                        face_crop, 
                        model_name="Facenet512", 
                        enforce_detection=False, # Crop is already a face
                        detector_backend='skip'
                    )
                    if embedding_obj and len(embedding_obj) > 0:
                        embedding = embedding_obj[0]["embedding"]
                        person_id = self._reidentify_and_update_database(embedding)
                        self.track_to_person[track_id] = person_id
                except Exception as e:
                    # This can happen if the crop is too small or invalid
                    print(f"Could not get embedding for new track {track_id}: {e}")

            if person_id == -1:
                continue

            # Save the face crop for every processed frame where a person is identified
            try:
                # Make filename unique by including coordinates to avoid overwrites
                face_filename = os.path.join(self.face_save_path, f"{person_id}_{frame_number}_{x1}_{y1}.jpg")
                face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(face_filename, face_crop_bgr)
            except Exception as e:
                print(f"Could not save face for person {person_id} at frame {frame_number}: {e}")

            # Step 4: Find the corresponding emotion data from the initial analysis
            best_match_face = None
            max_iou = 0.0
            tracked_box = [x1, y1, x2, y2]
            for face in all_faces:
                r = face['region']
                initial_box = [r['x'], r['y'], r['x'] + r['w'], r['y'] + r['h']]
                iou = calculate_iou(tracked_box, initial_box)
                if iou > max_iou:
                    max_iou = iou
                    best_match_face = face
            
            # If we found a matching face, record its data
            if best_match_face and max_iou > 0.5: # Use an IoU threshold
                results_data.append({
                    'frame_number': frame_number,
                    'person_id': person_id,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'emotions': best_match_face['emotion']
                })

        # Clean up old tracks from the mapping
        self.track_to_person = {tid: pid for tid, pid in self.track_to_person.items() if tid in current_track_ids}

        return results_data
