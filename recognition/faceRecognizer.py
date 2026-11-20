
import cv2
import numpy as np
from pathlib import Path


class FaceRecognizer:
    def __init__(self, model_path, threshold=0.363, top_k=2):

        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"SFace model not found: {model_path}"
            )
        if not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError(
                "FaceRecognizerSF is unavailable. "
                "Install a compatible opencv-contrib-python version."
            )

        self.model = cv2.FaceRecognizerSF.create(
            str(model_path),
            ""
        )
        if not isinstance(top_k, (int, np.integer)) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between -1 and 1")
        self.threshold = float(threshold)
        self.top_k = int(top_k)
    def extract_embedding(self, frame, detection):
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame")
        face_data = np.asarray(
            detection["yunet_face"],
            dtype=np.float32
        ).reshape(1, 15)
        if not np.all(np.isfinite(face_data)):
            raise ValueError("Invalid YuNet detection data")
        aligned_face = self.model.alignCrop(
            frame,
            face_data
        )
        if aligned_face is None or aligned_face.size == 0:
            raise ValueError("Face alignment failed")
        
        embedding = self.model.feature(aligned_face)

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(-1)

        if not np.all(np.isfinite(embedding)):
            raise ValueError("Embedding contains invalid values")

        norm = np.linalg.norm(embedding)

        if norm < 1e-12:
            raise ValueError("Invalid face embedding")

        return embedding / norm

    def identify(self, query_embedding, known_faces):

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(-1)

        if not np.all(np.isfinite(query_embedding)):
            raise ValueError("Invalid query embedding")

        query_norm = np.linalg.norm(query_embedding)

        if query_norm < 1e-12:
            raise ValueError("Query embedding has zero magnitude")

        query_embedding = query_embedding / query_norm

        if not known_faces:
            raise ValueError("No registered faces available")

        best_name = "Unknown"
        best_score = -1.0
        valid_people = 0


        for name, embeddings in known_faces.items():

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32
            )

            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            if embeddings.ndim != 2:
                raise ValueError(
                    f"Invalid embedding matrix for {name}"
                )

            if len(embeddings) == 0:
                continue

            if embeddings.shape[1] != query_embedding.size:
                raise ValueError(
                    f"Embedding dimensions mismatch for {name}"
                )

            if not np.all(np.isfinite(embeddings)):
                raise ValueError(
                    f"Invalid embeddings found for {name}"
                )

            similarities = embeddings @ query_embedding

            k = min(self.top_k, len(similarities))

            top_scores = np.partition(
                similarities,
                len(similarities) - k
            )[-k:]

            person_score = float(np.mean(top_scores))

            valid_people += 1

            if person_score > best_score:
                best_score = person_score
                best_name = name

        if valid_people == 0:
            raise ValueError("No valid registered embeddings found")

        if best_score >= self.threshold:
            return best_name, best_score

        return "Unknown", best_score
