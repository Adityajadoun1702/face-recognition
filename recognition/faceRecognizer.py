import cv2
from pathlib import Path
import numpy as np
class FaceRecognizer:
    def __init__(self,threshold):
        model_path=(Path(__file__).resolve().parents[1]
        /"models"
        /"model_weights"
        /"face_recognition_sface_2021dec.onnx"
        )
        if not model_path:
            raise ValueError(f"No model found {model_path}")
        if not hasattr(cv2,"FaceRecognizerSF"):
            raise ValueError("cv2 has no FaceRecognizerSF")
        self.model=cv2.FaceRecognizerSF.create(
            model=model_path,
            config="",
        )
        
        self.threshold=threshold
    def extract_embeddings(self,frame,detection):
        face_data=np.asarray(
            detection["yunet_face"],
            dtype=np.float32,
        ).reshape(1,15)

        aligned_face=self.model.alignCrop(
            frame,
            face_data
        )

        embedding=self.model.feature(aligned_face)

        embedding=np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(-1)

        norm=np.linalg.norm(embedding)

        if norm<1e-12:
            raise ValueError("Invalid face embedding")

        return embedding/norm

    def identify(self, query_embedding, names, known_embeddings, top_k=2):

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(-1)

        known_embeddings = np.asarray(
            known_embeddings,
            dtype=np.float32
        )

        names = np.asarray(names, dtype=str)

        if len(names) == 0 or len(known_embeddings) == 0:
            raise ValueError("No registered faces available")

        if known_embeddings.ndim != 2:
            raise ValueError("Expected a 2D embedding matrix")

        if len(names) != len(known_embeddings):
            raise ValueError("Names and embeddings are mismatched")

        if known_embeddings.shape[1] != query_embedding.size:
            raise ValueError("Embedding dimensions do not match")

        unique_names = np.unique(names)

        best_name = "Unknown"
        best_score = -1.0

        for name in unique_names:

            # Find all embeddings belonging to this person
            person_indices = np.flatnonzero(names == name)

            person_embeddings = known_embeddings[person_indices]

            # Calculate similarity against all their embeddings
            similarities = person_embeddings @ query_embedding

            # Select the top K scores
            k = min(top_k, len(similarities))

            top_scores = np.sort(similarities)[-k:]

            # Calculate average of top K
            person_score = float(np.mean(top_scores))

            if person_score > best_score:
                best_score = person_score
                best_name = name

        if best_score >= self.threshold:
            return best_name, best_score

        return "Unknown", best_score
