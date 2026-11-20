
import cv2
import numpy as np
from pathlib import Path


class FaceRecognizer:

    def __init__(self, model_path, threshold=0.363):

        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"SFace model not found: {model_path}"
            )

        if not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError(
                "FaceRecognizerSF is unavailable. "
                "Check your OpenCV installation."
            )

        # Load pretrained SFace model
        self.model = cv2.FaceRecognizerSF.create(
            str(model_path),
            ""
        )

        self.threshold = threshold

    def extract_embedding(self, frame, detection):

        # Retrieve complete YuNet detection information
        face_data = np.asarray(
            detection["yunet_face"],
            dtype=np.float32
        ).reshape(1, 15)

        # Align and crop the detected face
        aligned_face = self.model.alignCrop(
            frame,
            face_data
        )

        if aligned_face is None or aligned_face.size == 0:
            raise ValueError("Face alignment failed")

        # Extract facial features using SFace
        embedding = self.model.feature(aligned_face)

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(-1)

        # L2 normalization
        norm = np.linalg.norm(embedding)

        if norm < 1e-12:
            raise ValueError("Invalid face embedding")

        return embedding / norm

    def identify(self,query_embedding,known_names,known_embeddings,top_k=2):

        # Validate top_k
        if not isinstance(top_k, (int, np.integer)) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        # Convert inputs into NumPy arrays
        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(-1)

        known_names = np.asarray(
            known_names,
            dtype=str
        ).reshape(-1)

        known_embeddings = np.asarray(
            known_embeddings,
            dtype=np.float32
        )

        # Validate database
        if known_embeddings.ndim != 2:
            raise ValueError("Expected a 2D embedding matrix")

        if len(known_names) == 0 or len(known_embeddings) == 0:
            raise ValueError("No registered faces available")

        if len(known_names) != len(known_embeddings):
            raise ValueError(
                "Number of names and embeddings must match"
            )

        if query_embedding.size != known_embeddings.shape[1]:
            raise ValueError("Embedding dimensions do not match")

        if not np.all(np.isfinite(query_embedding)):
            raise ValueError("Query embedding contains invalid values")

        if not np.all(np.isfinite(known_embeddings)):
            raise ValueError("Database contains invalid embeddings")

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)

        if query_norm < 1e-12:
            raise ValueError("Invalid query embedding")

        query_embedding = query_embedding / query_norm

        # Normalize database embeddings
        norms = np.linalg.norm(
            known_embeddings,
            axis=1,
            keepdims=True
        )

        if np.any(norms < 1e-12):
            raise ValueError("Database contains zero embeddings")

        known_embeddings = known_embeddings / norms

        # Step 1: Compare query against every stored embedding
        similarities = known_embeddings @ query_embedding

        # Step 2: Find unique identities and their group indices
        unique_names, inverse = np.unique(
            known_names,
            return_inverse=True
        )

        # Step 3: Sort by identity, then descending similarity
        order = np.lexsort((-similarities, inverse))

        sorted_groups = inverse[order]
        sorted_scores = similarities[order]

        # Step 4: Calculate each embedding's rank within its identity
        positions = np.arange(len(sorted_scores))

        new_group = np.r_[
            True,
            sorted_groups[1:] != sorted_groups[:-1]
        ]

        group_starts = np.maximum.accumulate(
            np.where(new_group, positions, 0)
        )

        ranks = positions - group_starts

        # Step 5: Select top K scores from each identity
        selected = ranks < top_k

        # Step 6: Calculate average top K score per identity
        score_sums = np.bincount(
            sorted_groups[selected],
            weights=sorted_scores[selected],
            minlength=len(unique_names)
        )

        score_counts = np.bincount(
            sorted_groups[selected],
            minlength=len(unique_names)
        )

        person_scores = score_sums / score_counts

        # Step 7: Find the best matching identity
        best_index = int(np.argmax(person_scores))

        best_name = str(unique_names[best_index])
        best_score = float(person_scores[best_index])

        # Step 8: Apply recognition threshold
        if best_score >= self.threshold:
            return best_name, best_score

        return "Unknown", best_score
