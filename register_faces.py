
import cv2
import numpy as np
from pathlib import Path

from detector.faceDetector import FaceDetector
from recognition.faceRecognizer import FaceRecognizer


ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT / "dataset"

SFACE_MODEL = (
    ROOT
    / "models"
    / "model_weights"
    / "face_recognition_sface_2021dec.onnx"
)

OUTPUT_PATH = ROOT / "embeddings" / "known_faces.npz"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():

    detector = FaceDetector()
    recognizer = FaceRecognizer(SFACE_MODEL)

    all_names = []
    all_embeddings = []

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    for person_dir in sorted(DATASET_DIR.iterdir()):

        if not person_dir.is_dir():
            continue

        person_features = []

        image_paths = [
            p for p in person_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_path in image_paths:

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Cannot read: {image_path}")
                continue

            detections = detector.detect(image)

            if len(detections) != 1:
                print(
                    f"Skipping {image_path.name}: "
                    f"found {len(detections)} faces"
                )
                continue

            embedding = recognizer.extract_embedding(
                image,
                detections[0]
            )

            person_features.append(embedding)

        if not person_features:
            print(f"No valid images for {person_dir.name}")
            continue

        for embedding in person_features:
            all_names.append(person_dir.name)
            all_embeddings.append(embedding)

        print(
            f"Registered {person_dir.name}: "
            f"{len(person_features)} images"
        )

    if not all_embeddings:
        raise RuntimeError(
            "No valid face embeddings were generated."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,
        names=np.asarray(all_names, dtype=str),
        embeddings=np.stack(all_embeddings)
    )

    print(f"\nRegistration completed: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
