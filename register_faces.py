
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

    # Dictionary containing all registered identities
    known_faces = {}

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    # Iterate through each person's directory
    person_dirs = sorted(
        (p for p in DATASET_DIR.iterdir() if p.is_dir()),
        key=lambda p: p.name.casefold()
    )

    for person_dir in person_dirs:

        # Temporary storage for the current person's embeddings
        person_features = []

        # Collect supported images
        image_paths = sorted(
            [
                p for p in person_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda p: p.name.casefold()
        )

        for image_path in image_paths:

            # Load image
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Cannot read: {image_path}")
                continue

            # Detect faces using YuNet
            detections = detector.detect(image)

            # Require exactly one face per image
            if len(detections) != 1:
                print(
                    f"Skipping {image_path.name}: "
                    f"found {len(detections)} faces"
                )
                continue

            try:
                # Extract normalized face embedding using SFace
                embedding = recognizer.extract_embedding(
                    image,
                    detections[0]
                )

            except (ValueError, cv2.error) as error:
                print(
                    f"Embedding extraction failed for "
                    f"{image_path.name}: {error}"
                )
                continue

            # Store embedding temporarily
            person_features.append(embedding)

        # Skip identity if no valid embeddings were generated
        if not person_features:
            print(f"No valid images for {person_dir.name}")
            continue

        # Convert person's embeddings into a NumPy matrix
        person_embeddings = np.stack(
            person_features
        ).astype(np.float32)

        # Store identity and its embeddings
        known_faces[person_dir.name] = person_embeddings

        print(
            f"Registered {person_dir.name}: "
            f"{len(person_features)} embeddings"
        )

    # Ensure at least one identity was registered
    if not known_faces:
        raise RuntimeError(
            "No valid face embeddings were generated."
        )

    # Create output directory
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Prepare a pickle-free NPZ archive
    archive = {
        "names": np.asarray(
            list(known_faces.keys()),
            dtype=str
        )
    }

    # Store each person's embedding matrix separately
    for index, (name, embeddings) in enumerate(
        known_faces.items()
    ):
        archive[f"embeddings_{index}"] = embeddings

    # Save database
    np.savez_compressed(
        OUTPUT_PATH,
        **archive
    )

    total_embeddings = sum(
        len(embeddings)
        for embeddings in known_faces.values()
    )

    print("\nRegistration completed!")
    print(f"Registered people: {len(known_faces)}")
    print(f"Total embeddings: {total_embeddings}")
    print(f"Database saved at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
