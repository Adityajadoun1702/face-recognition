
import cv2
import numpy as np
from pathlib import Path

from videoCapture import VideoCapture
from detector.faceDetector import FaceDetector
from recognition.faceRecognizer import FaceRecognizer


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

SFACE_MODEL = (
    ROOT
    / "models"
    / "model_weights"
    / "face_recognition_sface_2021dec.onnx"
)

DATABASE_PATH = (
    ROOT
    / "embeddings"
    / "known_faces.npz"
)

CAMERA_INDEX = 0

# Provisional value: calibrate for Top-K matching
SIMILARITY_THRESHOLD = 0.5

TOP_K = 2


# --------------------------------------------------
# 2. LOAD REGISTERED FACE DATABASE
# --------------------------------------------------

def load_face_database(database_path):

    if not database_path.is_file():
        raise FileNotFoundError(
            f"Face database not found: {database_path}\n"
            "Run register_faces.py first."
        )

    known_faces = {}

    with np.load(
        database_path,
        allow_pickle=False
    ) as data:

        if "names" not in data.files:
            raise ValueError("Database does not contain names")

        names = data["names"].astype(str)

        for index, name in enumerate(names):

            key = f"embeddings_{index}"

            if key not in data.files:
                raise ValueError(
                    f"Missing embedding matrix: {key}"
                )

            embeddings = np.asarray(
                data[key],
                dtype=np.float32
            )

            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            if embeddings.ndim != 2 or len(embeddings) == 0:
                raise ValueError(
                    f"Invalid embedding matrix for {name}"
                )

            if not np.all(np.isfinite(embeddings)):
                raise ValueError(
                    f"Invalid numerical values for {name}"
                )

            if name in known_faces:
                raise ValueError(
                    f"Duplicate identity in database: {name}"
                )

            # Normalize once when loading the database
            norms = np.linalg.norm(
                embeddings,
                axis=1,
                keepdims=True
            )

            if np.any(norms < 1e-12):
                raise ValueError(
                    f"Zero embedding found for {name}"
                )

            embeddings = embeddings / norms

            known_faces[name] = embeddings

    if not known_faces:
        raise ValueError("Face database is empty")

    total_embeddings = sum(
        len(embeddings)
        for embeddings in known_faces.values()
    )

    print("\nFace database loaded successfully")
    print(f"Registered people: {len(known_faces)}")
    print(f"Total embeddings: {total_embeddings}")

    return known_faces


# --------------------------------------------------
# 3. MAIN APPLICATION
# --------------------------------------------------

def main():

    # Load registered identities
    known_faces = load_face_database(DATABASE_PATH)

    # Initialize YuNet
    detector = FaceDetector()

    # Initialize SFace
    recognizer = FaceRecognizer(
        SFACE_MODEL,
        threshold=SIMILARITY_THRESHOLD,
        top_k=TOP_K
    )

    camera = None

    try:

        # Open webcam
        camera = VideoCapture(CAMERA_INDEX)

        print("\nFace recognition started")
        print("Press Q or ESC to exit")

        # ------------------------------------------
        # 4. CONTINUOUS FRAME PROCESSING
        # ------------------------------------------

        for ret, frame in camera.get_frames():

            if not ret or frame is None:
                print("Unable to capture frame")
                break

            # --------------------------------------
            # 5. DETECT FACES USING YUNET
            # --------------------------------------

            detections = detector.detect(frame)

            # --------------------------------------
            # 6. RECOGNIZE EACH DETECTED FACE
            # --------------------------------------

            for detection in detections:

                x1, y1, x2, y2 = detection["bbox"]

                try:

                    # Generate face embedding using SFace
                    embedding = recognizer.extract_embedding(
                        frame,
                        detection
                    )

                    # Identify the person
                    name, similarity = recognizer.identify(
                        embedding,
                        known_faces
                    )

                    label = f"{name} | {similarity:.2f}"

                    # Green for recognized, red for unknown
                    color = (
                        (0, 255, 0)
                        if name != "Unknown"
                        else (0, 0, 255)
                    )

                except (ValueError, KeyError, cv2.error) as error:

                    print(f"Recognition error: {error}")

                    label = "Recognition Error"
                    color = (0, 165, 255)

                # ----------------------------------
                # 7. DRAW BOUNDING BOX
                # ----------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # ----------------------------------
                # 8. DISPLAY IDENTITY LABEL
                # ----------------------------------

                label_y = y1 - 10 if y1 > 25 else y1 + 20

                cv2.putText(
                    frame,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )

            # --------------------------------------
            # 9. DISPLAY FRAME
            # --------------------------------------

            cv2.imshow(
                "Real-Time Face Recognition",
                frame
            )

            # --------------------------------------
            # 10. EXIT CONDITION
            # --------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\nApplication interrupted")

    finally:

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        print("Face recognition stopped")


# --------------------------------------------------
# 11. APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
