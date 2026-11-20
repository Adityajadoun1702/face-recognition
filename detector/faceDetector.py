import cv2
from pathlib import Path
class FaceDetector:
    def __init__(self,confidence_threshold=0.8,nms_threshold=0.3):
        model_path=(Path(__file__).resolve().parents[1]
        /"models"
        /"model_weights"
        /"face_detection_yunet_2023mar.onnx"
        )
        if not model_path:
            raise FileNotFoundError(f"YuNet model not found: {model_path}")

        self.detector=cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(320,320),
            score_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            top_k=5000
        )
        self.input_size=None
    def detect(self,frame):
        if frame is None or frame.size==0:
            raise ValueError("recieved an empty frame")
        height,width=frame.shape[:2]
        if self.input_size!=(width,height):
            self.detector.setInputSize((width,height))
            self.input_size=(width,height)
        _,faces=self.detector.detect(frame)
        if faces is None or faces.size==0:
            return []
        detections=[]
        for face in faces:
            x,y,w,h=face[:4]
            confidence=float(face[14])
            x1 = max(0, int(x))
            y1 = max(0, int(y))

            x2 = min(width, int(x + w))
            y2 = min(height, int(y + h))
            face_data = face.astype("float32").reshape(1, 15).copy()

            detections.append({
                "bbox":(x1,y1,x2,y2),
                "confidence":confidence,
                "yunet_face":face_data
            })
        return detections

            
        