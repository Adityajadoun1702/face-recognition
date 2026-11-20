import cv2

from videoCapture import VideoCapture
from detector.faceDetector import FaceDetector

detector=FaceDetector()
camera=VideoCapture()
try:
    for ret,frame in camera.get_frames():
        if not ret:
            print("camera error")
            break
        detections=detector.detect(frame)
        for detection in detections:
            x1,y1,x2,y2=detection["bbox"]
            confidence=detection["confidence"]
            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0, 255, 0),
                2
            )
            cv2.putText(
                    frame,
                    f"Face: {confidence:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
        

        cv2.imshow("Face Detection", frame)



        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()