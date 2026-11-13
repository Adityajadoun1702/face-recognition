import cv2

class VideoCapture:
    def __init__(self,camera_index=0):
        self.cap=cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"camera {camera_index} initilized")
    def get_frames(self):
        while self.cap.isOpened():
            ret,frames=self.cap.read()
            if not ret:
                yield False,None
                break
            yield True,frames
    def release(self):
        cv2.destroyAllWindows()
        self.cap.release()
    
