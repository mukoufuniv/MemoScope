import cv2

class CameraStream:
    """Simple wrapper around cv2.VideoCapture."""
    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None):
        self.cap = cv2.VideoCapture(index)
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        success, frame = self.cap.read()
        return frame if success else None

    def release(self):
        self.cap.release()

