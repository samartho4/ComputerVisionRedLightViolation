
import cv2
import numpy as np
from collections import deque

class LineDetector:
    def __init__(self, num_frames_avg=10):
        self.y_start_queue = deque(maxlen=num_frames_avg)
        self.y_end_queue = deque(maxlen=num_frames_avg)

    def detect_white_line(self, frame, color):
        pass
