
import cv2
import numpy as np

def detect_traffic_light_color(image, rect):
    x, y, w, h = rect
    roi = image[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_lower, red_upper = np.array([0, 120, 70]), np.array([10, 255, 255])
    yellow_lower, yellow_upper = np.array([20, 100, 100]), np.array([30, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    if cv2.countNonZero(red_mask) > 0:
        return image, "red"
    elif cv2.countNonZero(yellow_mask) > 0:
        return image, "yellow"
    return image, "green"
