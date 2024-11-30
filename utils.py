
import cv2

def draw_penalized_text(frame):
    font = cv2.FONT_HERSHEY_TRIPLEX
    y_pos = 180
    cv2.putText(frame, "Fined license plates:", (25, y_pos), font, 1, (255, 255, 255), 2)
    y_pos += 80
    for text in penalized_texts:
        cv2.putText(frame, "-> " + text, (40, y_pos), font, 1, (255, 255, 255), 2)
        y_pos += 60
