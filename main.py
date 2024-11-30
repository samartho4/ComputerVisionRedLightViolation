
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
from traffic_light import detect_traffic_light_color
from stop_line_detection import LineDetector
from license_plate import extract_license_plate, apply_ocr_to_image
from database import (
    create_database_and_table,
    update_database_with_violation,
    print_all_violations,
    clear_license_plates,
)
import cv2
import re
import matplotlib.pyplot as plt

penalized_texts = []

def main():
    create_database_and_table(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
    clear_license_plates(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
    vid = cv2.VideoCapture("traffic_video.mp4")
    detector = LineDetector()

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        rect = (1700, 40, 100, 250)
        frame, color = detect_traffic_light_color(frame, rect)
        frame, mask_line = detector.detect_white_line(frame, color)
        if color == "red":
            frame, license_plate_images = extract_license_plate(frame, mask_line)
            for license_plate_image in license_plate_images:
                text = apply_ocr_to_image(license_plate_image)
                if text and re.match("^[A-Z]{2}\s[0-9]{3,4}$", text) and text not in penalized_texts:
                    penalized_texts.append(text)
                    plt.imshow(license_plate_image, cmap="gray")
                    plt.axis("off")
                    plt.show()
                    update_database_with_violation(text, DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
        if penalized_texts:
            draw_penalized_text(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:
            break
    vid.release()
    cv2.destroyAllWindows()
    print_all_violations(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)

if __name__ == "__main__":
    main()
