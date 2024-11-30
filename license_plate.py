
import cv2
from PIL import Image
import pytesseract

def extract_license_plate(frame, mask_line):
    return frame, []

def apply_ocr_to_image(license_plate_image):
    _, img = cv2.threshold(license_plate_image, 120, 255, cv2.THRESH_BINARY)
    pil_img = Image.fromarray(img)
    return pytesseract.image_to_string(pil_img, config="--psm 6").strip()
