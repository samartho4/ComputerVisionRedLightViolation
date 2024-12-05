import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# Load the trained CNN model and label encoder
cnn_model = load_model('cnn_model.h5')
label_encoder_classes = np.load('label_encoder.npy', allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = label_encoder_classes

# Load YOLO for object detection
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Function to display images with labeled traffic lights and predict colors for each traffic light using CNN model and YOLO for detection
def display_and_predict_labeled_images(image_paths):
    output_folder = 'labeled_images'
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Could not open {img_path}")
            continue
        
        height, width, channels = image.shape
        
        # Detecting objects with YOLO
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 9:  # Class ID 9 corresponds to traffic light in COCO dataset
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                cropped_image = image[y:y+h, x:x+w]
                
                # Check if the cropped image is empty before resizing
                if cropped_image.size == 0:
                    print(f"Error: Cropped image is empty for {img_path}")
                    continue
                
                resized_image = cv2.resize(cropped_image, (32, 32))
                resized_image_normalized = resized_image.astype('float32') / 255.0
                
                # Predict the color using the trained CNN model
                predicted_color_probabilities = cnn_model.predict(resized_image_normalized.reshape(1, 32, 32, 3))
                predicted_color_label_index = np.argmax(predicted_color_probabilities)
                predicted_color_label = label_encoder.inverse_transform([predicted_color_label_index])[0]
                
                # Draw a rectangle around the detected light and label it with the predicted color
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f"{predicted_color_label} ({predicted_color_probabilities[0][predicted_color_label_index]*100:.2f}%)", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save the labeled image to the output folder
        output_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, image)
        print(f"Saved labeled image to {output_path}")

# Example usage of display_and_predict_labeled_images function with some test images
test_image_paths = [os.path.join("test_dataset", i) for i in os.listdir("test_dataset") if os.path.isfile(os.path.join("test_dataset", i))]
#test_image_paths = ["test_dataset/00009.jpg","test_dataset/00035.jpg","test_dataset/02975.jpg","test_dataset/02842.jpg"]
display_and_predict_labeled_images(test_image_paths)