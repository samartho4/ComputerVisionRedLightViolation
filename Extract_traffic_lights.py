import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras import Sequential
from keras import layers
from keras.datasets import mnist
from keras import utils
import os

# Load the JSON file
with open('train.json', 'r') as file:
    json_data = json.load(file)

# Prepare lists to hold image data and labels
images = []
labels = []

# Iterate over each annotation
for ann in json_data['annotations']: 
    filename = ann['filename']
    bndbox = ann['bndbox']
    
    if ann['inbox']:
        inbox = ann['inbox']
        color = inbox[0]['color']

        # Load the image
        image = cv2.imread(filename)
        if image is None:
            print(f"Error: Could not open {filename}")
            continue

        # Crop the image using bounding box coordinates
        xmin, ymin, xmax, ymax = int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])
        cropped_image = image[ymin:ymax, xmin:xmax]

        # Check if the cropped image is empty
        if cropped_image.size == 0:
            print(f"Error: Cropped image is empty for {filename}")
            continue

        # Resize the cropped image to a fixed size (e.g., 32x32)
        resized_image = cv2.resize(cropped_image, (32, 32))

        # Append the image data and label to the lists
        images.append(resized_image)
        labels.append(color)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize the pixel values to the range [0, 1]
images = images.astype('float32') / 255.0

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Convert labels to categorical format for CNN
y_train_cat = utils.to_categorical(y_train)
y_test_cat = utils.to_categorical(y_test)

# Define the CNN model with additional layers and dropout for better accuracy
cnn_model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the CNN model with a lower learning rate for better accuracy
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model with more epochs and validation split for better accuracy
cnn_model.fit(X_train, y_train_cat, epochs=75, validation_split=0.2)

# Evaluate the CNN model on the test set
loss, accuracy = cnn_model.evaluate(X_test, y_test_cat)
print(f"CNN Test Accuracy: {accuracy * 100:.2f}%")

# Predict on test set
y_pred_cat = cnn_model.predict(X_test)
y_pred = np.argmax(y_pred_cat, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

# Plot confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.show()

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

display_and_predict_labeled_images(test_image_paths)