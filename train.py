import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras import Sequential
from keras import layers
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
plt.title('Confusion Matrix')
plt.show()

# Save the trained model and label encoder
cnn_model.save('cnn_model.h5')
with open('label_encoder.npy', 'wb') as f:
    np.save(f, label_encoder.classes_)