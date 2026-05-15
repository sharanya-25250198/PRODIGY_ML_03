import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dataset path
dataset_path = r"C:\Users\shara\Downloads\kagglecatsanddogs_5340\PetImages"

categories = ["Cat", "Dog"]

data = []
labels = []

# Load images
for category in categories:
    path = os.path.join(dataset_path, category)
    label = categories.index(category)

    count = 0

    for img in os.listdir(path):

        try:
            img_path = os.path.join(path, img)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (64, 64))

            data.append(image.flatten())
            labels.append(label)

            count += 1

            # Limit images for faster execution
            if count == 200:
                break

        except:
            pass

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM model
model = SVC(kernel='linear')

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Display sample prediction
sample_image = X_test[0].reshape(64, 64)

prediction = model.predict([X_test[0]])

label_name = categories[prediction[0]]

plt.imshow(sample_image, cmap='gray')
plt.title(f"Prediction: {label_name}")
plt.axis('off')

plt.show()