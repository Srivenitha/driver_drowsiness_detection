import tensorflow as tf
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt


model = tf.keras.models.load_model("drowsiness_cnn.h5")
class_names = ["Closed", "No_Yawn", "Open", "Yawn"]



def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    return class_names[class_id], prediction[0]


test_root = "dataset_new/test"

for folder in class_names:
    folder_path = os.path.join(test_root, folder)
    if not os.path.exists(folder_path):
        print(f" Folder not found: {folder_path}")
        continue

    
    img_name = random.choice(os.listdir(folder_path))
    img_path = os.path.join(folder_path, img_name)

    
    pred_class, confidence = predict_image(img_path)

    
    print(f"\nActual: {folder}")
    print(f"Predicted: {pred_class}")
    print(f"Confidence: {np.max(confidence)*100:.2f}%")
    print(f"File: {img_name}")

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f"Actual: {folder} | Predicted: {pred_class}")
    plt.axis("off")
    plt.show()




#test with your own images
# new_image_path = "test1.webp"  

# if os.path.exists(new_image_path):
#     pred_class, confidence = predict_image(new_image_path)
#     print("\n New Image Test:")
#     print(f"Predicted Class: {pred_class}")
#     print(f"Confidence: {np.max(confidence)*100:.2f}%")

#     img = cv2.imread(new_image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(img)
#     plt.title(f"Predicted: {pred_class}")
#     plt.axis("off")
#     plt.show()
# else:
#     print(" Image not found:", new_image_path)




























