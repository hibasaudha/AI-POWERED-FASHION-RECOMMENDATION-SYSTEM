import numpy as np
import cv2
import os
import random
import cvlib as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def suggest_colors(skin_tone, gender):
    """Returns a list of recommended colors based on skin tone and gender."""
    color_recommendations = {
        "light": {
            "man": ["Powder Blue", "Cool Gray", "Navy Blue", "Silver", "Sky Blue"],
            "woman": ["Lavender", "Rose Pink", "Soft Lilac", "Mint Green", "Pearl White"]
        },
        "mid-light": {
            "man": ["Olive Green", "Warm Mustard", "Beige", "Terracotta", "Sandy Brown"],
            "woman": ["Peach", "Blush Pink", "Dusty Rose", "Soft Taupe", "Coral"]
        },
        "mid-dark": {
            "man": ["Emerald Green", "Deep Burgundy", "Mustard Yellow", "Copper Brown"],
            "woman": ["Royal Blue", "Teal", "Rust Red", "Plum", "Warm Caramel"]
        },
        "dark": {
            "man": ["Bright Red", "Electric Blue", "Rich Forest Green", "Deep Mahogany"],
            "woman": ["Gold", "Pure White", "Cobalt Blue", "Fuchsia", "Vibrant Yellow"]
        }
    }
    
    return color_recommendations.get(skin_tone.lower(), {}).get(gender.lower(), [])

def load_dress_image(color, dress_dataset_path="images_compressed"):
    """Loads and displays a dress image matching the given color."""
    color_dir = os.path.join(dress_dataset_path, color.replace(" ", "_").lower())
    if os.path.isdir(color_dir):
        images = [img for img in os.listdir(color_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
        if images:
            img_path = os.path.join(color_dir, random.choice(images))
            dress_img = cv2.imread(img_path)
            if dress_img is not None:
                cv2.imshow("Recommended Dress", dress_img)
                cv2.waitKey(3000)  # Display for 3 seconds
                cv2.destroyAllWindows()
                return img_path
    return "No suitable dress image found"

def analyze_skin_tone_and_gender(skin_tone_model, gender_model, class_names, dress_dataset_path="images_compressed"):
    """Captures video, detects face, classifies skin tone & gender, and suggests dresses."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        return

    gender_classes = ['man', 'woman']

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, _ = cv.detect_face(frame)

        for (x1, y1, x2, y2) in faces:
            face_region = frame[y1:y2, x1:x2]

            if face_region.shape[0] < 10 or face_region.shape[1] < 10:
                continue

            # Skin tone classification
            resized_face = cv2.resize(face_region, (64, 64))
            normalized_face = resized_face / 255.0
            input_data = np.expand_dims(normalized_face, axis=0)
            predictions = skin_tone_model.predict(input_data)
            predicted_skin_tone = class_names[np.argmax(predictions)]

            # Gender classification
            face_crop = cv2.resize(face_region, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            gender_predictions = gender_model.predict(face_crop)[0]
            predicted_gender = gender_classes[np.argmax(gender_predictions)]

            # Get recommended dress colors
            recommended_colors = suggest_colors(predicted_skin_tone, predicted_gender)
            dress_image = "No dress found"
            if recommended_colors:
                dress_image = load_dress_image(random.choice(recommended_colors), dress_dataset_path)

            # Draw face rectangle and put text labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Skin Tone: {predicted_skin_tone}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Gender: {predicted_gender}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Dress: {dress_image}", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Skin Tone & Gender Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        skin_tone_model = load_model("skin_tone_model.h5")
        gender_model = load_model("gender_detection_model.h5")
        class_names = ["dark", "light", "mid-dark", "mid-light"]
        analyze_skin_tone_and_gender(skin_tone_model, gender_model, class_names)
    except Exception as e:
        print(f"Error: {e}")
