import os
import openai
import requests
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from io import BytesIO
import speech_recognition as sr

# Set OpenAI API Key
openai.OpenAI(api_key="Osk-proj-q60gSHoHxHTw-rXTnmJGDCPBcQY2oMwEK4yp-Dc0TWs2-T4AOdunkvBjhXy0SxFSdOVMSPAQewT3BlbkFJnABQvKCTTekkn6qRJ5rv5zfMCtR__Gxb2nW90AfJWxRA0dvEPEjPjICaPkXiRJ-si6-PNjNPAA")  # Ensure API key is set properly

if not openai.api_key:
    raise ValueError("OpenAI API key is missing. Set it as an environment variable or provide it directly.")

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to recognize speech and convert to text
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak the object name:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized Object: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
        return None
    except sr.RequestError:
        print("Speech service error.")
        return None

# Function to generate an image using OpenAI’s DALL·E
def generate_image(object_name):
    client = openai.OpenAI()  # ✅ Create OpenAI client

    response = client.images.generate(
        model="dall-e-3",  # ✅ Use "dall-e-2" if needed
        prompt=f"A realistic image of {object_name}",
        size="512x512",
        n=1
    )

    return response.data[0].url


# Function to download and process the generated image
def get_object_image(object_url):
    response = requests.get(object_url)
    object_img = Image.open(BytesIO(response.content)).convert("RGBA")
    object_img = object_img.resize((200, 200))  # Resize object
    return object_img

# Function to overlay the object on the user's hand
def overlay_object_on_hand(frame, object_img, hand_landmarks):
    if hand_landmarks is None:
        return frame  # Skip processing if no hand is detected

    h, w, _ = frame.shape
    x, y = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * w), \
           int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * h)

    # Convert object to OpenCV format
    object_img = np.array(object_img)
    object_img = cv2.cvtColor(object_img, cv2.COLOR_RGBA2BGRA)

    # Define overlay size and position
    obj_h, obj_w, _ = object_img.shape
    x_offset, y_offset = x - obj_w // 2, y - obj_h // 2

    # Ensure within frame bounds
    x_offset = max(0, min(x_offset, w - obj_w))
    y_offset = max(0, min(y_offset, h - obj_h))

    # Overlay object on frame
    frame[y_offset:y_offset+obj_h, x_offset:x_offset+obj_w] = object_img[:, :, :3]

    return frame

# Main function to run real-time hand tracking and object overlay
def main():
    object_name = recognize_speech()
    if not object_name:
        return

    object_url = generate_image(object_name)
    object_img = get_object_image(object_url)

    cap = cv2.VideoCapture(0)  # Open webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw hand landmarks and overlay object
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                frame = overlay_object_on_hand(frame, object_img, hand_landmarks)

        # Show the result
        cv2.imshow("AI Hand Object Placement", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
