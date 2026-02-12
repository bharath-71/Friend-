import streamlit as st
import cv2
import time

st.title("Emotion Detection Camera")

run = st.checkbox("Start Camera")

frame_placeholder = st.empty()
emotion_placeholder = st.empty()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not working")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion = "Neutral 😐"

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.7, minNeighbors=20
        )

        if len(smiles) > 0:
            emotion = "Happy 🙂"
        else:
            emotion = "Neutral 😐"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame)
    emotion_placeholder.subheader(f"Detected Emotion: {emotion}")

    time.sleep(0.03)

cap.release()
