import streamlit as st
import cv2

st.title("Live Camera with Face Detection")

start = st.button("Start Camera")
stop = st.button("Stop Camera")

frame_slot = st.image([])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

if start:
    st.session_state.camera_on = True

if stop:
    st.session_state.camera_on = False

if st.session_state.camera_on:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_slot.image(frame)
