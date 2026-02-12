import streamlit as st
import cv2

st.title("Camera Test")

if st.button("Capture Frame"):
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Camera not working")
    else:
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption="Captured Image")
