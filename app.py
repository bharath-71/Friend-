import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import time

# ---------------- PAGE ----------------
st.set_page_config(page_title="Emotion Detection")
st.title("🎭 Live Emotion Detection")

# ---------------- LOAD ----------------
@st.cache_resource
def load_models():
    model = load_model("emotion_model.h5")
    face = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return model, face

model, face_cascade = load_models()
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ---------------- SESSION ----------------
if "run" not in st.session_state:
    st.session_state.run = False
if "cap" not in st.session_state:
    st.session_state.cap = None
if "emotion_buf" not in st.session_state:
    st.session_state.emotion_buf = deque(maxlen=7)

# ---------------- UI ----------------
run = st.checkbox("Start Camera")
frame_window = st.image([])

# ---------------- CAMERA LOOP ----------------
if run:
    st.session_state.run = True

    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap = st.session_state.cap

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]

            try:
                roi = cv2.resize(roi, (48, 48)) / 255.0
                roi = roi.reshape(1, 48, 48, 1)

                preds = model.predict(roi, verbose=0)
                idx = np.argmax(preds)

                st.session_state.emotion_buf.append(idx)
                idx = max(
                    set(st.session_state.emotion_buf),
                    key=st.session_state.emotion_buf.count
                )
                emotion = labels[idx]

            except:
                continue

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(
                frame, emotion, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2
            )

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.03)

else:
    st.session_state.run = False

    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

    st.session_state.emotion_buf.clear()
    frame_window.empty()
    st.info("📷 Camera stopped")
