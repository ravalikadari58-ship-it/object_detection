import streamlit as st
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from datetime import datetime
import os
import time

# ---------------- UI ----------------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1581091215367-59ab6b59f3b6");
    background-size: cover;
}
h1 {text-align:center; color:red;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚨 Smart CCTV Surveillance 📹</h1>", unsafe_allow_html=True)

# ---------------- MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- SIDEBAR ----------------
option = st.sidebar.radio("🔍 Detection Type", ["All", "Persons", "Objects"])

# ---------------- SESSION STATE ----------------
if "run" not in st.session_state:
    st.session_state.run = False

if "data" not in st.session_state:
    st.session_state.data = []

if "person_total" not in st.session_state:
    st.session_state.person_total = 0

if "object_total" not in st.session_state:
    st.session_state.object_total = 0

# ---------------- BUTTONS ----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🎥 Start"):
        st.session_state.run = True

with col2:
    if st.button("🛑 Stop"):
        st.session_state.run = False

# ---------------- CAMERA ----------------
FRAME_WINDOW = st.image([])

if st.session_state.run:
    cap = cv2.VideoCapture(0)

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        results = model(frame)

        person_count = 0
        object_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label == "person":
                    person_count += 1
                    color = (0,0,255)
                else:
                    object_count += 1
                    color = (0,255,0)

                # FILTER
                if option == "Persons" and label != "person":
                    continue
                if option == "Objects" and label == "person":
                    continue

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # -------- SAVE ONLY WHEN DETECTED --------
        if person_count > 0 or object_count > 0:
            timestamp = datetime.now().strftime("%H:%M:%S")

            st.session_state.person_total += person_count
            st.session_state.object_total += object_count

            st.session_state.data.append({
                "Time": timestamp,
                "Persons": person_count,
                "Objects": object_count
            })

        FRAME_WINDOW.image(frame, channels="BGR")

        time.sleep(0.1)

    cap.release()

# ---------------- SHOW DATA ----------------
if len(st.session_state.data) > 0:

    df = pd.DataFrame(st.session_state.data)

    st.subheader("📋 Detection Log")

    st.dataframe(df)

    st.subheader("📊 Total Count")

    st.write(f"👤 Total Persons: {st.session_state.person_total}")
    st.write(f"📦 Total Objects: {st.session_state.object_total}")

    # -------- BAR GRAPH --------
    st.subheader("📊 Bar Graph")

    fig1, ax1 = plt.subplots()
    ax1.bar(["Persons", "Objects"],
            [st.session_state.person_total,
             st.session_state.object_total])
    st.pyplot(fig1)

    # -------- PIE CHART --------
    st.subheader("🥧 Pie Chart")

    fig2, ax2 = plt.subplots()
    ax2.pie([st.session_state.person_total,
             st.session_state.object_total],
            labels=["Persons", "Objects"],
            autopct='%1.1f%%')
    st.pyplot(fig2)

    # -------- SAVE --------
    if st.button("💾 Save Report"):
        df.to_csv("report.csv", index=False)
        st.success("Saved Successfully ✅")