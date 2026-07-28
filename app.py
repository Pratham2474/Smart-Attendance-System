"""
Live Smart Attendance System using Face Recognition (Streamlit)
-----------------------------------------------------------------
Features
  1. Register New User  -> capture face samples live from webcam (browser)
  2. Train Model         -> train an OpenCV LBPH face recognizer
  3. Live Attendance     -> live webcam recognition, auto-marks attendance
                            with Date + Time, shows the recognized profile
  4. Dashboard           -> registered profiles + attendance analytics
  5. Attendance Records  -> full log, filterable, downloadable as CSV

Run with:   streamlit run app.py
"""

import os
import pickle
import threading
from datetime import datetime

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

# --------------------------------------------------------------------------
# Paths & one-time setup
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
LABELS_FILE = os.path.join(TRAINER_DIR, "labels.pickle")
PROFILES_CSV = os.path.join(BASE_DIR, "profiles.csv")
ATTENDANCE_CSV = os.path.join(BASE_DIR, "attendance.csv")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)

if not os.path.exists(PROFILES_CSV):
    pd.DataFrame(columns=["Id", "Name", "Department", "Email", "RegisteredOn"]).to_csv(
        PROFILES_CSV, index=False
    )

if not os.path.exists(ATTENDANCE_CSV):
    pd.DataFrame(columns=["Id", "Name", "Date", "Time"]).to_csv(ATTENDANCE_CSV, index=False)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Lower = stricter match required (LBPH returns a distance, not a similarity score)
CONFIDENCE_THRESHOLD = 70
MAX_SAMPLES = 40

st.set_page_config(page_title="Smart Attendance System", page_icon="🧑‍💻", layout="wide")


# --------------------------------------------------------------------------
# Thread-safe shared state (webrtc frame callbacks run on a worker thread)
# --------------------------------------------------------------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_recognized = None
        self.marked_today = set()
        self._load_today_marked()

    def _load_today_marked(self):
        today = datetime.now().strftime("%Y-%m-%d")
        df = pd.read_csv(ATTENDANCE_CSV, dtype=str)
        if not df.empty:
            self.marked_today = set(df[df["Date"] == today]["Id"].tolist())

    def mark_attendance(self, id_, name):
        with self.lock:
            today = datetime.now().strftime("%Y-%m-%d")
            now_time = datetime.now().strftime("%H:%M:%S")
            if str(id_) not in self.marked_today:
                new_row = pd.DataFrame(
                    [[id_, name, today, now_time]], columns=["Id", "Name", "Date", "Time"]
                )
                new_row.to_csv(ATTENDANCE_CSV, mode="a", header=False, index=False)
                self.marked_today.add(str(id_))
                return True
        return False

    def set_last_recognized(self, info):
        with self.lock:
            self.last_recognized = info

    def get_last_recognized(self):
        with self.lock:
            return self.last_recognized


@st.cache_resource
def get_shared_state():
    return SharedState()


shared_state = get_shared_state()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_profiles():
    return pd.read_csv(PROFILES_CSV, dtype=str).fillna("")


def load_attendance():
    return pd.read_csv(ATTENDANCE_CSV, dtype=str).fillna("")


def save_profile(id_, name, dept, email):
    df = load_profiles()
    if id_ in df["Id"].values:
        return False
    new_row = pd.DataFrame(
        [[id_, name, dept, email, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]],
        columns=["Id", "Name", "Department", "Email", "RegisteredOn"],
    )
    new_row.to_csv(PROFILES_CSV, mode="a", header=False, index=False)
    return True


def get_label_map():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def load_recognizer():
    if os.path.exists(TRAINER_FILE):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_FILE)
        return recognizer
    return None


def train_model():
    faces, ids, label_map = [], [], {}
    current_label = 0

    for folder in sorted(os.listdir(DATASET_DIR)):
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(folder_path) or "_" not in folder:
            continue
        student_id = folder.split("_")[0]
        images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))]
        if not images:
            continue
        label_map[current_label] = student_id
        for img_name in images:
            img = cv2.imread(os.path.join(folder_path, img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            ids.append(current_label)
        current_label += 1

    if not faces:
        return False, "No training data found. Please register at least one user first."

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_FILE)
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)

    return True, f"Model trained on {len(faces)} face samples across {current_label} people."


def sample_photo_for(id_, name):
    """Return path to a sample face image for a given profile, if it exists."""
    folder = os.path.join(DATASET_DIR, f"{id_}_{name.replace(' ', '_')}")
    if os.path.isdir(folder):
        imgs = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png")))
        if imgs:
            return os.path.join(folder, imgs[0])
    return None


# --------------------------------------------------------------------------
# Video processors
# --------------------------------------------------------------------------
class RegisterProcessor(VideoProcessorBase):
    def __init__(self):
        self.count = 0
        self.max_samples = MAX_SAMPLES
        self.save_dir = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if self.save_dir and self.count < self.max_samples:
                face_img = cv2.resize(gray[y : y + h, x : x + w], (200, 200))
                cv2.imwrite(os.path.join(self.save_dir, f"{self.count}.jpg"), face_img)
                self.count += 1

        cv2.putText(
            img,
            f"Samples captured: {self.count}/{self.max_samples}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")


class AttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.recognizer = load_recognizer()
        self.labels = get_label_map()
        self.profiles = load_profiles().set_index("Id").to_dict(orient="index")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y : y + h, x : x + w], (200, 200))
            name_text = "Unknown"
            box_color = (0, 0, 255)

            if self.recognizer is not None:
                label_pred, conf = self.recognizer.predict(face_img)
                if conf < CONFIDENCE_THRESHOLD and label_pred in self.labels:
                    student_id = self.labels[label_pred]
                    profile = self.profiles.get(student_id)
                    if profile:
                        name_text = profile["Name"]
                        box_color = (0, 200, 0)
                        shared_state.mark_attendance(student_id, profile["Name"])
                        shared_state.set_last_recognized(
                            {
                                "Id": student_id,
                                "Name": profile["Name"],
                                "Department": profile.get("Department", ""),
                                "Email": profile.get("Email", ""),
                                "Date": datetime.now().strftime("%Y-%m-%d"),
                                "Time": datetime.now().strftime("%H:%M:%S"),
                                "Confidence": round(100 - conf, 1),
                            }
                        )

            cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(img, name_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        now = datetime.now()
        cv2.putText(
            img,
            now.strftime("%Y-%m-%d  %H:%M:%S"),
            (10, img.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.sidebar.title("🧑‍💻 Smart Attendance")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🧑‍💻 Register New User", "🎯 Train Model", "📸 Live Attendance", "📊 Dashboard", "📁 Attendance Records"],
)

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# ---------------------------- HOME ----------------------------------------
if page == "🏠 Home":
    st.title("Live Smart Attendance System")
    st.caption("Face Recognition • Live Camera • Date & Time • Profile Dashboard")

    profiles = load_profiles()
    attendance = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")
    present_today = attendance[attendance["Date"] == today]["Id"].nunique() if not attendance.empty else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Registered Users", len(profiles))
    c2.metric("Present Today", present_today)
    c3.metric("Total Attendance Records", len(attendance))

    st.markdown(
        """
        ### How it works
        1. **Register New User** — enter details and capture ~40 live face samples from your webcam.
        2. **Train Model** — trains an LBPH face recognizer on all registered faces.
        3. **Live Attendance** — starts the webcam, recognizes faces in real time, and
           automatically marks attendance (once per person per day) with date & time.
        4. **Dashboard** — view all profiles, attendance stats and trends.
        5. **Attendance Records** — browse, filter, and download the full log as CSV.
        """
    )

# ---------------------------- REGISTER -------------------------------------
elif page == "🧑‍💻 Register New User":
    st.title("Register New User")

    col1, col2 = st.columns([1, 1])
    with col1:
        id_ = st.text_input("ID / Roll Number")
        name = st.text_input("Full Name")
        dept = st.text_input("Department / Class")
        email = st.text_input("Email")
        st.info(f"Look directly at the camera. {MAX_SAMPLES} samples will be captured automatically.")

    with col2:
        ctx = webrtc_streamer(
            key="register",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=RegisterProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

        if ctx.video_processor:
            if id_.strip() and name.strip():
                folder = os.path.join(DATASET_DIR, f"{id_.strip()}_{name.strip().replace(' ', '_')}")
                os.makedirs(folder, exist_ok=True)
                ctx.video_processor.save_dir = folder
            if ctx.state.playing:
                st_autorefresh(interval=1000, key="register_refresh")
            st.progress(min(ctx.video_processor.count / MAX_SAMPLES, 1.0))
            st.write(f"Samples captured: **{ctx.video_processor.count} / {MAX_SAMPLES}**")

    if st.button("💾 Save Profile", type="primary"):
        if id_.strip() and name.strip():
            ok = save_profile(id_.strip(), name.strip(), dept.strip(), email.strip())
            if ok:
                st.success(f"Profile saved for **{name}** (ID {id_}). Now go to 'Train Model'.")
            else:
                st.warning("A profile with this ID already exists.")
        else:
            st.error("ID and Name are required.")

# ---------------------------- TRAIN -----------------------------------------
elif page == "🎯 Train Model":
    st.title("Train Recognition Model")
    st.write("Trains the face recognizer on every registered user's captured samples.")

    n_people = len([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    st.write(f"People with captured samples: **{n_people}**")

    if st.button("🚀 Train Model Now", type="primary"):
        with st.spinner("Training..."):
            ok, msg = train_model()
        if ok:
            st.success(msg)
        else:
            st.error(msg)

# ---------------------------- LIVE ATTENDANCE -------------------------------
elif page == "📸 Live Attendance":
    st.title("Live Attendance")

    if not os.path.exists(TRAINER_FILE):
        st.warning("No trained model found yet. Register users and train the model first.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            ctx = webrtc_streamer(
                key="attendance",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=AttendanceProcessor,
                media_stream_constraints={"video": True, "audio": False},
            )
            if ctx.state.playing:
                st_autorefresh(interval=2000, key="attendance_refresh")

        with col2:
            st.subheader("Recognized Profile")
            info = shared_state.get_last_recognized()
            if info:
                photo = sample_photo_for(info["Id"], info["Name"])
                if photo:
                    st.image(photo, width=150, caption=info["Name"])
                st.write(f"**Name:** {info['Name']}")
                st.write(f"**ID:** {info['Id']}")
                st.write(f"**Department:** {info['Department']}")
                st.write(f"**Email:** {info['Email']}")
                st.write(f"**Date:** {info['Date']}")
                st.write(f"**Time:** {info['Time']}")
                st.write(f"**Match confidence:** {info['Confidence']}%")
            else:
                st.info("No one recognized yet — step in front of the camera.")

        st.subheader("Today's Attendance")
        df = load_attendance()
        today = datetime.now().strftime("%Y-%m-%d")
        st.dataframe(df[df["Date"] == today], use_container_width=True)

# ---------------------------- DASHBOARD -------------------------------------
elif page == "📊 Dashboard":
    st.title("Profiles & Analytics Dashboard")

    profiles = load_profiles()
    attendance = load_attendance()

    st.subheader("Registered Profiles")
    search = st.text_input("Search by name or ID")
    view = profiles.copy()
    if search:
        mask = view["Name"].str.contains(search, case=False) | view["Id"].str.contains(search, case=False)
        view = view[mask]

    if not attendance.empty:
        counts = attendance.groupby("Id").size().rename("Days Present")
        view = view.merge(counts, left_on="Id", right_index=True, how="left")
        view["Days Present"] = view["Days Present"].fillna(0).astype(int)
    else:
        view["Days Present"] = 0

    st.dataframe(view, use_container_width=True)

    st.subheader("Attendance Trend")
    if not attendance.empty:
        daily = attendance.groupby("Date").size().rename("Count")
        st.bar_chart(daily)
    else:
        st.info("No attendance recorded yet.")

# ---------------------------- RECORDS ---------------------------------------
elif page == "📁 Attendance Records":
    st.title("Attendance Records")

    attendance = load_attendance()
    if attendance.empty:
        st.info("No attendance recorded yet.")
    else:
        dates = pd.to_datetime(attendance["Date"])
        min_d, max_d = dates.min().date(), dates.max().date()
        date_range = st.date_input("Filter by date range", value=(min_d, max_d))

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            mask = (dates.dt.date >= start) & (dates.dt.date <= end)
            filtered = attendance[mask]
        else:
            filtered = attendance

        st.dataframe(filtered, use_container_width=True)
        st.download_button(
            "⬇️ Download as CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="attendance_export.csv",
            mime="text/csv",
        )
