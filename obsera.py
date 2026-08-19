
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

import sys
import time
import math
import threading
from pathlib import Path
from collections import deque

import numpy as np
import cv2
import mediapipe as mp

# Optional ML libs
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception:
    tf = None
    load_model = None

try:
    import joblib
except Exception:
    joblib = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="OBSERA — Drowsiness Monitor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONFIG — SAME CONCEPT AS ORIGINAL FILE
# ============================================================

EYE_MODEL_PATH = Path("models/eye_cnn_premade.h5")
YAWN_MODEL_PATH = Path("models/yawn_svm.joblib")

CAM_W, CAM_H = 320, 240
PRED_EVERY = 6
PLOT_LEN = 200
EAR_CALIB_SAMPLES = 90
EAR_DELTA = 0.10
EAR_DENOM = 0.20

# weights and scoring
W_EYE_CNN = 0.6
W_EAR = 0.4
EYE_MAX_POINTS = 50.0
W_YAWN_POINTS = 30.0
W_GAZE_POINTS = 2.0

# sliding window yawn alert
WINDOW_SECONDS = 60.0
NEEDED_HIGH_SECONDS = 10.0
SCORE_THRESHOLD = 60.0
ALARM_COOLDOWN = 15.0

# immediate EAR-based alert
EAR_THRESHOLD = 0.21
EYE_CLOSED_REQUIRED_SECONDS = 3.0

# alarm specifics
ALARM_DURATION_SEC = 12

# mediapipe indices
mp_face = mp.solutions.face_mesh
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_L = 61
MOUTH_R = 291


# ============================================================
# HELPERS — SAME AS ORIGINAL
# ============================================================

def eye_aspect_ratio(landmarks, idxs):
    pts = [(landmarks[i].x, landmarks[i].y) for i in idxs]

    def d(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    A = d(pts[1], pts[5])
    B = d(pts[2], pts[4])
    C = d(pts[0], pts[3])

    return (A + B) / (2.0 * C) if C > 0 else 0.0


def mouth_aspect_ratio(landmarks):
    try:
        top = landmarks[MOUTH_TOP]
        bottom = landmarks[MOUTH_BOTTOM]
        left = landmarks[MOUTH_L]
        right = landmarks[MOUTH_R]

        vert = math.hypot(
            top.x - bottom.x,
            top.y - bottom.y
        )

        horiz = math.hypot(
            left.x - right.x,
            left.y - right.y
        )

        return vert / horiz if horiz > 0 else 0.0

    except Exception:
        return 0.0


def crop_patch(image, landmarks, indices, pad=6):
    h, w = image.shape[:2]

    pts = [
        (
            int(landmarks[i].x * w),
            int(landmarks[i].y * h)
        )
        for i in indices
    ]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    x1 = max(0, min(xs) - pad)
    x2 = min(w - 1, max(xs) + pad)

    y1 = max(0, min(ys) - pad)
    y2 = min(h - 1, max(ys) + pad)

    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    return image[y1:y2, x1:x2]


def prep_for_model(bgr, size=96):
    try:
        im = cv2.resize(
            bgr,
            (size, size)
        ).astype("float32") / 255.0

        return np.expand_dims(im, 0)

    except Exception:
        return None


# ============================================================
# LOAD MODELS DEFENSIVELY — SAME CONCEPT AS ORIGINAL
# ============================================================

eye_model = None

if EYE_MODEL_PATH.exists() and load_model is not None:
    try:
        eye_model = load_model(
            str(EYE_MODEL_PATH),
            compile=False
        )
        print("Loaded eye model.")

    except Exception as e:
        print("Failed to load eye model:", e)
        eye_model = None

else:
    print("Eye model not loaded (optional).")


yawn_model = None

if YAWN_MODEL_PATH.exists() and joblib is not None:
    try:
        yawn_model = joblib.load(
            str(YAWN_MODEL_PATH)
        )
        print("Loaded yawn model.")

    except Exception as e:
        print("Failed to load yawn model:", e)
        yawn_model = None

else:
    print("Yawn model not loaded (optional).")


# ============================================================
# SHARED STATE
# ============================================================

class AppState:

    def __init__(self):
        self.lock = threading.Lock()

        self.ear = None
        self.mar = None
        self.eye_prob = None
        self.yawn_flag = 0
        self.yawn_prob = None
        self.gaze = "Center"

        self.score = 0.0
        self.fps = 0.0

        self.baseline = None
        self.closed_thresh = None
        self.ear_score = 0.0

        self.eye_closed_start = None
        self.yawn_duration = 0.0

        self.alert = False
        self.alert_reason = ""

        self.last_alarm_time = 0.0
        self.alarm_until = 0.0

        self.score_history = deque(
            maxlen=PLOT_LEN
        )

        self.last_alert_event = 0.0
        self.sound_event_played = 0.0


app_state = AppState()


# ============================================================
# STREAMLIT VIDEO PROCESSOR
#
# The PyQt VideoWorker + PredictWorker from the original file
# are combined here only because Streamlit-WebRTC supplies the
# video frames itself. Detection/model/scoring logic remains
# the same.
# ============================================================

class DrowsinessProcessor(VideoProcessorBase):

    def __init__(self):

        self.ear_buf = deque(
            maxlen=EAR_CALIB_SAMPLES
        )

        self.baseline = None
        self.calibrated = False
        self.frame_idx = 0

        self.eye_closed_start = None
        self.yawn_start = None

        # Keep the latest model prediction between prediction frames.
        self.latest_pred = {
            "eye_prob": None,
            "yawn_flag": 0,
            "yawn_prob": None
        }

        self.score_history = []

        self.last_alarm_time = 0.0
        self.alarm_until = 0.0
        self.alert_reason = ""

        self.ptime = time.time()

        self.mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.35
        )

    # --------------------------------------------------------
    # MODEL PREDICTION — SAME LOGIC AS PredictWorker
    # --------------------------------------------------------

    def predict(self, frame, lm):

        info = {
            "eye_prob": None,
            "yawn_flag": 0,
            "yawn_prob": None
        }

        # Eye CNN
        try:
            lp = crop_patch(
                frame,
                lm,
                LEFT_EYE_IDX,
                pad=6
            )

            rp = crop_patch(
                frame,
                lm,
                RIGHT_EYE_IDX,
                pad=6
            )

            if (
                eye_model is not None
                and lp is not None
                and rp is not None
            ):

                a = prep_for_model(
                    lp,
                    size=96
                )

                b = prep_for_model(
                    rp,
                    size=96
                )

                if (
                    a is not None
                    and b is not None
                ):

                    pL = float(
                        eye_model.predict(
                            a,
                            verbose=0
                        )[0][0]
                    )

                    pR = float(
                        eye_model.predict(
                            b,
                            verbose=0
                        )[0][0]
                    )

                    info["eye_prob"] = (
                        pL + pR
                    ) / 2.0

        except Exception:
            info["eye_prob"] = None

        # Yawn SVM — SAME LOGIC AS ORIGINAL
        try:
            mar = mouth_aspect_ratio(lm)

            if (
                yawn_model is not None
            ):

                try:

                    feat = np.array(
                        [[mar]]
                    )

                    if hasattr(
                        yawn_model,
                        "predict_proba"
                    ):

                        p = float(
                            yawn_model
                            .predict_proba(
                                feat
                            )[0][1]
                        )

                        info["yawn_prob"] = p

                        info["yawn_flag"] = (
                            1 if p >= 0.6 else 0
                        )

                    else:

                        pred = int(
                            yawn_model.predict(
                                feat
                            )[0]
                        )

                        info["yawn_flag"] = pred

                        info["yawn_prob"] = (
                            1.0
                            if pred == 1
                            else 0.0
                        )

                except Exception:

                    info["yawn_flag"] = (
                        1 if mar > 0.45 else 0
                    )

                    info["yawn_prob"] = min(
                        1.0,
                        mar * 2.0
                    )

            else:

                info["yawn_flag"] = (
                    1 if mar > 0.45 else 0
                )

                info["yawn_prob"] = min(
                    1.0,
                    mar * 2.0
                )

        except Exception:
            pass

        return info

    # --------------------------------------------------------
    # ALARM TRIGGER
    # --------------------------------------------------------

    def trigger_alarm(self, reason):

        now = time.time()

        if (
            now - self.last_alarm_time
            > ALARM_COOLDOWN
        ):

            self.last_alarm_time = now
            self.alarm_until = (
                now + ALARM_DURATION_SEC
            )
            self.alert_reason = reason

    # --------------------------------------------------------
    # FRAME PROCESSING
    # --------------------------------------------------------

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        image = cv2.flip(
            image,
            1
        )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        res = self.mesh.process(rgb)

        info = {
            "ear": None,
            "mar": None,
            "eye_prob": None,
            "yawn_flag": 0,
            "yawn_prob": None,
            "gaze": "Center",
            "score": 0.0,
            "fps": 0.0
        }

        if res.multi_face_landmarks:

            lm = (
                res.multi_face_landmarks[0]
                .landmark
            )

            # ------------------------------------------------
            # EAR — SAME AS ORIGINAL
            # ------------------------------------------------

            try:

                el = eye_aspect_ratio(
                    lm,
                    LEFT_EYE_IDX
                )

                er = eye_aspect_ratio(
                    lm,
                    RIGHT_EYE_IDX
                )

                ear = float(
                    (el + er) / 2.0
                )

                info["ear"] = ear

                if ear > 0:
                    self.ear_buf.append(ear)

            except Exception:

                ear = None

            # ------------------------------------------------
            # MAR — SAME AS ORIGINAL
            # ------------------------------------------------

            try:

                mar = mouth_aspect_ratio(lm)

                info["mar"] = mar

            except Exception:

                mar = None

            # ------------------------------------------------
            # GAZE — SAME AS ORIGINAL
            # ------------------------------------------------

            try:

                left_x = np.mean([
                    lm[i].x
                    for i in LEFT_EYE_IDX
                ])

                right_x = np.mean([
                    lm[i].x
                    for i in RIGHT_EYE_IDX
                ])

                center_x = (
                    left_x + right_x
                ) / 2.0

                info["gaze"] = (
                    "Left"
                    if center_x < 0.45
                    else (
                        "Right"
                        if center_x > 0.55
                        else "Center"
                    )
                )

            except Exception:
                pass

            # ------------------------------------------------
            # EAR CALIBRATION — SAME AS ORIGINAL
            # ------------------------------------------------

            if (
                not self.calibrated
                and
                len(self.ear_buf)
                >= EAR_CALIB_SAMPLES
            ):

                baseline = float(
                    np.median(
                        np.array(
                            self.ear_buf
                        )
                    )
                )

                self.baseline = max(
                    0.18,
                    baseline
                )

                self.calibrated = True

                print(
                    "[calibrated] baseline EAR:",
                    self.baseline
                )

            if (
                self.calibrated
                and
                self.baseline is not None
            ):

                closed_thresh = max(
                    0.06,
                    self.baseline - EAR_DELTA
                )

            else:

                closed_thresh = 0.22

            # ------------------------------------------------
            # EAR SCORE — SAME AS ORIGINAL
            # ------------------------------------------------

            if info["ear"] is not None:

                raw = (
                    closed_thresh
                    -
                    info["ear"]
                )

                ear_score = max(
                    0.0,
                    min(
                        1.0,
                        raw / EAR_DENOM
                    )
                )

            else:

                ear_score = 0.0

            info["ear_score"] = ear_score
            info["baseline"] = self.baseline
            info["closed_thresh"] = (
                closed_thresh
            )

            # ------------------------------------------------
            # PREDICTION EVERY N FRAMES
            # ------------------------------------------------

            if (
                self.frame_idx
                % PRED_EVERY
            ) == 0:

                self.latest_pred = self.predict(
                    image.copy(),
                    lm
                )

            info.update(self.latest_pred)

            # ------------------------------------------------
            # SCORE
            #
            # Same base score calculation as original.
            # Yawn points are later included in the final
            # combined score exactly as in MainWindow.
            # ------------------------------------------------

            combined_eye = ear_score

            eye_points = (
                combined_eye
                * EYE_MAX_POINTS
            )

            yawn_points = 0.0

            gaze_points = (
                W_GAZE_POINTS
                if info.get("gaze")
                in ("Left", "Right")
                else 0.0
            )

            info["score"] = float(
                min(
                    100.0,
                    eye_points
                    + yawn_points
                    + gaze_points
                )
            )

            # ------------------------------------------------
            # EAR IMMEDIATE ALERT
            # SAME RULE AS ORIGINAL
            # ------------------------------------------------

            if info["ear"] is not None:

                if (
                    info["ear"]
                    < EAR_THRESHOLD
                ):

                    if (
                        self.eye_closed_start
                        is None
                    ):

                        self.eye_closed_start = (
                            time.time()
                        )

                    closed_duration = (
                        time.time()
                        -
                        self.eye_closed_start
                    )

                    if (
                        closed_duration
                        >= EYE_CLOSED_REQUIRED_SECONDS
                    ):

                        self.trigger_alarm(
                            "Eyes remained closed for 3 seconds."
                        )

                else:

                    self.eye_closed_start = None

            else:

                self.eye_closed_start = None

            # ------------------------------------------------
            # YAWN ALERT — MORE THAN 3 SECONDS CONTINUOUSLY
            # Uses the original yawn SVM flag.
            # ------------------------------------------------

            if info.get("yawn_flag", 0) == 1:

                if self.yawn_start is None:
                    self.yawn_start = time.time()

                yawn_duration = (
                    time.time() - self.yawn_start
                )

                if yawn_duration >= 3.0:
                    self.trigger_alarm(
                        "Yawning detected for more than 3 seconds."
                    )

            else:
                self.yawn_start = None

            # ------------------------------------------------
            # FINAL COMBINED SCORE
            # SAME LOGIC AS ORIGINAL on_frame_info
            # ------------------------------------------------

            ear_score = info.get(
                "ear_score",
                0.0
            )

            cnn = info.get(
                "eye_prob"
            )

            if cnn is None:

                combined_eye = ear_score

            else:

                num = (
                    W_EYE_CNN
                    * float(cnn)
                    +
                    W_EAR
                    * ear_score
                )

                denom = (
                    W_EYE_CNN
                    +
                    W_EAR
                )

                combined_eye = (
                    num / denom
                    if denom > 0
                    else ear_score
                )

            eye_pts = (
                combined_eye
                * EYE_MAX_POINTS
            )

            if info.get(
                "yawn_flag"
            ):

                yawn_p = info.get(
                    "yawn_prob",
                    1.0
                )

                yawn_pts = (
                    float(yawn_p)
                    * W_YAWN_POINTS
                )

            else:

                yawn_pts = 0.0

            gaze_pts = (
                W_GAZE_POINTS
                if info.get("gaze")
                in ("Left", "Right")
                else 0.0
            )

            final = float(
                min(
                    100.0,
                    eye_pts
                    + yawn_pts
                    + gaze_pts
                )
            )

            info["score"] = final

            # ------------------------------------------------
            # SLIDING WINDOW — SAME RULE AS ORIGINAL
            # ------------------------------------------------

            now = time.time()

            self.score_history.append(
                (
                    now,
                    final
                )
            )

            cutoff = (
                now
                -
                WINDOW_SECONDS
            )

            self.score_history = [
                (t, s)
                for t, s
                in self.score_history
                if t >= cutoff
            ]

            cumulative = 0.0

            if len(
                self.score_history
            ) >= 2:

                for i in range(
                    1,
                    len(self.score_history)
                ):

                    t_prev, s_prev = (
                        self.score_history[i - 1]
                    )

                    t_now, _ = (
                        self.score_history[i]
                    )

                    if (
                        s_prev
                        > SCORE_THRESHOLD
                    ):

                        cumulative += (
                            t_now
                            -
                            t_prev
                        )

            if (
                cumulative
                >= NEEDED_HIGH_SECONDS
            ):

                self.trigger_alarm(
                    "High drowsiness score sustained."
                )

                self.score_history = []

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        ctime = time.time()

        info["fps"] = round(
            1.0 / (
                ctime - self.ptime
            ),
            1
        ) if ctime != self.ptime else 0.0

        self.ptime = ctime

        self.frame_idx += 1

        # ----------------------------------------------------
        # ALARM STATUS
        # ----------------------------------------------------

        alert = (
            time.time()
            < self.alarm_until
        )

        # ----------------------------------------------------
        # CAMERA OVERLAY
        # ----------------------------------------------------

        if info["ear"] is not None:

            cv2.putText(
                image,
                f"EAR: {info['ear']:.3f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        if info["mar"] is not None:

            cv2.putText(
                image,
                f"MAR: {info['mar']:.3f}",
                (12, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        if info["yawn_prob"] is not None:

            cv2.putText(
                image,
                f"Yawn: {info['yawn_prob']:.2f}",
                (12, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        cv2.putText(
            image,
            f"Score: {info['score']:.1f}",
            (12, 109),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        if alert:

            cv2.rectangle(
                image,
                (0, image.shape[0] - 75),
                (
                    image.shape[1],
                    image.shape[0]
                ),
                (0, 0, 200),
                -1
            )

            cv2.putText(
                image,
                "DROWSINESS DETECTED",
                (35, image.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

        # ----------------------------------------------------
        # SAVE STATE FOR STREAMLIT UI
        # ----------------------------------------------------

        with app_state.lock:

            app_state.ear = info.get("ear")
            app_state.mar = info.get("mar")
            app_state.eye_prob = info.get(
                "eye_prob"
            )
            app_state.yawn_flag = info.get(
                "yawn_flag",
                0
            )
            app_state.yawn_prob = info.get(
                "yawn_prob"
            )
            app_state.gaze = info.get(
                "gaze",
                "Center"
            )

            app_state.score = info.get(
                "score",
                0.0
            )

            app_state.fps = info.get(
                "fps",
                0.0
            )

            app_state.baseline = (
                self.baseline
            )

            app_state.closed_thresh = (
                closed_thresh
                if res.multi_face_landmarks
                else None
            )

            app_state.ear_score = (
                info.get(
                    "ear_score",
                    0.0
                )
            )

            if self.eye_closed_start:
                app_state.eye_closed_start = (
                    self.eye_closed_start
                )
            else:
                app_state.eye_closed_start = None

            if self.yawn_start is not None:
                app_state.yawn_duration = (
                    time.time() - self.yawn_start
                )
            else:
                app_state.yawn_duration = 0.0

            app_state.alert = alert
            app_state.alert_reason = (
                self.alert_reason
            )

            app_state.score_history.append(
                info.get("score", 0.0)
            )

            app_state.last_alert_event = (
                self.last_alarm_time
            )

        return frame.from_ndarray(
            image,
            format="bgr24"
        )

    def __del__(self):

        try:
            self.mesh.close()
        except Exception:
            pass


# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f7f8fa;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
    }

    .title {
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 40px;
        font-weight: 700;
        letter-spacing: -1px;
        color: #17191c;
        margin-bottom: 2px;
    }

    .subtitle {
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 15px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .section {
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 19px;
        font-weight: 600;
        color: #202124;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .metric {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 11px;
        padding: 14px 12px;
        text-align: center;
        min-height: 92px;
    }

    .metric-label {
        color: #6b7280;
        font-size: 12px;
        margin-bottom: 7px;
    }

    .metric-value {
        color: #17191c;
        font-size: 24px;
        font-weight: 650;
    }

    .normal {
        background: #ecfdf3;
        border: 1px solid #bbf7d0;
        color: #166534;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        font-size: 17px;
        font-weight: 600;
    }

    .danger {
        background: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
        padding: 18px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: 700;
    }

    .warning {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        font-size: 17px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="title">OBSERA</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Driver Drowsiness Monitoring System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CAMERA
# ============================================================

st.markdown(
    '<div class="section">Camera</div>',
    unsafe_allow_html=True
)

ctx = webrtc_streamer(
    key="obsera-drowsiness-camera",
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)


# ============================================================
# ============================================================
# DETECTION STATUS BELOW CAMERA
# ============================================================

st.markdown(
    '<div class="section">Detection Status</div>',
    unsafe_allow_html=True
)

status_box = st.empty()

# STATUS
# ============================================================

st.markdown(
    '<div class="section">Status</div>',
    unsafe_allow_html=True
)

status_box = st.empty()


# ============================================================
# ALARM SOUND
#
# Browser sound is used because Streamlit runs in the browser.
# The original Windows winsound cannot directly control the
# user's browser audio.
# ============================================================

sound_box = st.empty()


def play_alarm_sound():

    components.html(
        """
        <script>
        const AudioContext =
            window.AudioContext ||
            window.webkitAudioContext;

        const ctx = new AudioContext();

        function beep(freq, duration, delay) {

            setTimeout(() => {

                const oscillator =
                    ctx.createOscillator();

                const gain =
                    ctx.createGain();

                oscillator.type = "square";

                oscillator.frequency.value =
                    freq;

                gain.gain.setValueAtTime(
                    0.18,
                    ctx.currentTime
                );

                oscillator.connect(gain);
                gain.connect(ctx.destination);

                oscillator.start();

                oscillator.stop(
                    ctx.currentTime + duration
                );

            }, delay);
        }

        if (ctx.state === "suspended") {
            ctx.resume();
        }

        beep(1200, 0.40, 0);
        beep(900, 0.40, 550);
        beep(1200, 0.40, 1100);
        </script>
        """,
        height=1
    )


# ============================================================
# DASHBOARD REFRESH
# ============================================================

if ctx.state.playing:

    refresh = st.empty()

    while ctx.state.playing:

        with app_state.lock:

            ear = app_state.ear
            mar = app_state.mar
            eye_prob = app_state.eye_prob
            yawn_flag = app_state.yawn_flag
            yawn_prob = app_state.yawn_prob
            gaze = app_state.gaze
            score = app_state.score
            fps = app_state.fps
            closed_thresh = (
                app_state.closed_thresh
            )
            alert = app_state.alert
            reason = app_state.alert_reason

            if app_state.eye_closed_start is not None:
                eye_closed_duration = (
                    time.time()
                    - app_state.eye_closed_start
                )
            else:
                eye_closed_duration = 0.0

            yawn_duration = app_state.yawn_duration

        # ----------------------------------------------------
        # STATUS

        if alert:

            status_box.markdown(
                f'''
                <div style="
                    background:#fff1f2;
                    border:2px solid #e11d48;
                    color:#9f1239;
                    padding:20px;
                    border-radius:12px;
                    text-align:center;
                    font-size:21px;
                    font-weight:700;
                    box-shadow:0 4px 18px rgba(0,0,0,.10);
                ">
                    DROWSINESS ALERT
                    <br>
                    <span style="
                        font-size:15px;
                        font-weight:400;
                    ">
                        {reason}
                    </span>
                </div>
                ''',
                unsafe_allow_html=True
            )

            # Play sound once for this alarm event.
            if (
                app_state.sound_event_played
                != app_state.last_alert_event
            ):
                play_alarm_sound()
                app_state.sound_event_played = (
                    app_state.last_alert_event
                )

        elif eye_closed_duration > 0:

            status_box.markdown(
                f'''
                <div class="warning">
                    Eyes closed: {eye_closed_duration:.1f} seconds
                </div>
                ''',
                unsafe_allow_html=True
            )

        elif yawn_flag:

            status_box.markdown(
                f'''
                <div class="warning">
                    Yawning detected: {yawn_duration:.1f} seconds
                </div>
                ''',
                unsafe_allow_html=True
            )

        else:

            status_box.markdown(
                '''
                <div class="normal">
                    Driver appears alert
                </div>
                ''',
                unsafe_allow_html=True
            )

        yawn_prob_text = (
            f"{yawn_prob:.2f}"
            if yawn_prob is not None
            else "n/a"
        )

        ear_text = (
            f"{ear:.3f}"
            if ear is not None
            else "n/a"
        )

        mar_text = (
            f"{mar:.3f}"
            if mar is not None
            else "n/a"
        )

        st.caption(
            f"Eye closure: {eye_closed_duration:.1f}s  |  "
            f"Yawn duration: {yawn_duration:.1f}s  |  "
            f"EAR: {ear_text}  |  "
            f"MAR: {mar_text}  |  "
            f"Yawn probability: {yawn_prob_text}  |  "
            f"Score: {score:.1f}  |  "
            f"Gaze: {gaze}  |  "
            f"FPS: {fps:.1f}"
        )

        time.sleep(0.25)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#9ca3af;
        font-size:12px;
        margin-top:35px;
        padding-bottom:20px;
    ">
        OBSERA · Driver Drowsiness Monitoring
    </div>
    """,
    unsafe_allow_html=True
)
