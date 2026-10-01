from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import av
import cv2
import streamlit as st
from deep_sort_realtime.deepsort_tracker import DeepSort
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO


st.set_page_config(page_title="Fieldnote | Object tracking", page_icon="◉", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#17231f; --muted:#697871; --paper:#f1f3ed; --line:#d8dfd8; --green:#17634c; --lime:#d5ee72; --coral:#d95c3d; }
    .stApp { background: radial-gradient(ellipse at 92% 0%, #e2ead7 0, transparent 32%), var(--paper); color:var(--ink); font-family:'Manrope',sans-serif; }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1440px; padding-top:1.25rem; padding-bottom:2rem; }
    h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:0 !important; color:var(--ink); }
    .brandline { display:flex; align-items:center; justify-content:space-between; padding:4px 0 16px; border-bottom:1px solid var(--line); }
    .brand { font-size:18px; font-weight:800; letter-spacing:0; }
    .brand i { display:inline-grid; place-items:center; width:30px; height:30px; margin-right:8px; border-radius:7px; background:var(--green); color:white; font-style:normal; }
    .micro { color:var(--muted); font:10px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:0; }
    .hero { display:flex; align-items:end; justify-content:space-between; gap:24px; padding:28px 0 23px; }
    .eyebrow { margin:0 0 8px; color:var(--green); font:10px 'DM Mono',monospace; text-transform:uppercase; }
    .hero h1 { margin:0; font-size:clamp(28px,4vw,42px); line-height:1.08; }
    .hero p { max-width:390px; margin:0; color:var(--muted); font-size:13px; line-height:1.65; }
    .panel { padding:18px; border:1px solid var(--line); border-radius:8px; background:#fffefa; }
    .panel-title { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px; font-size:14px; font-weight:800; }
    .stage { overflow:hidden; border:1px solid #28332e; border-radius:7px; background:#17211d; }
    .stage-label { display:flex; justify-content:space-between; padding:10px 13px; border-bottom:1px solid #34413a; color:#bec8c0; font:10px 'DM Mono',monospace; text-transform:uppercase; }
    .stage-label b { color:#d5ee72; font-weight:500; }
    .metric { padding:10px 0; border-top:1px solid var(--line); }
    .metric strong { display:block; font-size:21px; line-height:1; }
    .metric span { display:block; margin-top:6px; color:var(--muted); font:9px 'DM Mono',monospace; text-transform:uppercase; }
    .note { color:var(--muted); font-size:11px; line-height:1.6; }
    div[data-testid="stFileUploader"] { border:1px dashed #bbc8bc; border-radius:6px; background:#f8faf5; }
    .stButton > button { border-radius:5px; font-weight:700; }
    .stButton > button[kind="primary"] { border-color:var(--green); background:var(--green); }
    @media(max-width:700px) { .block-container{padding:1rem 1rem 2rem} .hero{display:block;padding:23px 0 16px}.hero p{margin-top:10px} }
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_OPTIONS = ["yolo11n.pt", "yolo11s.pt", "yolov8n.pt"]
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


@st.cache_resource(show_spinner=False)
def load_detector(weights: str) -> YOLO:
    return YOLO(weights)


def new_tracker() -> DeepSort:
    return DeepSort(max_age=30, n_init=2, max_iou_distance=0.7)


def predict_detections(
    model: YOLO, frame: Any, confidence: float
) -> list[tuple[list[float], float, str]]:
    result = model.predict(frame, conf=confidence, verbose=False)[0]
    names = result.names
    detections: list[tuple[list[float], float, str]] = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].cpu().tolist())
        class_id = int(box.cls[0].item())
        detections.append(
            ([x1, y1, x2 - x1, y2 - y1], float(box.conf[0].item()), str(names[class_id]))
        )
    return detections


def track_frame(
    frame: Any,
    model: YOLO,
    tracker: DeepSort,
    confidence: float,
) -> tuple[Any, int, set[int]]:
    detections = predict_detections(model, frame, confidence)
    tracks = tracker.update_tracks(detections, frame=frame)
    visible_ids: set[int] = set()
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        left, top, right, bottom = (int(value) for value in track.to_ltrb())
        track_id = int(track.track_id)
        visible_ids.add(track_id)
        class_name = str(track.get_det_class() or "object")
        color = ((track_id * 67 + 65) % 190 + 50, (track_id * 37 + 95) % 150 + 70, (track_id * 97 + 40) % 180 + 60)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label = f"{class_name}  ID {track_id}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_top = max(0, top - label_height - 10)
        cv2.rectangle(frame, (left, label_top), (left + label_width + 8, label_top + label_height + 8), color, -1)
        cv2.putText(frame, label, (left + 4, label_top + label_height + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (17, 25, 22), 2, cv2.LINE_AA)
    return frame, len(detections), visible_ids


class LiveTrackingProcessor(VideoProcessorBase):
    def __init__(self, model: YOLO, confidence: float) -> None:
        self.model = model
        self.confidence = confidence
        self.tracker = new_tracker()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        annotated, _, _ = track_frame(image, self.model, self.tracker, self.confidence)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def process_video(upload: Any, model: YOLO, confidence: float, progress: Any, preview: Any) -> bytes:
    suffix = Path(upload.name).suffix or ".mp4"
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / f"input{suffix}"
        output_path = Path(directory) / "tracked.mp4"
        input_path.write_bytes(upload.getvalue())
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("OpenCV could not open this video. Try an MP4, MOV, or AVI file.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 24.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("OpenCV could not create the annotated MP4 output on this system.")

        tracker = new_tracker()
        frame_number = 0
        all_track_ids: set[int] = set()
        detection_total = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                annotated, detection_count, track_ids = track_frame(image, model, tracker, confidence)
                writer.write(annotated)
                detection_total += detection_count
                all_track_ids.update(track_ids)
                frame_number += 1
                if frame_number == 1 or frame_number % 8 == 0:
                    preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                    if total_frames > 0:
                        progress.progress(min(frame_number / total_frames, 1.0), text=f"Frame {frame_number:,} of {total_frames:,}")
            if total_frames == 0:
                progress.progress(1.0, text=f"Processed {frame_number:,} frames")
        finally:
            capture.release()
            writer.release()
        return output_path.read_bytes()


def render_live(model_name: str, confidence: float) -> None:
    st.markdown('<div class="panel-title">Live camera <span class="micro">01 / REAL TIME</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="note">Allow camera access when prompted. The video is processed on this machine; each object receives a persistent track ID.</p>', unsafe_allow_html=True)
    if st.button("Load detector and open camera", type="primary", key="open-camera"):
        st.session_state["live_model"] = model_name
    if st.session_state.get("live_model") != model_name:
        st.info("Load the detector to start the camera. The first run downloads the selected YOLO weights.")
        return
    try:
        with st.spinner(f"Loading {model_name}…"):
            model = load_detector(model_name)
        webrtc_streamer(
            key="object-tracking-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: LiveTrackingProcessor(model, confidence),
            async_processing=True,
        )
    except Exception as error:
        st.error(f"Could not start live tracking: {error}")


def render_video(model_name: str, confidence: float) -> None:
    st.markdown('<div class="panel-title">Video file <span class="micro">02 / FRAME BY FRAME</span></div>', unsafe_allow_html=True)
    upload = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv", "m4v"], help="The video is processed locally. Large files take longer.")
    if upload is None:
        st.markdown('<p class="note">Upload a clip to detect objects, assign tracking IDs, preview the annotated frames, and export the result.</p>', unsafe_allow_html=True)
        return
    st.video(upload)
    if st.button("Run detection and tracking", type="primary", key="process-video"):
        progress = st.progress(0, text="Starting video processing…")
        preview = st.empty()
        try:
            with st.spinner(f"Loading {model_name}…"):
                model = load_detector(model_name)
            result = process_video(upload, model, confidence, progress, preview)
            st.session_state["processed_video"] = result
            st.success("Tracking complete. The exported video includes labels, boxes, and IDs.")
        except Exception as error:
            st.error(f"Video processing failed: {error}")
    result = st.session_state.get("processed_video")
    if result:
        st.video(result)
        st.download_button("Download annotated video", data=result, file_name="tracked-objects.mp4", mime="video/mp4")


st.markdown(
    '<div class="brandline"><div class="brand"><i>◉</i>fieldnote<span style="color:#17634c">.</span></div><div class="micro">COMPUTER VISION / 001 &nbsp; · &nbsp; LOCAL PROCESSING</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<section class="hero"><div><p class="eyebrow">See what moves</p><h1>Objects in frame.<br>Identity over time.</h1></div><p>Detect people and objects in live video or recorded footage. Deep SORT follows each detection across frames and keeps its tracking ID consistent.</p></section>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Detector settings")
    model_name = st.selectbox("YOLO model", MODEL_OPTIONS, help="Nano is fastest; small is more accurate and needs more compute.")
    confidence = st.slider("Detection confidence", min_value=0.10, max_value=0.90, value=0.35, step=0.05, format="%.2f")
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.caption("OpenCV · YOLO · Deep SORT")
    st.markdown('<p class="note">Models run locally after their weights are downloaded once. Camera access requires localhost or a secure HTTPS origin.</p>', unsafe_allow_html=True)

live_tab, video_tab = st.tabs(["Live camera", "Video file"])
with live_tab:
    render_live(model_name, confidence)
with video_tab:
    render_video(model_name, confidence)