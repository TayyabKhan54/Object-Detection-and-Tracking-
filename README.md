# Fieldnote Object Tracking

Real-time object detection and tracking for a webcam or uploaded video. Frames are handled with OpenCV, objects are detected with Ultralytics YOLO, and Deep SORT maintains track IDs across frames.

## Run

Use Python 3.10-3.13, then install the dependencies and launch Streamlit:

```powershell
cd "3"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The first detector load downloads the selected model weights. Camera permission is required for live mode; upload mode accepts MP4, MOV, AVI, MKV, and M4V files.

## Notes

- Inference and tracking run on the server running Streamlit. The app does not upload video to a hosted inference service.
- The YOLO nano models are the quickest to run. Larger models may improve detection quality at the cost of speed.
- The first few frames are used to confirm each Deep SORT track before its ID is drawn.