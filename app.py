"""
Streamlit app: upload image/video or use webcam, run inference, show Real/Fake and optional Grad-CAM overlay.
Run from project root: streamlit run app.py
"""

import os
import sys

import streamlit as st

# Project root for imports
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "saved_models", "best_model.pth")


def main():
    st.set_page_config(page_title="DeepShield", page_icon="🛡️")
    st.title("DeepShield – Deepfake Detection")

    if not os.path.isfile(MODEL_PATH):
        st.error(f"Model not found at `{MODEL_PATH}`. Train the model first (e.g. run `python -m training.train`).")
        return

    mode = st.radio("Input", ["Upload image", "Upload video", "Webcam"], horizontal=True)

    show_gradcam = st.checkbox("Show Grad-CAM overlay (slower)", value=False)

    if mode == "Upload image":
        file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if file is not None:
            with st.spinner("Running inference…"):
                from inference.predict import predict_image, predict_with_gradcam
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[-1]) as tmp:
                    tmp.write(file.read())
                    path = tmp.name
                try:
                    if show_gradcam:
                        out = predict_with_gradcam(MODEL_PATH, path)
                        st.subheader(f"**{out['label']}** (confidence: {out['confidence']:.2%})")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(out["overlay"], channels="BGR", use_container_width=True, caption="Grad-CAM overlay")
                        with col2:
                            st.image(path, use_container_width=True, caption="Original")
                    else:
                        out = predict_image(MODEL_PATH, path)
                        st.subheader(f"**{out['label']}** (confidence: {out['confidence']:.2%})")
                        st.image(path, use_container_width=True)
                finally:
                    os.unlink(path)

    elif mode == "Upload video":
        file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
        if file is not None:
            import tempfile
            import cv2
            from inference.predict import load_model, get_transform, preprocess_image, predict
            suffix = os.path.splitext(file.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.read())
                vid_path = tmp.name
            try:
                with st.spinner("Loading model and running on first frame…"):
                    model, device = load_model(MODEL_PATH)
                    transform = get_transform()
                    cap = cv2.VideoCapture(vid_path)
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        st.error("Could not read video.")
                    else:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        tensor = preprocess_image(frame_rgb, transform)
                        label, confidence = predict(model, tensor, device)
                        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "Real" else (0, 0, 255), 2)
                        st.image(frame, channels="BGR", use_container_width=True, caption="First frame prediction")
                        st.caption("For full-video real-time inference use: `python -m inference.realtime_inference --video <path>`")
            finally:
                os.unlink(vid_path)

    else:  # Webcam
        st.info("Use the realtime script for webcam: `python -m inference.realtime_inference [--gradcam]`")
        st.caption("Streamlit does not support direct webcam capture in all environments; Member 2/3 can add a proper webcam flow (e.g. via FastAPI + frontend).")


if __name__ == "__main__":
    main()
