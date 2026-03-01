"""
DeepShield — Deepfake Detection
Run from project root: streamlit run app.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "saved_models", "best_model.pth")

# Auto-download from GitHub release if model is missing
MODEL_RELEASE_URL = "https://github.com/priyankagnana/DeepShield/releases/download/model/best_model.pth"


def _ensure_model():
    """Download best_model.pth from GitHub release if not present locally."""
    if os.path.isfile(MODEL_PATH):
        return True
    try:
        import urllib.request
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_RELEASE_URL, MODEL_PATH)
        return os.path.isfile(MODEL_PATH)
    except Exception:
        return False


st.set_page_config(
    page_title="DeepShield",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — minimal, theme-agnostic (no hardcoded light/dark colors)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Give the page enough top breathing room so the title isn't clipped */
.block-container { padding-top: 2.8rem !important; }

/* Page title */
.ds-title {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.3px;
    margin: 0 0 2px;
}
.ds-subtitle {
    font-size: 0.83rem;
    opacity: 0.5;
    margin: 0 0 20px;
}

/* Verdict row */
.verdict-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    border-radius: 10px;
    border: 1px solid;
    margin: 14px 0 10px;
}
.verdict-row.real {
    border-color: #2d8f4e;
    background: rgba(45,143,78,0.08);
}
.verdict-row.fake {
    border-color: #b94040;
    background: rgba(185,64,64,0.08);
}
.verdict-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
}
.verdict-dot.real { background: #2d8f4e; }
.verdict-dot.fake { background: #b94040; }
.verdict-label {
    font-size: 1.15rem;
    font-weight: 700;
}
.verdict-label.real { color: #2d8f4e; }
.verdict-label.fake { color: #b94040; }
.verdict-meta {
    font-size: 0.82rem;
    opacity: 0.6;
    margin-left: auto;
}

/* Confidence bar */
.conf-wrap { margin: 6px 0 14px; }
.conf-label { font-size: 0.78rem; opacity: 0.55; margin-bottom: 4px; }
.conf-track {
    height: 6px;
    border-radius: 3px;
    background: rgba(128,128,128,0.15);
    overflow: hidden;
}
.conf-fill-real { background: #2d8f4e; height: 100%; border-radius: 3px; }
.conf-fill-fake { background: #b94040; height: 100%; border-radius: 3px; }

/* Section label */
.sec-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.45;
    margin: 18px 0 8px;
}

/* Divider */
.thin-hr { border: none; border-top: 1px solid rgba(128,128,128,0.15); margin: 16px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def _load_model():
    from inference.predict import load_model
    return load_model(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Shared UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def _verdict_row(label: str, confidence: float, prob_real: float, note: str = ""):
    cls = "real" if label == "Real" else "fake"
    fake_pct = (1 - prob_real) * 100
    real_pct = prob_real * 100
    meta = f"P(Real) {real_pct:.1f}%  ·  P(Fake) {fake_pct:.1f}%"
    if note:
        meta += f"  ·  {note}"
    fill = f"{confidence * 100:.1f}%"
    st.markdown(f"""
<div class="verdict-row {cls}">
  <div class="verdict-dot {cls}"></div>
  <span class="verdict-label {cls}">{label}</span>
  <span class="verdict-meta">{meta}</span>
</div>
<div class="conf-wrap">
  <div class="conf-label">Confidence — {confidence:.1%}</div>
  <div class="conf-track">
    <div class="conf-fill-{cls}" style="width:{fill}"></div>
  </div>
</div>
""", unsafe_allow_html=True)


def _donut(prob_real: float):
    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Pie(
            values=[prob_real, 1 - prob_real],
            labels=["Real", "Fake"],
            hole=0.65,
            marker_colors=["#2d8f4e", "#b94040"],
            textinfo="label+percent",
            textfont=dict(size=12),
            hovertemplate="%{label}: %{value:.3f}<extra></extra>",
        ))
        fig.add_annotation(
            text=f"{'Real' if prob_real >= 0.5 else 'Fake'}",
            font=dict(size=16, family="sans-serif"),
            showarrow=False,
        )
        fig.update_layout(
            margin=dict(t=8, b=8, l=8, r=8),
            height=220,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.metric("P(Real)", f"{prob_real:.3f}")
        st.metric("P(Fake)", f"{1 - prob_real:.3f}")


def _authenticity_gauge(prob_real: float):
    """Horizontal 0–100% authenticity (P(Real)) gauge."""
    try:
        import plotly.graph_objects as go
        pct = prob_real * 100
        color = "#2d8f4e" if prob_real >= 0.5 else "#b94040"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number=dict(suffix="%", font=dict(size=22)),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1),
                bar=dict(color=color),
                steps=[dict(range=[0, 100], color="rgba(128,128,128,0.1)")],
                threshold=dict(line=dict(color="#b94040", width=2), thickness=0.8, value=50),
            ),
            title=dict(text="Authenticity (P(Real))", font=dict(size=13)),
        ))
        fig.update_layout(
            margin=dict(t=36, b=24, l=24, r=24),
            height=180,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.metric("Authenticity", f"{prob_real:.1%}")


def _certainty_badge(prob_real: float):
    """Decision margin and High/Medium/Low certainty label."""
    margin = abs(prob_real - 0.5)
    if margin >= 0.4:
        level, color = "High", "#2d8f4e"
    elif margin >= 0.2:
        level, color = "Medium", "#7a8f2d"
    else:
        level, color = "Low", "#8f6b2d"
    st.markdown(
        f'<div style="font-size:0.8rem; opacity:0.7;">'
        f'Decision margin: <strong>{margin:.2f}</strong>  ·  '
        f'Certainty: <span style="color:{color}; font-weight:600;">{level}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _gradcam_histogram(heatmap: np.ndarray):
    """Histogram of Grad-CAM heatmap values (where the model focused)."""
    try:
        import plotly.graph_objects as go
        flat = heatmap.flatten()
        flat = flat[flat > 0]  # ignore zeros
        if flat.size == 0:
            st.caption("No activation data.")
            return
        fig = go.Figure(go.Histogram(
            x=flat,
            nbinsx=25,
            marker_color="rgba(74,127,165,0.7)",
            hovertemplate="Intensity ≈ %{x:.2f}<br>Pixels: %{y}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Attention intensity",
            yaxis_title="Pixel count",
            height=160,
            margin=dict(t=8, b=36, l=40, r=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        fig.update_xaxes(gridcolor="rgba(128,128,128,0.1)")
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


def _gradcam_legend():
    """Color bar: Low contribution → High contribution (JET-like)."""
    legend = np.zeros((20, 200, 3), dtype=np.uint8)
    for i in range(200):
        # Approximate JET: blue (low) → cyan → green → yellow → red (high)
        t = i / 199.0
        if t < 0.25:
            r = 0
            g = int(4 * t * 255)
            b = 255
        elif t < 0.5:
            r = 0
            g = 255
            b = int((1 - 4 * (t - 0.25)) * 255)
        elif t < 0.75:
            r = int(4 * (t - 0.5) * 255)
            g = 255
            b = 0
        else:
            r = 255
            g = int((1 - 4 * (t - 0.75)) * 255)
            b = 0
        legend[:, i] = [b, g, r]  # BGR for cv2
    st.image(legend, channels="BGR", width=200)
    st.caption("Low contribution ← → High contribution")


def _timeline(frame_results: list, show_uncertainty_band: bool = True):
    """P(Real) per frame with optional shaded 0.4–0.6 uncertainty zone."""
    try:
        import plotly.graph_objects as go
        frames = [r["frame_idx"] for r in frame_results]
        probs  = [r["prob_real"] for r in frame_results]
        pt_colors = ["#2d8f4e" if p >= 0.5 else "#b94040" for p in probs]
        fig = go.Figure()
        if show_uncertainty_band and frames:
            fig.add_hrect(
                y0=0.4, y1=0.6,
                fillcolor="rgba(128,128,128,0.12)",
                line_width=0,
                annotation_text="Uncertain",
                annotation_position="top right",
                annotation_font_size=10,
            )
        fig.add_hline(
            y=0.5, line_dash="dot", line_color="rgba(128,128,128,0.5)",
            annotation_text="threshold", annotation_font_size=11,
            annotation_position="bottom right",
        )
        fig.add_trace(go.Scatter(
            x=frames, y=probs,
            mode="lines+markers",
            line=dict(color="rgba(74,127,165,0.7)", width=1.8),
            marker=dict(color=pt_colors, size=7, line=dict(width=1, color="white")),
            name="P(Real)",
            hovertemplate="Frame %{x}<br>P(Real) = %{y:.3f}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Frame",
            yaxis_title="P(Real)",
            yaxis=dict(range=[0, 1], gridcolor="rgba(128,128,128,0.1)"),
            xaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
            height=240,
            margin=dict(t=10, b=36, l=36, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        import pandas as pd
        df = pd.DataFrame(frame_results)
        st.line_chart(df.set_index("frame_idx")["prob_real"], use_container_width=True)


def _frame_ratio_donut(real_votes: int, total: int):
    """Small donut: share of frames classified Real vs Fake."""
    if total == 0:
        return
    try:
        import plotly.graph_objects as go
        fake_votes = total - real_votes
        fig = go.Figure(go.Pie(
            values=[real_votes, fake_votes],
            labels=["Real", "Fake"],
            hole=0.6,
            marker_colors=["#2d8f4e", "#b94040"],
            textinfo="label+percent",
            textfont=dict(size=11),
            hovertemplate="%{label}: %{value} frames<extra></extra>",
        ))
        fig.update_layout(
            margin=dict(t=8, b=8, l=8, r=8),
            height=180,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.metric("Real frames", real_votes)
        st.metric("Fake frames", total - real_votes)


def _histogram(frame_results: list):
    try:
        import plotly.graph_objects as go
        probs = [r["prob_real"] for r in frame_results]
        fig = go.Figure(go.Histogram(
            x=probs, nbinsx=10,
            marker_color="rgba(74,127,165,0.65)",
            hovertemplate="P(Real) ≈ %{x:.2f}<br>Frames: %{y}<extra></extra>",
        ))
        fig.add_vline(
            x=0.5, line_dash="dot",
            line_color="rgba(185,64,64,0.6)",
            annotation_text="0.5",
            annotation_font_size=11,
        )
        fig.update_layout(
            xaxis_title="P(Real)",
            yaxis_title="Frames",
            yaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
            xaxis=dict(gridcolor="rgba(128,128,128,0.1)"),
            height=240,
            margin=dict(t=10, b=36, l=36, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass


def _sec(text: str):
    st.markdown(f'<div class="sec-label">{text}</div>', unsafe_allow_html=True)


def _hr():
    st.markdown('<hr class="thin-hr">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Image tab
# ─────────────────────────────────────────────────────────────────────────────
def _image_tab(show_gradcam: bool):
    file = st.file_uploader(
        "Upload an image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )
    if file is None:
        return

    with st.spinner("Analyzing…"):
        suffix = os.path.splitext(file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            path = tmp.name
        try:
            if show_gradcam:
                from inference.predict import predict_with_gradcam
                out = predict_with_gradcam(MODEL_PATH, path)
            else:
                from inference.predict import predict_image
                out = predict_image(MODEL_PATH, path)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            os.unlink(path)
            return

    _hr()
    _verdict_row(out["label"], out["confidence"], out["prob_real"])
    _certainty_badge(out["prob_real"])

    if show_gradcam:
        c1, c2, c3 = st.columns([1, 1, 0.9])
        with c1:
            _sec("Original")
            st.image(path, width=300)
        with c2:
            _sec("Grad-CAM")
            st.image(out["overlay"], channels="BGR", width=300)
            _gradcam_legend()
        with c3:
            _sec("Authenticity")
            _authenticity_gauge(out["prob_real"])
            _sec("Probability Split")
            _donut(out["prob_real"])
        _sec("Grad-CAM attention distribution")
        _gradcam_histogram(out["heatmap"])
    else:
        c1, c2 = st.columns([1, 0.95])
        with c1:
            _sec("Uploaded Image")
            st.image(path, width=340)
        with c2:
            _sec("Authenticity")
            _authenticity_gauge(out["prob_real"])
            _sec("Probability Split")
            _donut(out["prob_real"])

    os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# Video tab
# ─────────────────────────────────────────────────────────────────────────────
def _video_tab(show_gradcam: bool):
    up_col, cfg_col = st.columns([2, 1])
    with up_col:
        file = st.file_uploader(
            "Upload a video (MP4 / AVI / MOV)",
            type=["mp4", "avi", "mov"],
            label_visibility="visible",
        )
    with cfg_col:
        num_frames = st.slider(
            "Frames to sample", min_value=4, max_value=32, value=16, step=4,
            help="More frames → more accurate, slower",
        )

    if file is None:
        return

    suffix = os.path.splitext(file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        vid_path = tmp.name

    try:
        with st.spinner(f"Sampling {num_frames} frames…"):
            from inference.predict import predict_video
            result = predict_video(MODEL_PATH, vid_path, num_frames=num_frames)

        real_votes = sum(1 for r in result["frame_results"] if r["label"] == "Real")
        fake_votes = result["frames_analyzed"] - real_votes

        _hr()
        _verdict_row(
            result["label"], result["confidence"], result["prob_real"],
            note=f"{real_votes} real / {fake_votes} fake frames",
        )
        _certainty_badge(result["prob_real"])

        # Metrics row: frames, avg P(Real), real/fake counts, stability (std)
        probs = [r["prob_real"] for r in result["frame_results"]] if result["frame_results"] else []
        stability_std = float(np.std(probs)) if len(probs) > 1 else 0.0
        stability_label = "Stable" if stability_std < 0.15 else ("Mixed" if stability_std < 0.3 else "Variable")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Frames Analyzed", result["frames_analyzed"])
        m2.metric("Avg P(Real)", f"{result['prob_real']:.3f}")
        m3.metric("Real Frames", real_votes)
        m4.metric("Fake Frames", fake_votes)
        m5.metric("Score Stability", f"{stability_std:.2f} ({stability_label})")

        _hr()

        # Authenticity gauge + Frame ratio donut
        gauge_col, ratio_col, _ = st.columns([1, 1, 1])
        with gauge_col:
            _sec("Video Authenticity")
            _authenticity_gauge(result["prob_real"])
        with ratio_col:
            _sec("Real vs Fake Frames")
            _frame_ratio_donut(real_votes, result["frames_analyzed"])

        _hr()

        # Charts
        if result["frame_results"]:
            ch1, ch2 = st.columns([2, 1])
            with ch1:
                _sec("P(Real) per Frame")
                _timeline(result["frame_results"], show_uncertainty_band=True)
            with ch2:
                _sec("Score Distribution")
                _histogram(result["frame_results"])

        # Per-frame table
        with st.expander("Per-frame details"):
            import pandas as pd
            df = pd.DataFrame(result["frame_results"])
            df = df.rename(columns={
                "frame_idx": "Frame",
                "label": "Verdict",
                "prob_real": "P(Real)",
                "confidence": "Confidence",
            })
            st.dataframe(
                df[["Frame", "Verdict", "P(Real)", "Confidence"]],
                use_container_width=True,
                hide_index=True,
            )

        # First frame preview — constrained width
        cap = cv2.VideoCapture(vid_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            if show_gradcam:
                from inference.predict import get_transform, preprocess_image
                from explainability.gradcam import generate_gradcam
                from explainability.heatmap_utils import overlay_heatmap
                import torch
                model, device = _load_model()
                t_fn = get_transform()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = preprocess_image(frame_rgb, t_fn)
                t = tensor.clone().detach().to(device).requires_grad_(True)
                heatmap = generate_gradcam(model, t,
                                           target_layer=model.backbone.features[-1])
                frame = overlay_heatmap(heatmap, frame, alpha=0.45)

            # Subtle text annotation — white outline on dark, dark outline on light
            lbl = f"{result['label']}  {result['confidence']:.0%}"
            cv2.putText(frame, lbl, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
            cv2.putText(frame, lbl, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)

            _hr()
            _sec("First Frame Preview")
            _, fc, _ = st.columns([0.2, 0.6, 0.2])
            with fc:
                st.image(frame, channels="BGR", use_container_width=True)
                if result["frame_results"]:
                    first = result["frame_results"][0]
                    st.caption(
                        f"Frame 0 verdict: {first['label']}  ·  "
                        f"P(Real) = {first['prob_real']:.3f}  ·  "
                        f"Confidence = {first['confidence']:.2f}"
                    )
                if show_gradcam:
                    _gradcam_legend()

    finally:
        os.unlink(vid_path)


# ─────────────────────────────────────────────────────────────────────────────
# Webcam tab
# ─────────────────────────────────────────────────────────────────────────────
def _webcam_tab():
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        import av

        model, device = _load_model()

        # Result placeholder rendered ABOVE the streamer
        result_placeholder = st.empty()

        class _Processor(VideoProcessorBase):
            def __init__(self):
                from inference.predict import get_transform
                self._transform = get_transform()
                self.label = ""
                self.confidence = 0.0
                self.prob_real = 0.5
                self._n = 0

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                if self._n % 3 == 0:
                    from inference.predict import preprocess_image, predict
                    import torch
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tensor = preprocess_image(rgb, self._transform)
                    with torch.no_grad():
                        self.label, self.confidence = predict(model, tensor, device)
                        logit = model(tensor.to(device))[0, 0].item()
                        self.prob_real = torch.sigmoid(torch.tensor(logit)).item()
                self._n += 1

                # Only keep the thin P(Real) progress bar at the top of the frame
                h, w = img.shape[:2]
                bar_w = int(w * self.prob_real)
                bar_col = (45, 143, 78) if self.prob_real >= 0.5 else (64, 64, 185)
                cv2.rectangle(img, (0, 0), (w, 5), (50, 50, 50), -1)
                cv2.rectangle(img, (0, 0), (bar_w, 5), bar_col, -1)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        RTC_CONFIG = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        _, cam_col, _ = st.columns([0.05, 0.9, 0.05])
        with cam_col:
            ctx = webrtc_streamer(
                key="deepshield-webcam",
                video_processor_factory=_Processor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={"video": True, "audio": False},
            )

        # Show latest prediction above the video: verdict, certainty, live gauge, status
        if ctx and ctx.video_processor:
            proc = ctx.video_processor
            if proc.label:
                with result_placeholder.container():
                    st.markdown(
                        '<span style="font-size:0.7rem; font-weight:600; opacity:0.6;">LIVE</span>',
                        unsafe_allow_html=True,
                    )
                    _verdict_row(proc.label, proc.confidence, proc.prob_real)
                    _certainty_badge(proc.prob_real)
                    _sec("Live Authenticity")
                    st.progress(float(proc.prob_real), text=f"P(Real) = {proc.prob_real:.0%}")
            else:
                result_placeholder.caption("Starting analysis…")

        st.caption(
            "**How to read:** Top bar = P(Real) (green = real, red = fake). "
            "Verdict above updates every 3 frames. Use **Settings** (top-right) to switch theme."
        )

    except ImportError:
        st.info("Install streamlit-webrtc to enable the live webcam feed.")
        st.code("pip install streamlit-webrtc av", language="bash")
        st.caption(
            "Alternatively, run the CLI script for webcam inference:  "
            "`python -m inference.realtime_inference [--gradcam]`"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar() -> bool:
    with st.sidebar:
        st.markdown("**DeepShield**")
        st.caption("Deepfake detection · offline · explainable")
        st.divider()

        show_gradcam = st.checkbox(
            "Grad-CAM overlay",
            value=False,
            help="Shows which facial regions influenced the prediction. Adds ~1–2 s.",
        )

        st.divider()
        st.markdown("**Score guide**")
        st.caption("P(Real) > 0.9 → likely authentic")
        st.caption("P(Real) 0.6–0.9 → probably real")
        st.caption("P(Real) 0.4–0.6 → uncertain")
        st.caption("P(Real) < 0.4 → likely deepfake")

        st.divider()
        st.caption("Model: EfficientNet-B0")
        st.caption("Dataset: 140k-faces")
        st.caption("Inference: 100% offline")

        st.divider()
        st.caption("**Theme:** Use the menu (⋮) at top-right → Settings → Theme (Light / Dark).")

    return show_gradcam


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="ds-title">DeepShield</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ds-subtitle">Real-time deepfake detection · '
        'EfficientNet-B0 · Grad-CAM · Offline</div>',
        unsafe_allow_html=True,
    )

    if not os.path.isfile(MODEL_PATH):
        with st.spinner("Downloading model from GitHub…"):
            if not _ensure_model():
                st.error(
                    f"Model not found at `{MODEL_PATH}` and download failed. "
                    "Train locally with `python -m training.train` or check your connection."
                )
                return

    show_gradcam = _sidebar()

    tab_img, tab_vid, tab_cam = st.tabs(["Image", "Video", "Webcam"])

    with tab_img:
        _image_tab(show_gradcam)

    with tab_vid:
        _video_tab(show_gradcam)

    with tab_cam:
        _webcam_tab()


if __name__ == "__main__":
    main()
    