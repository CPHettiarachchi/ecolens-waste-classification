"""
app/app.py
----------
EcoLens Streamlit Web App
Run: streamlit run app/app.py  (from EcoLens root folder)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import time
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="EcoLens — AI Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #40916c 100%);
    padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem; color: white; text-align: center;
}
.main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
.main-header p  { font-size: 0.9rem; opacity: 0.85; margin-top: 0.4rem; }
.pred-card {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 2px solid #86efac; border-radius: 14px;
    padding: 1.5rem; text-align: center; margin: 1rem 0;
}
.pred-card .label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #166534; }
.pred-card .cls   { font-size: 2rem; font-weight: 700; color: #14532d; margin: 0.3rem 0; }
.pred-card .conf  { font-size: 1rem; color: #15803d; }
.tip-card {
    background: #fffbeb; border-left: 4px solid #f59e0b;
    border-radius: 8px; padding: 0.8rem 1.2rem; margin: 0.75rem 0;
    font-size: 0.9rem; color: #78350f;
}
.metric-card {
    background: #f8fffe; border: 1px solid #d0f0e0;
    border-left: 4px solid #40916c; border-radius: 10px;
    padding: 1rem 1.2rem; margin: 0.4rem 0;
}
.metric-card h3   { color: #1a472a; margin: 0 0 0.2rem; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .val { font-size: 1.8rem; font-weight: 700; color: #2d6a4f; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_predictor(model_path):
    try:
        from inference import WastePredictor
        return WastePredictor(model_path=model_path), None
    except FileNotFoundError:
        return None, f"Model not found at: {model_path}. Run training first."
    except Exception as e:
        return None, str(e)


# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    model_path = st.text_input("Model Path", value="models/best_model.pth")
    top_k = st.slider("Top-K Predictions", 1, 6, 3)
    show_all = st.toggle("Show All Class Probabilities", value=True)

    st.divider()
    st.markdown("### 🗑️ Classes")
    for cls in ["📦 Cardboard", "🍶 Glass", "🥫 Metal", "📄 Paper", "🧴 Plastic", "🗑️ Trash"]:
        st.caption(cls)

    st.divider()
    st.markdown("### 📊 Model")
    st.caption("**Architecture:** EfficientNet-B3")
    st.caption("**Classes:** 6 waste categories")
    st.caption("**Framework:** PyTorch + timm")

    try:
        with open("reports/evaluation_report.json") as f:
            eval_data = json.load(f)
        st.metric("Test Accuracy", f"{eval_data['test_accuracy']*100:.1f}%")
        st.metric("Macro F1", f"{eval_data['macro_f1']:.3f}")
    except FileNotFoundError:
        st.caption("Run evaluate.py to see metrics")


st.markdown("""
<div class="main-header">
    <h1>♻️ EcoLens</h1>
    <p>Intelligent Waste Classification · EfficientNet-B3 Transfer Learning</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading model..."):
    predictor, error = load_predictor(model_path)

if error:
    st.error(f"❌ {error}")
    st.info("💡 Train the model first:\n```\npython src/train.py\n```")
    st.stop()

col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("📤 Upload Image")
    uploaded = st.file_uploader("Drop a waste image", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                 label_visibility="collapsed")
    image = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Input Image", use_column_width=True)
        w, h = image.size
        st.caption(f"Resolution: {w}×{h} px")

with col_result:
    st.subheader("🔍 Results")
    if image is None:
        st.info("👈 Upload an image to classify it")
    else:
        with st.spinner("Classifying..."):
            t0 = time.time()
            result = predictor.predict(image, top_k=top_k)
            latency = (time.time() - t0) * 1000

        cls_name = result["top_class"].replace("_", " ").title()
        conf_pct = result["confidence"] * 100

        st.markdown(f"""
        <div class="pred-card">
            <div class="label">Classified As</div>
            <div class="cls">{cls_name}</div>
            <div class="conf">Confidence: {conf_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="tip-card">
            <strong>♻️ Tip:</strong> {result['disposal_tip']}
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"⚡ Inference: {latency:.0f}ms")

        # Top-K bar chart
        topk = result["top_k"]
        labels = [r["class"].replace("_", " ").title() for r in topk]
        values = [r["confidence"] * 100 for r in topk]
        colors = ["#40916c" if i == 0 else "#95d5b2" for i in range(len(topk))]

        fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                               marker_color=colors,
                               text=[f"{v:.1f}%" for v in values],
                               textposition="outside"))
        fig.update_layout(margin=dict(l=0, r=40, t=10, b=10), height=160,
                          xaxis=dict(range=[0, 110], title="Confidence (%)"),
                          yaxis=dict(autorange="reversed"),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        if show_all:
            all_cls   = list(result["all_probs"].keys())
            all_probs = list(result["all_probs"].values())
            fig2 = px.bar(x=all_cls, y=[p*100 for p in all_probs],
                          labels={"x": "Class", "y": "Probability (%)"},
                          color=[p*100 for p in all_probs],
                          color_continuous_scale="Greens")
            fig2.update_layout(margin=dict(l=0,r=0,t=10,b=60), height=240,
                               coloraxis_showscale=False,
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            fig2.update_xaxes(tickangle=30)
            st.plotly_chart(fig2, use_container_width=True)

# Dashboard
st.divider()
st.subheader("📈 Performance Dashboard")

try:
    with open("reports/evaluation_report.json") as f:
        eval_data = json.load(f)
    with open("reports/training_history.json") as f:
        history = json.load(f)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>Test Accuracy</h3><div class="val">{eval_data["test_accuracy"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>Top-3 Accuracy</h3><div class="val">{eval_data["top3_accuracy"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>Macro F1</h3><div class="val">{eval_data["macro_f1"]:.3f}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>Macro Precision</h3><div class="val">{eval_data["macro_precision"]:.3f}</div></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📉 Training Curves", "📊 Per-Class F1"])
    with tab1:
        epochs = list(range(1, len(history["train_acc"]) + 1))
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=epochs, y=history["train_acc"], name="Train Acc", line=dict(color="#40916c", width=2)))
        fig3.add_trace(go.Scatter(x=epochs, y=history["val_acc"],   name="Val Acc",   line=dict(color="#2d6a4f", width=2, dash="dash")))
        fig3.add_trace(go.Scatter(x=epochs, y=history["val_loss"],  name="Val Loss",  line=dict(color="#ef4444", width=2), yaxis="y2"))
        fig3.update_layout(yaxis=dict(title="Accuracy", range=[0,1]),
                           yaxis2=dict(title="Loss", overlaying="y", side="right"),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02),
                           margin=dict(l=10,r=10,t=30,b=10), height=320,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        per_class = eval_data["per_class"]
        cls_list  = list(per_class.keys())
        f1_scores = [per_class[c]["f1-score"] for c in cls_list]
        fig4 = px.bar(x=cls_list, y=f1_scores, color=f1_scores,
                      color_continuous_scale="Greens",
                      labels={"x": "Class", "y": "F1 Score"})
        fig4.update_layout(height=280, coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

except FileNotFoundError:
    st.info("📊 Run `python src/evaluate.py` to populate the performance dashboard.")

st.divider()
st.markdown('<div style="text-align:center;color:#6b7280;font-size:0.8rem;">EcoLens · EfficientNet-B3 · PyTorch · Streamlit</div>', unsafe_allow_html=True)
