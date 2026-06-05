import streamlit as st
import json
import random
import plotly.graph_objects as go

st.set_page_config(
    page_title="EN→DE Translation Comparison",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for mobile + clean look
st.markdown("""
<style>
    .block-container { max-width: 900px; padding-top: 2rem; }
    .stButton > button { width: 100%; }
    div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }
    .model-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #ccc;
    }
    .model-card.nllb { border-left-color: #ff6b6b; }
    .model-card.pipeline { border-left-color: #4ecdc4; }
    .model-card.finetuned { border-left-color: #45b7d1; }
    .model-name { font-weight: 700; font-size: 0.9rem; margin-bottom: 0.3rem; }
    .model-scores { font-size: 0.75rem; color: #666; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    with open("predictions/translations.json", "r") as f:
        return json.load(f)

data = load_data()

# Header
st.title("🌐 EN → DE Translation Comparison")
st.caption(f"comparing NLLB, MarianMT Pipeline and Fine-Tuned MarianMT on {len(data)} Europarl test sentences")

# Session state
if "index" not in st.session_state:
    st.session_state.index = 0

example = data[st.session_state.index]

# Source & Reference
st.markdown("#### Source (EN)")
st.info(example["source"])

st.markdown("#### Reference (DE)")
st.success(example["reference"])

# Navigation buttons centered
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col1:
    if st.button("⬅ Prev", use_container_width=True):
        st.session_state.index = max(0, st.session_state.index - 1)
        st.rerun()
with col2:
    if st.button("🎲 Random", use_container_width=True):
        st.session_state.index = random.randint(0, len(data) - 1)
        st.rerun()
with col3:
    if st.button("Next ➡", use_container_width=True):
        st.session_state.index = min(len(data) - 1, st.session_state.index + 1)
        st.rerun()
with col4:
    st.markdown(f"<div style='text-align:center; padding-top:0.5rem; color:#888;'>{st.session_state.index + 1} / {len(data)}</div>", unsafe_allow_html=True)

st.markdown("---")

# Model translations as cards
st.markdown("#### Model Translations")

models = [
    ("NLLB-200", "nllb", "nllb", "#ff6b6b"),
    ("MarianMT Pipeline", "pipeline", "pipeline", "#4ecdc4"),
    ("MarianMT Fine-Tuned", "finetuned", "finetuned", "#45b7d1"),
]

for label, key, css_class, color in models:
    m = example[key]
    st.markdown(f"""
    <div class="model-card {css_class}">
        <div class="model-name">{label}</div>
        {m['translation']}
        <div class="model-scores">
            BLEU: {m['bleu']:.1f} · METEOR: {m['meteor']:.3f} · chrF: {m['chrf']:.1f}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Score comparison with plotly
st.markdown("#### Score Comparison")

metrics = ["BLEU", "METEOR (×100)", "chrF"]
nllb_scores = [example["nllb"]["bleu"], example["nllb"]["meteor"] * 100, example["nllb"]["chrf"]]
pipeline_scores = [example["pipeline"]["bleu"], example["pipeline"]["meteor"] * 100, example["pipeline"]["chrf"]]
finetuned_scores = [example["finetuned"]["bleu"], example["finetuned"]["meteor"] * 100, example["finetuned"]["chrf"]]

fig = go.Figure()
fig.add_trace(go.Bar(name="NLLB", x=metrics, y=nllb_scores, marker_color="#ff6b6b"))
fig.add_trace(go.Bar(name="Pipeline", x=metrics, y=pipeline_scores, marker_color="#4ecdc4"))
fig.add_trace(go.Bar(name="Fine-Tuned", x=metrics, y=finetuned_scores, marker_color="#45b7d1"))

fig.update_layout(
    barmode="group",
    height=350,
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    yaxis_title="Score",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
fig.update_yaxes(gridcolor="#eee")

st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("MODULE Applied Machine Learning for Language Processing · Group D · Hochschule Campus Wien")