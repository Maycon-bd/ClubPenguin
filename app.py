# app.py — MatoScan | Plataforma Integrada de Identificação de Culturas
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import time

# ─── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="MatoScan | Identificação de Culturas",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paleta matplotlib alinhada ao tema AgriTech ─────────────────────────────
plt.rcParams.update({
    'text.color':       '#CBD5C0',
    'axes.labelcolor':  '#CBD5C0',
    'xtick.color':      '#7A9E7E',
    'ytick.color':      '#7A9E7E',
    'figure.facecolor': '#0E1A10',
    'axes.facecolor':   '#0E1A10',
    'axes.edgecolor':   '#1E3A22',
    'grid.color':       '#1A2E1C',
    'axes.spines.top':  False,
    'axes.spines.right': False,
})

# ─── Design System — AgriTech Professional ────────────────────────────────────
st.markdown("""
<style>
/* ── Fontes premium ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #CBD5C0 !important;
    background: #0B1610 !important;
}

/* ── Fundo principal ── */
[data-testid="stAppViewContainer"] { background: #0B1610 !important; }
[data-testid="stHeader"] { background: #0B1610 !important; border-bottom: 1px solid #1A3020; }

/* ── Sidebar — painel de controle ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1A0D 0%, #081208 100%) !important;
    border-right: 1px solid #1E3A22 !important;
}
[data-testid="stSidebar"] * { color: #B8D4B0 !important; }
[data-testid="stSidebar"] label { font-weight: 600 !important; font-size: 0.85rem !important; text-transform: uppercase; letter-spacing: 0.5px; color: #7FAF86 !important; }

/* ── Headings ── */
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #E8F5E4 !important;
    letter-spacing: -0.3px;
}

/* ══════════════════════════════
   HEADER STRIP — Estilo AgriOps
   ══════════════════════════════ */
.agro-header {
    background: linear-gradient(135deg, #112A16 0%, #0D2010 50%, #142818 100%);
    border: 1px solid #1E4025;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 18px;
    position: relative;
    overflow: hidden;
}
.agro-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(80,200,90,0.06) 0%, transparent 70%);
}
.agro-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6FCF7A 0%, #4CAF50 60%, #D4A017 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1;
}
.agro-tagline {
    font-size: 0.8rem;
    color: #5A8A60;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-top: 3px;
}
.agro-version {
    background: rgba(76,175,80,0.12);
    border: 1px solid rgba(76,175,80,0.25);
    border-radius: 20px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #5CB860;
    margin-left: auto;
}
.agro-status {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.78rem; color: #5A8A60;
}
.status-dot {
    width: 7px; height: 7px;
    background: #4CAF50;
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(76,175,80,0.6);
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ══════════════════════════════
   KPI CARDS — Estilo Field Monitor
   ══════════════════════════════ */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}
.kpi-card {
    background: linear-gradient(145deg, #0F2214 0%, #0B1A0F 100%);
    border: 1px solid #1E3A24;
    border-radius: 10px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: rgba(76,175,80,0.35); }
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #4CAF50, #2E7D32);
    border-radius: 10px 0 0 10px;
}
.kpi-card.gold::after { background: linear-gradient(180deg, #D4A017, #A07010); }
.kpi-card.blue::after { background: linear-gradient(180deg, #29B6F6, #0277BD); }
.kpi-card.orange::after { background: linear-gradient(180deg, #FF8F00, #E65100); }
.kpi-label {
    font-size: 0.72rem;
    color: #5A8A60;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #E8F5E4;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #4CAF50;
    margin-top: 4px;
    font-weight: 500;
}

/* ══════════════════════════════
   SECTION CARDS
   ══════════════════════════════ */
.section-card {
    background: #0E1A10;
    border: 1px solid #1E3A22;
    border-radius: 10px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #5A8A60;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}
.section-divider { border: none; border-top: 1px solid #1E3A22; margin: 14px 0; }

/* ══════════════════════════════
   RESULTADO — Identification Card
   ══════════════════════════════ */
.result-card {
    background: linear-gradient(135deg, #0D2A12 0%, #0A1F0D 100%);
    border: 1px solid rgba(76,175,80,0.3);
    border-left: 4px solid #4CAF50;
    border-radius: 10px;
    padding: 20px 22px;
    margin: 14px 0;
    position: relative;
}
.result-badge {
    display: inline-block;
    background: rgba(76,175,80,0.12);
    border: 1px solid rgba(76,175,80,0.3);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.7rem;
    font-weight: 700;
    color: #4CAF50;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.result-culture {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #A5D6A7;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.result-desc {
    font-size: 0.88rem;
    color: #6A9E70;
    margin-top: 8px;
    line-height: 1.5;
}
.confidence-bar {
    background: #1A3020;
    border-radius: 4px;
    height: 6px;
    margin-top: 12px;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #2E7D32, #66BB6A);
    transition: width 0.5s ease;
}

/* ══════════════════════════════
   TABELA DE MEDIÇÕES
   ══════════════════════════════ */
.measure-table { width: 100%; border-collapse: collapse; }
.measure-table td {
    padding: 8px 10px;
    font-size: 0.9rem;
    border-bottom: 1px solid #1A2E1C;
    color: #B8D4B0;
}
.measure-table td:first-child {
    color: #6A9E70;
    font-weight: 600;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    width: 55%;
}
.measure-table td:last-child {
    font-family: 'JetBrains Mono', monospace;
    color: #D4EADA;
    font-weight: 500;
    text-align: right;
}
.measure-table tr:last-child td { border-bottom: none; }

/* ══════════════════════════════
   BOTÃO PRINCIPAL
   ══════════════════════════════ */
div.stButton > button {
    background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%) !important;
    color: #E8F5E4 !important;
    border: 1px solid rgba(76,175,80,0.4) !important;
    padding: 11px 24px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.3px !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 12px rgba(46,125,50,0.2) !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(76,175,80,0.35) !important;
    background: linear-gradient(135deg, #388E3C 0%, #2E7D32 100%) !important;
}

/* ── Upload zone ── */
div[data-testid="stFileUploader"] {
    background: #0E1A10 !important;
    border: 1.5px dashed rgba(76,175,80,0.3) !important;
    border-radius: 10px !important;
    padding: 16px !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] iframe {
    border-radius: 8px;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #1B5E20, #4CAF50) !important;
}

/* ── Info / Warning ── */
div[data-testid="stAlert"] {
    background: #0E1A10 !important;
    border: 1px solid #1E3A22 !important;
    border-radius: 8px !important;
    color: #7FAF86 !important;
}

/* ── Sidebar nav labels ── */
.nav-section {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #3D6B42 !important;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 6px;
    padding-left: 2px;
}

/* ── Crop badge chips ── */
.crop-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.crop-chip {
    background: rgba(76,175,80,0.1);
    border: 1px solid rgba(76,175,80,0.2);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #7FAF86;
    font-weight: 600;
}

/* ── Divider ── */
hr { border-color: #1A3020 !important; }

/* ── Block container ── */
.block-container { padding-top: 1.8rem !important; padding-bottom: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constantes ───────────────────────────────────────────────────────────────
CULTURAS_INFO = {
    'Milho':  {'icon': '🌽', 'en': 'corn',     'color': '#FFB300'},
    'Soja':   {'icon': '🫘', 'en': 'soybean',  'color': '#8D6E63'},
    'Arroz':  {'icon': '🌾', 'en': 'rice',     'color': '#A5D6A7'},
    'Trigo':  {'icon': '🌾', 'en': 'wheat',    'color': '#D4A017'},
    'Feijao': {'icon': '🫘', 'en': 'bean',     'color': '#E57373'},
}

# ─── Loaders ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return (joblib.load('seed_classifier_model.pkl'),
                joblib.load('model_features.pkl'),
                joblib.load('model_classes.pkl'))
    except FileNotFoundError:
        st.error("❌ Execute `python train_model.py` para gerar o modelo antes de continuar.")
        st.stop()

@st.cache_resource
def load_clip():
    with st.spinner("Carregando módulo de visão computacional..."):
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return m, p

@st.cache_data
def load_reference_data():
    try:
        return pd.read_csv('seeds_dataset.csv')
    except FileNotFoundError:
        return None

# ─── Inicialização do Session State para Persistência de Predições ──────────────
if 'cv_pred' not in st.session_state:
    st.session_state.cv_pred = None
if 'cv_conf' not in st.session_state:
    st.session_state.cv_conf = None
if 'cv_probs' not in st.session_state:
    st.session_state.cv_probs = None
if 'ml_pred' not in st.session_state:
    st.session_state.ml_pred = None
if 'ml_conf' not in st.session_state:
    st.session_state.ml_conf = None
if 'ml_probs' not in st.session_state:
    st.session_state.ml_probs = None

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Painel de Informações Corporativas
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px 0; border-bottom: 1px solid #1E3A22; margin-bottom: 4px;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:700;
                    background:linear-gradient(135deg,#6FCF7A,#4CAF50); -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;">🌱 MatoScan</div>
        <div style="font-size:0.68rem; color:#3D6B42; text-transform:uppercase; letter-spacing:1.5px; margin-top:2px;">
            Agro Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Culturas Suportadas</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="crop-chips">
        <span class="crop-chip">🌽 Milho</span>
        <span class="crop-chip">🫘 Soja</span>
        <span class="crop-chip">🌾 Arroz</span>
        <span class="crop-chip">🌾 Trigo</span>
        <span class="crop-chip">🫘 Feijão</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:18px 0'>", unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Instruções de Diagnóstico</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#6A9E70; line-height:1.45; padding-left: 2px;">
        1. 📸 Envie uma imagem na <strong>Análise Visual</strong> para predição por semelhança semântica.<br><br>
        2. 📐 Ajuste os seletores na <strong>Análise Morfológica</strong> para classificação laboratorial.<br><br>
        3. 🤝 Acompanhe a <strong>Fusão de Sensores</strong> no painel unificado central.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:18px 0'>", unsafe_allow_html=True)
    
    if st.button("🗑️ LIMPAR DIAGNÓSTICOS"):
        st.session_state.cv_pred = None
        st.session_state.cv_conf = None
        st.session_state.cv_probs = None
        st.session_state.ml_pred = None
        st.session_state.ml_conf = None
        st.session_state.ml_probs = None
        st.rerun()

    st.markdown("""
    <div style="margin-top:auto; padding-top:20px; border-top:1px solid #1A3020;">
        <div style="font-size:0.7rem; color:#3D6B42; line-height:1.6;">
            <div style="font-weight:700; color:#4A7C50; margin-bottom:4px;">v1.0.0 — MatoScan</div>
            Modelo ML: Random Forest (99.4%)<br>
            Modelo CV: CLIP ViT-B/32<br>
            Dataset Ref: 1.750 amostras
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="agro-header">
    <div>
        <div class="agro-logo">🌱 MatoScan</div>
        <div class="agro-tagline">Plataforma Agropecuária Integrada de Diagnóstico Botânico</div>
    </div>
    <div style="margin-left:32px; display:flex; gap:24px; align-items:center; flex-wrap:wrap;">
        <div class="agro-status">
            <div class="status-dot"></div>
            Fusão de Dados Ativa
        </div>
        <div style="font-size:0.78rem; color:#3D6B42;">|</div>
        <div style="font-size:0.78rem; color:#5A8A60;">🤖 ML: Random Forest</div>
        <div style="font-size:0.78rem; color:#3D6B42;">|</div>
        <div style="font-size:0.78rem; color:#5A8A60;">👁️ CV: CLIP ViT-B/32</div>
    </div>
    <div class="agro-version">v1.0.0</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS — Monitor do Sistema
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Culturas Suportadas</div>
        <div class="kpi-value">5</div>
        <div class="kpi-sub">Milho · Soja · Arroz · Trigo · Feijão</div>
    </div>
    <div class="kpi-card gold">
        <div class="kpi-label">Precisão Estatística (ML)</div>
        <div class="kpi-value">99.4%</div>
        <div class="kpi-sub">Random Forest treinado localmente</div>
    </div>
    <div class="kpi-card blue">
        <div class="kpi-label">Amostras Morfológicas</div>
        <div class="kpi-value">1.750</div>
        <div class="kpi-sub">350 amostras balanceadas por grão</div>
    </div>
    <div class="kpi-card orange">
        <div class="kpi-label">Variáveis de Mapeamento</div>
        <div class="kpi-value">8</div>
        <div class="kpi-sub">Dimensões físicas + métricas derivadas</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAINEL CENTRAL DE ENTRADA — DIAGNÓSTICO EM COLUNAS SIMULTÂNEAS
# ══════════════════════════════════════════════════════════════════════════════
col_esq, col_dir = st.columns([1, 1.05], gap="large")

# ─── Coluna Esquerda: Diagnóstico por Foto (Visão Computacional) ─────────────
with col_esq:
    st.markdown('<div class="section-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📸 Análise por Imagem (Visão Computacional)</div>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Arraste e solte uma imagem da planta ou dos grãos (PNG, JPG, JPEG):",
                                type=["png", "jpg", "jpeg"], key="cv_file_uploader")
    
    # Se a foto foi removida do seletor, reseta a predição local
    if not uploaded:
        st.session_state.cv_pred = None
        st.session_state.cv_conf = None
        st.session_state.cv_probs = None
        
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Imagem sob inspeção óptica", use_container_width=True)
        cv_btn = st.button("🔍 ANALISAR IMAGEM", key="btn_run_cv")
        
        if cv_btn:
            clip_model, clip_proc = load_clip()
            culturas_cv = ['Milho','Soja','Arroz','Trigo','Feijao']
            prompts_cv  = [
                "a close-up photo of corn plant or corn grain seeds",
                "a close-up photo of soybean plant or soybean seeds",
                "a close-up photo of rice plant or rice grains",
                "a close-up photo of wheat plant or wheat grains",
                "a close-up photo of bean plant or bean seeds",
            ]
            
            with st.status("⚙️ Processando imagem via CLIP...", expanded=True) as s:
                st.write("Processando imagem e redimensionando para a rede neural...")
                time.sleep(0.3)
                st.write("Calculando vetores de similaridade semântica...")
                time.sleep(0.3)
                st.write("Finalizando probabilidade ponderada...")
                
                inputs = clip_proc(text=prompts_cv, images=img,
                                   return_tensors="pt", padding=True)
                with torch.no_grad():
                    out = clip_model(**inputs)
                probs = out.logits_per_image.softmax(dim=1).numpy()[0]
                s.update(label="✅ Módulo visual concluído", state="complete", expanded=False)
                
            pred_idx = int(np.argmax(probs))
            st.session_state.cv_pred = culturas_cv[pred_idx]
            st.session_state.cv_conf = probs[pred_idx]
            st.session_state.cv_probs = probs
            
        if st.session_state.cv_pred is not None:
            pred = st.session_state.cv_pred
            conf = st.session_state.cv_conf
            probs = st.session_state.cv_probs
            culturas_cv = ['Milho','Soja','Arroz','Trigo','Feijao']
            pred_idx = culturas_cv.index(pred)
            
            info = CULTURAS_INFO.get(pred, {'icon': '🌱'})
            nome_exib = 'Feijão' if pred == 'Feijao' else pred
            conf_pct = int(conf * 100)
            
            st.markdown(f"""
            <div class="result-card" style="border-left-color: #29B6F6; border-color: rgba(41, 182, 246, 0.3); background: linear-gradient(135deg, #09212D 0%, #06151E 100%);">
                <div class="result-badge" style="color: #29B6F6; background: rgba(41,182,246,0.12); border-color: rgba(41,182,246,0.3);">✓ Visão Computacional</div>
                <div class="result-culture" style="color: #81D4FA;">{info['icon']} {nome_exib}</div>
                <div class="result-desc" style="color: #5A7E8D;">Mapeamento contextual zero-shot via rede CLIP ViT-B/32.</div>
                <div class="confidence-bar" style="background: #142630;">
                    <div class="confidence-fill" style="width:{conf_pct}%; background: linear-gradient(90deg, #0288D1, #29B6F6);"></div>
                </div>
                <div style="font-size:0.78rem; color:#5A7E8D; margin-top:5px; font-family:'JetBrains Mono',monospace;">
                    Confiança Visual: {conf:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Plot de probabilidades da Imagem
            fig_cv, ax_cv = plt.subplots(figsize=(5.5, 2.2))
            nomes_pt = ['Feijão' if c == 'Feijao' else c for c in culturas_cv]
            bar_clrs = ['#29B6F6' if i == pred_idx else '#142630' for i in range(len(culturas_cv))]
            bars = ax_cv.barh(nomes_pt, probs, color=bar_clrs, height=0.55, edgecolor='#0E1A10', linewidth=0.5)
            ax_cv.spines['left'].set_color('#1E3A22')
            ax_cv.tick_params(colors='#7FAF86', labelsize=8.5)
            ax_cv.xaxis.set_visible(False)
            for bar in bars:
                w = bar.get_width()
                ax_cv.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{w:.1%}', va='center', ha='left',
                        color='#81D4FA', fontsize=8,
                        fontfamily='monospace', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_cv, use_container_width=True)
    else:
        st.markdown("""
        <div style="background:#09161B; border:1px dashed #1E3E4B; border-radius:8px;
                    padding:50px 10px; text-align:center; color:#5A7E8D; font-size:0.9rem; margin-top:10px;">
            📸 Faça upload de uma foto do grão ou da planta para<br>
            iniciar a predição por redes neurais contextuais.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Coluna Direita: Diagnóstico por Medições (Morfometria de Suporte) ───────
with col_dir:
    st.markdown('<div class="section-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📐 Análise Morfológica (Características Físicas)</div>', unsafe_allow_html=True)
    
    # Sliders interativos na tela
    comprimento_mm = st.slider('Comprimento da Semente (mm)', 1.0, 25.0, 8.0, 0.1, key="comp_slider_main")
    largura_mm     = st.slider('Largura da Semente (mm)',      1.0, 15.0, 6.0, 0.1, key="larg_slider_main")
    espessura_mm   = st.slider('Espessura da Semente (mm)',    0.5, 12.0, 4.0, 0.1, key="esp_slider_main")
    massa_mg       = st.slider('Massa estimada (mg)',          5.0, 600.0, 200.0, 1.0, key="massa_slider_main")
    
    # Cálculo das características derivadas
    area_mm2      = comprimento_mm * largura_mm * np.pi / 4
    perimetro_mm  = np.pi * (3*(comprimento_mm/2 + largura_mm/2)
                              - np.sqrt((3*comprimento_mm/2 + largura_mm/2)
                                        * (comprimento_mm/2 + 3*largura_mm/2)))
    compacidade   = (4 * np.pi * area_mm2) / (perimetro_mm ** 2)
    razao_aspecto = comprimento_mm / largura_mm
    
    model, model_features, model_classes = load_model()
    
    user_input = pd.DataFrame([[
        comprimento_mm, largura_mm, espessura_mm, massa_mg,
        area_mm2, perimetro_mm, compacidade, razao_aspecto
    ]], columns=model_features)
    
    # Tabela dinâmica de variáveis morfométricas calculadas
    st.markdown(f"""
    <table class="measure-table">
        <tr><td>Área Projetada (Elipse)</td><td>{area_mm2:.2f} mm²</td></tr>
        <tr><td>Perímetro Estimado (Ramanujan)</td><td>{perimetro_mm:.2f} mm</td></tr>
        <tr><td>Compacidade (Circularidade)</td><td>{compacidade:.3f}</td></tr>
        <tr><td>Razão de Aspecto (Alongamento)</td><td>{razao_aspecto:.2f}×</td></tr>
    </table>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 ANALISAR MEDIÇÕES", key="btn_run_ml")
    
    if analyze_btn:
        try:
            pred = model.predict(user_input)[0]
            proba = model.predict_proba(user_input)[0]
            
            st.session_state.ml_pred = pred
            st.session_state.ml_conf = proba[list(model_classes).index(pred)]
            st.session_state.ml_probs = proba
        except Exception as e:
            st.error(f"Erro na inferência morfométrica: {e}")
            
    if st.session_state.ml_pred is not None:
        pred = st.session_state.ml_pred
        conf = st.session_state.ml_conf
        proba = st.session_state.ml_probs
        
        info = CULTURAS_INFO.get(pred, {'icon': '🌱'})
        nome_exib = 'Feijão' if pred == 'Feijao' else pred
        conf_pct = int(conf * 100)
        
        st.markdown(f"""
        <div class="result-card">
            <div class="result-badge">✓ Machine Learning</div>
            <div class="result-culture">{info['icon']} {nome_exib}</div>
            <div class="result-desc">Predição via RandomForest Classifier (200 estimadores) com dados de laboratório.</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width:{conf_pct}%"></div>
            </div>
            <div style="font-size:0.78rem; color:#5A8A60; margin-top:5px; font-family:'JetBrains Mono',monospace;">
                Confiança Física: {conf:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabela ordenada das probabilidades do classificador RF
        st.markdown('<div class="section-divider" style="margin: 10px 0;"></div>', unsafe_allow_html=True)
        classes_pt = ['Feijão' if c == 'Feijao' else c for c in model_classes]
        proba_df = pd.DataFrame({'Cultura': classes_pt, 'Confiança (%)': [f"{p*100:.1f}%" for p in proba]})
        proba_df = proba_df.sort_values('Confiança (%)', ascending=False)
        st.dataframe(proba_df, hide_index=True, use_container_width=True)
    else:
        st.markdown("""
        <div style="background:#0E1A10; border:1px dashed #1E3A22; border-radius:8px;
                    padding:40px 10px; text-align:center; color:#3D6B42; font-size:0.9rem; margin-top:10px;">
            📐 Mova os controles deslizantes acima e selecione <br>
            <strong style="color:#4CAF50;">ANALISAR MEDIÇÕES</strong> para ver o resultado do classificador físico.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FUSÃO DE DADOS — DIAGNÓSTICO CONSOLIDADO E CONVERGÊNCIA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-card" style="margin-top: 16px;">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🤝 Diagnóstico Consolidado (Fusão de Sensores)</div>', unsafe_allow_html=True)

cv_pred = st.session_state.cv_pred
ml_pred = st.session_state.ml_pred

if cv_pred is not None and ml_pred is not None:
    if cv_pred == ml_pred:
        nome_cultura = 'Feijão' if cv_pred == 'Feijao' else cv_pred
        info = CULTURAS_INFO.get(cv_pred, {'icon': '🌱'})
        st.markdown(f"""
        <div style="background: rgba(76, 175, 80, 0.07); border: 1.5px solid rgba(76, 175, 80, 0.35); border-radius: 8px; padding: 18px 24px; display: flex; align-items: center; gap: 20px;">
            <div style="font-size: 3.5rem; line-height: 1;">🏆</div>
            <div>
                <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: #4CAF50; font-weight: 700; margin-bottom: 2px;">✓ Diagnóstico Unificado Confirmado</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #E8F5E4;">
                    {info['icon']} Cultura Identificada: {nome_cultura}
                </div>
                <div style="font-size: 0.88rem; color: #7FAF86; margin-top: 4px; line-height: 1.45;">
                    Excelente! Ambas as metodologias do MatoScan divergiram para a mesma classificação. A rede neural óptica (CLIP) 
                    e o modelo matemático morfológico (Random Forest) concluíram em concordância que a amostra trata-se de **{nome_cultura}**.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        nome_cv = 'Feijão' if cv_pred == 'Feijao' else cv_pred
        nome_ml = 'Feijão' if ml_pred == 'Feijao' else ml_pred
        info_cv = CULTURAS_INFO.get(cv_pred, {'icon': '🌱'})
        info_ml = CULTURAS_INFO.get(ml_pred, {'icon': '🌱'})
        st.markdown(f"""
        <div style="background: rgba(255, 143, 0, 0.08); border: 1.5px solid rgba(255, 143, 0, 0.4); border-radius: 8px; padding: 18px 24px; display: flex; align-items: center; gap: 20px;">
            <div style="font-size: 3.5rem; line-height: 1;">⚠️</div>
            <div>
                <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: #FF8F00; font-weight: 700; margin-bottom: 2px;">⚠ Alerta de Divergência de Sinais</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; color: #FFE082;">
                    Divergência entre Visão Óptica e Medições Físicas
                </div>
                <div style="font-size: 0.88rem; color: #B39D82; margin-top: 6px; line-height: 1.5;">
                    • A **Análise Óptica (Imagem)** diagnosticou a espécie como <strong>{info_cv['icon']} {nome_cv}</strong>.<br>
                    • A **Análise Geométrica (Sliders)** diagnosticou a semente como <strong>{info_ml['icon']} {nome_ml}</strong>.<br>
                    <span style="color: #FFB74D; font-weight: 500;">Sugestão:</span> Verifique se os sliders de tamanho/massa refletem com exatidão a semente analisada ou se a iluminação da imagem enviada não está distorcendo a cor/contorno do grão.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
elif cv_pred is not None:
    nome_cv = 'Feijão' if cv_pred == 'Feijao' else cv_pred
    info = CULTURAS_INFO.get(cv_pred, {'icon': '🌱'})
    st.markdown(f"""
    <div style="background: rgba(41, 182, 246, 0.06); border: 1px dashed rgba(41, 182, 246, 0.35); border-radius: 8px; padding: 14px 18px; display: flex; align-items: center; gap: 15px;">
        <div style="font-size: 2rem;">📸</div>
        <div>
            <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; color: #29B6F6; font-weight: 600;">Diagnóstico por Imagem Ativo</div>
            <div style="font-size: 0.95rem; color: #E8F5E4;">A imagem sob análise possui maior afinidade visual com: <strong>{info['icon']} {nome_cv}</strong>.</div>
            <div style="font-size: 0.78rem; color: #5A7E8D; margin-top: 2px;">Dica: Você pode calibrar os sliders à direita para conferir se as propriedades geométricas da semente conferem com as métricas padrão de laboratório.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
elif ml_pred is not None:
    nome_ml = 'Feijão' if ml_pred == 'Feijao' else ml_pred
    info = CULTURAS_INFO.get(ml_pred, {'icon': '🌱'})
    st.markdown(f"""
    <div style="background: rgba(76, 175, 80, 0.06); border: 1px dashed rgba(76, 175, 80, 0.35); border-radius: 8px; padding: 14px 18px; display: flex; align-items: center; gap: 15px;">
        <div style="font-size: 2rem;">📐</div>
        <div>
            <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; color: #4CAF50; font-weight: 600;">Diagnóstico por Medições Físicas Ativo</div>
            <div style="font-size: 0.95rem; color: #E8F5E4;">Os parâmetros geométricos classificaram a amostra como: <strong>{info['icon']} {nome_ml}</strong>.</div>
            <div style="font-size: 0.78rem; color: #6A9E70; margin-top: 2px;">Dica: Faça o envio de uma foto do lote na coluna da esquerda para validar o diagnóstico com a análise ótica do CLIP.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center; padding:24px; color:#3D6B42; font-size:0.88rem; border:1px dashed #1D3A22; border-radius:8px; background:#08100A;">
        🍃 <strong>Sensores em Espera</strong> — Aguardando inferências. Execute a predição visual ou a física nos blocos acima para obter o relatório analítico consolidado.
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO INFERIOR — GRÁFICOS E ANÁLISE CIENTÍFICA
# ══════════════════════════════════════════════════════════════════════════════
seeds_df = load_reference_data()
if seeds_df is not None:
    st.markdown('<div class="section-card" style="margin-top: 16px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Análise de Proximidade e Distribuições Científicas</div>', unsafe_allow_html=True)
    
    col_g1, col_g2 = st.columns([1, 1.02], gap="large")
    
    PALETTE = {
        'Milho':'#FFB300','Soja':'#A1887F',
        'Arroz':'#A5D6A7','Trigo':'#D4A017','Feijao':'#EF9A9A'
    }
    LABELS_PT = {'Milho':'Milho','Soja':'Soja','Arroz':'Arroz','Trigo':'Trigo','Feijao':'Feijão'}
    
    with col_g1:
        # Scatter Plot: Comprimento x Largura
        fig1, ax1 = plt.subplots(figsize=(6.5, 3.8))
        for cultura, grupo in seeds_df.groupby('cultura'):
            ax1.scatter(grupo['comprimento_mm'], grupo['largura_mm'],
                        color=PALETTE.get(cultura, '#888'), alpha=0.35,
                        s=18, label=LABELS_PT.get(cultura, cultura),
                        edgecolors='none')
        # Destacar amostra configurada nos sliders em tempo real
        ax1.scatter(comprimento_mm, largura_mm, color='#FFFFFF',
                    edgecolor='#4CAF50', s=180, marker='D', zorder=8,
                    linewidths=1.5, label='Amostra Atual')
        ax1.set_title('Distribuição Morfológica — Comprimento × Largura',
                      fontsize=10, fontweight='bold', color='#A5D6A7',
                      pad=10, family='Space Grotesk')
        ax1.set_xlabel('Comprimento (mm)', fontsize=8.5)
        ax1.set_ylabel('Largura (mm)', fontsize=8.5)
        leg = ax1.legend(frameon=True, facecolor='#0A1A0D', edgecolor='#1E3A22',
                         framealpha=0.95, fontsize=7.5, ncol=2)
        for t in leg.get_texts(): t.set_color('#8FAF86')
        ax1.grid(True, linestyle='--', alpha=0.12)
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        
    with col_g2:
        # Boxplot de Massa & Histograma/Barra de Compacidade
        fig2, axes = plt.subplots(1, 2, figsize=(6.5, 3.8))
        
        # Boxplot de distribuição de massa
        data_box = [seeds_df[seeds_df['cultura']==c]['massa_mg'].values
                    for c in ['Milho','Soja','Arroz','Trigo','Feijao']]
        labels_box = ['Milho','Soja','Arroz','Trigo','Feijão']
        bp = axes[0].boxplot(data_box, patch_artist=True, notch=False,
                             medianprops={'color':'#4CAF50','linewidth':2})
        colors_box = ['#FFB300','#A1887F','#A5D6A7','#D4A017','#EF9A9A']
        for patch, c in zip(bp['boxes'], colors_box):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        for whisker in bp['whiskers']: whisker.set(color='#3D5A3E', linewidth=1)
        for cap in bp['caps']: cap.set(color='#3D5A3E', linewidth=1)
        axes[0].axhline(massa_mg, color='#FFFFFF', linestyle='--',
                        linewidth=1.2, alpha=0.7, label='Amostra')
        axes[0].set_xticklabels(labels_box, fontsize=7.5)
        axes[0].set_ylabel('Massa (mg)', fontsize=8)
        axes[0].set_title('Massa por Cultura', fontsize=9,
                           color='#A5D6A7', fontweight='bold', family='Space Grotesk')
        axes[0].grid(True, linestyle='--', alpha=0.1, axis='y')
        
        # Bar chart de Compacidade
        comp_means = seeds_df.groupby('cultura')['compacidade'].mean()
        c_labels = ['Milho','Soja','Arroz','Trigo','Feijão']
        c_keys = ['Milho','Soja','Arroz','Trigo','Feijao']
        c_vals = [comp_means.get(k, 0) for k in c_keys]
        bar_colors = ['#FFB30088','#A1887F88','#A5D6A788','#D4A01788','#EF9A9A88']
        axes[1].bar(c_labels, c_vals, color=bar_colors, edgecolor='#2d5a2d', linewidth=0.8)
        axes[1].axhline(compacidade, color='#FFFFFF', linestyle='--', linewidth=1.2, alpha=0.7)
        axes[1].set_ylabel('Compacidade média', fontsize=8)
        axes[1].set_title('Compacidade por Cultura', fontsize=9,
                           color='#A5D6A7', fontweight='bold', family='Space Grotesk')
        axes[1].set_xticklabels(c_labels, fontsize=7.5)
        axes[1].grid(True, linestyle='--', alpha=0.1, axis='y')
        
        plt.tight_layout(pad=1.5)
        st.pyplot(fig2, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
