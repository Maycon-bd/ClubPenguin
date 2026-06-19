# app.py — MatoScan | Plataforma de Identificação de Culturas
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
[data-testid="stSidebar"] .stRadio > label { text-transform: none !important; font-size: 0.95rem !important; color: #C8DCC0 !important; }

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
    font-size: 0.78rem;
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
.measure-table .derived td { background: rgba(76,175,80,0.04); }

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

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Painel de Controle
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo sidebar
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

    st.markdown('<div class="nav-section">Módulo de Análise</div>', unsafe_allow_html=True)
    modo = st.radio(
        "",
        ["⚙️ Morfometria da Semente", "📸 Visão Computacional"],
        label_visibility="collapsed"
    )

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

    st.markdown("<hr style='margin:20px 0'>", unsafe_allow_html=True)

    if modo == "⚙️ Morfometria da Semente":
        st.markdown('<div class="nav-section">Parâmetros da Amostra</div>', unsafe_allow_html=True)
        comprimento_mm = st.slider('Comprimento (mm)', 1.0, 25.0, 8.0, 0.1)
        largura_mm     = st.slider('Largura (mm)',      1.0, 15.0, 6.0, 0.1)
        espessura_mm   = st.slider('Espessura (mm)',    0.5, 12.0, 4.0, 0.1)
        massa_mg       = st.slider('Massa (mg)',        5.0, 600.0, 200.0, 1.0)
        st.markdown("<hr style='margin:14px 0'>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 ANALISAR AMOSTRA")

    st.markdown("""
    <div style="margin-top:auto; padding-top:20px; border-top:1px solid #1A3020;">
        <div style="font-size:0.7rem; color:#3D6B42; line-height:1.6;">
            <div style="font-weight:700; color:#4A7C50; margin-bottom:4px;">v1.0.0 — MatoScan</div>
            Modelo: Random Forest<br>
            Acurácia: 99.4%<br>
            Amostras: 1.750
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
        <div class="agro-tagline">Plataforma de Identificação de Culturas por IA</div>
    </div>
    <div style="margin-left:32px; display:flex; gap:24px; align-items:center; flex-wrap:wrap;">
        <div class="agro-status">
            <div class="status-dot"></div>
            Sistema Online
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
# KPI CARDS — Métricas do Sistema
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Culturas Identificadas</div>
        <div class="kpi-value">5</div>
        <div class="kpi-sub">Milho · Soja · Arroz · Trigo · Feijão</div>
    </div>
    <div class="kpi-card gold">
        <div class="kpi-label">Acurácia do Modelo</div>
        <div class="kpi-value">99.4%</div>
        <div class="kpi-sub">↑ Random Forest · 200 árvores</div>
    </div>
    <div class="kpi-card blue">
        <div class="kpi-label">Amostras de Treino</div>
        <div class="kpi-value">1.750</div>
        <div class="kpi-sub">350 por cultura · Dataset morfológico</div>
    </div>
    <div class="kpi-card orange">
        <div class="kpi-label">Features Analisadas</div>
        <div class="kpi-value">8</div>
        <div class="kpi-sub">Morfometria + features derivadas</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODO 1 — MORFOMETRIA
# ══════════════════════════════════════════════════════════════════════════════
if modo == "⚙️ Morfometria da Semente":
    model, model_features, model_classes = load_model()
    seeds_df = load_reference_data()

    # Calcular features derivadas
    area_mm2      = comprimento_mm * largura_mm * np.pi / 4
    perimetro_mm  = np.pi * (3*(comprimento_mm/2 + largura_mm/2)
                              - np.sqrt((3*comprimento_mm/2 + largura_mm/2)
                                        * (comprimento_mm/2 + 3*largura_mm/2)))
    compacidade   = (4 * np.pi * area_mm2) / (perimetro_mm ** 2)
    razao_aspecto = comprimento_mm / largura_mm

    user_input = pd.DataFrame([[
        comprimento_mm, largura_mm, espessura_mm, massa_mg,
        area_mm2, perimetro_mm, compacidade, razao_aspecto
    ]], columns=model_features)

    col_a, col_b = st.columns([1, 1.15], gap="large")

    # ── Coluna Esquerda ──
    with col_a:
        # Painel de medições
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📐 Medições da Amostra</div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <table class="measure-table">
            <tr><td>Comprimento</td><td>{comprimento_mm:.1f} mm</td></tr>
            <tr><td>Largura</td><td>{largura_mm:.1f} mm</td></tr>
            <tr><td>Espessura</td><td>{espessura_mm:.1f} mm</td></tr>
            <tr><td>Massa</td><td>{massa_mg:.0f} mg</td></tr>
        </table>
        <hr class="section-divider">
        <div class="section-title" style="margin-top:10px;">⚡ Parâmetros Derivados</div>
        <table class="measure-table">
            <tr class="derived"><td>Área Projetada</td><td>{area_mm2:.2f} mm²</td></tr>
            <tr class="derived"><td>Perímetro</td><td>{perimetro_mm:.2f} mm</td></tr>
            <tr class="derived"><td>Compacidade</td><td>{compacidade:.3f}</td></tr>
            <tr class="derived"><td>Razão de Aspecto</td><td>{razao_aspecto:.2f}×</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

        # Resultado
        if analyze_btn or 'done' in st.session_state:
            st.session_state.done = True
            try:
                pred      = model.predict(user_input)[0]
                proba     = model.predict_proba(user_input)[0]
                conf      = proba[list(model_classes).index(pred)]
                info      = CULTURAS_INFO.get(pred, {'icon':'🌱','color':'#4CAF50'})
                nome_exib = 'Feijão' if pred == 'Feijao' else pred
                conf_pct  = int(conf * 100)

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-badge">✓ Cultura Identificada</div>
                    <div class="result-culture">{info['icon']} {nome_exib}</div>
                    <div class="result-desc">Classificação baseada em morfometria da semente via Random Forest.</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{conf_pct}%"></div>
                    </div>
                    <div style="font-size:0.78rem; color:#5A8A60; margin-top:5px; font-family:'JetBrains Mono',monospace;">
                        Confiança: {conf:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Tabela de probabilidades
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📊 Matriz de Probabilidades</div>', unsafe_allow_html=True)
                classes_pt  = ['Feijão' if c == 'Feijao' else c for c in model_classes]
                proba_df    = pd.DataFrame({'Cultura': classes_pt, 'Confiança (%)': [f"{p*100:.1f}%" for p in proba]})
                proba_df    = proba_df.sort_values('Confiança (%)', ascending=False)
                st.dataframe(proba_df, hide_index=True, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Erro na análise: {e}")
        else:
            st.markdown("""
            <div style="background:#0A1A0D; border:1px dashed #1E3A22; border-radius:8px;
                        padding:20px; text-align:center; color:#3D6B42; font-size:0.9rem;">
                ⬅️ Ajuste os parâmetros no painel lateral e clique em<br>
                <strong style="color:#4CAF50;">ANALISAR AMOSTRA</strong>
            </div>
            """, unsafe_allow_html=True)

    # ── Coluna Direita — Visualizações ──
    with col_b:
        if seeds_df is not None:
            PALETTE = {
                'Milho':'#FFB300','Soja':'#A1887F',
                'Arroz':'#A5D6A7','Trigo':'#D4A017','Feijao':'#EF9A9A'
            }
            LABELS_PT = {'Milho':'Milho','Soja':'Soja','Arroz':'Arroz','Trigo':'Trigo','Feijao':'Feijão'}

            # ── Scatter: Comprimento × Largura ──
            fig1, ax1 = plt.subplots(figsize=(6.5, 3.8))
            for cultura, grupo in seeds_df.groupby('cultura'):
                ax1.scatter(grupo['comprimento_mm'], grupo['largura_mm'],
                            color=PALETTE.get(cultura, '#888'), alpha=0.35,
                            s=18, label=LABELS_PT.get(cultura, cultura),
                            edgecolors='none')
            ax1.scatter(comprimento_mm, largura_mm, color='#FFFFFF',
                        edgecolor='#4CAF50', s=180, marker='D', zorder=8,
                        linewidths=1.5, label='Amostra')
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

            # ── Radar / Boxplot por Razão de Aspecto ──
            fig2, axes = plt.subplots(1, 2, figsize=(6.5, 3.4))

            # Box: massa
            data_box  = [seeds_df[seeds_df['cultura']==c]['massa_mg'].values
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
                            linewidth=1.2, alpha=0.7, label=f'Amostra')
            axes[0].set_xticklabels(labels_box, fontsize=7.5)
            axes[0].set_ylabel('Massa (mg)', fontsize=8)
            axes[0].set_title('Massa por Cultura', fontsize=9,
                               color='#A5D6A7', fontweight='bold', family='Space Grotesk')
            axes[0].grid(True, linestyle='--', alpha=0.1, axis='y')

            # Bar: compacidade média
            comp_means = seeds_df.groupby('cultura')['compacidade'].mean()
            c_labels   = ['Milho','Soja','Arroz','Trigo','Feijão']
            c_keys     = ['Milho','Soja','Arroz','Trigo','Feijao']
            c_vals     = [comp_means.get(k, 0) for k in c_keys]
            bar_colors = ['#FFB30088','#A1887F88','#A5D6A788','#D4A01788','#EF9A9A88']
            bars = axes[1].bar(c_labels, c_vals, color=bar_colors,
                               edgecolor='#2d5a2d', linewidth=0.8)
            axes[1].axhline(compacidade, color='#FFFFFF', linestyle='--',
                            linewidth=1.2, alpha=0.7)
            axes[1].set_ylabel('Compacidade média', fontsize=8)
            axes[1].set_title('Compacidade por Cultura', fontsize=9,
                               color='#A5D6A7', fontweight='bold', family='Space Grotesk')
            axes[1].set_xticklabels(c_labels, fontsize=7.5)
            axes[1].grid(True, linestyle='--', alpha=0.1, axis='y')

            plt.tight_layout(pad=1.5)
            st.pyplot(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODO 2 — VISÃO COMPUTACIONAL (CLIP)
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div style="background:#0A1A0D; border:1px solid #1E3A22; border-left:4px solid #29B6F6;
                border-radius:8px; padding:12px 18px; margin-bottom:18px; font-size:0.88rem; color:#7FAFB0;">
        📡 <strong style="color:#81D4FA;">Modo Visão Computacional</strong> — Faça upload de uma foto
        da planta ou grão. O sistema CLIP analisa localmente sem enviar dados para servidores externos.
    </div>
    """, unsafe_allow_html=True)

    col_up, col_res = st.columns([1, 1.05], gap="large")

    with col_up:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📂 Entrada de Imagem</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Arraste uma foto da planta ou grão (PNG, JPG):",
                                    type=["png","jpg","jpeg"])
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Imagem carregada", use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            cv_btn = st.button("🔍 IDENTIFICAR CULTURA")
        else:
            st.markdown("""
            <div style="text-align:center; padding:30px 10px; color:#3D6B42; font-size:0.88rem;">
                Nenhuma imagem carregada.<br>
                <span style="font-size:0.8rem;">Suporte: JPG, PNG · Max 200 MB</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔬 Resultado da Análise Visual</div>', unsafe_allow_html=True)

        if uploaded and (cv_btn or 'cv_done' in st.session_state):
            st.session_state.cv_done = True
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
                st.write("Pré-processando imagem...")
                time.sleep(0.4)
                st.write("Extraindo embeddings visuais...")
                time.sleep(0.4)
                st.write("Computando similaridade semântica...")

                inputs = clip_proc(text=prompts_cv, images=img,
                                   return_tensors="pt", padding=True)
                with torch.no_grad():
                    out = clip_model(**inputs)
                probs = out.logits_per_image.softmax(dim=1).numpy()[0]
                s.update(label="✅ Análise concluída", state="complete", expanded=False)

            pred_idx  = int(np.argmax(probs))
            pred      = culturas_cv[pred_idx]
            conf      = probs[pred_idx]
            info      = CULTURAS_INFO.get(pred, {'icon':'🌱'})
            nome_exib = 'Feijão' if pred == 'Feijao' else pred
            conf_pct  = int(conf * 100)

            st.markdown(f"""
            <div class="result-card">
                <div class="result-badge">✓ Identificado por CV</div>
                <div class="result-culture">{info['icon']} {nome_exib}</div>
                <div class="result-desc">Classificação por similaridade semântica entre imagem e descrição da cultura (CLIP zero-shot).</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{conf_pct}%"></div>
                </div>
                <div style="font-size:0.78rem; color:#5A8A60; margin-top:5px; font-family:'JetBrains Mono',monospace;">
                    Similaridade: {conf:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gráfico de probabilidades
            fig, ax = plt.subplots(figsize=(5.5, 2.8))
            nomes_pt = ['Feijão' if c == 'Feijao' else c for c in culturas_cv]
            bar_clrs = ['#4CAF50' if i == pred_idx else '#1E3A22' for i in range(len(culturas_cv))]
            bars = ax.barh(nomes_pt, probs, color=bar_clrs, height=0.55,
                           edgecolor='#0E1A10', linewidth=0.5)
            ax.spines['left'].set_color('#1E3A22')
            ax.tick_params(colors='#7FAF86', labelsize=9)
            ax.xaxis.set_visible(False)
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{w:.1%}', va='center', ha='left',
                        color='#A5D6A7', fontsize=8.5,
                        fontfamily='monospace', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        else:
            st.markdown("""
            <div style="text-align:center; padding:50px 10px; color:#3D6B42; font-size:0.88rem;">
                Carregue uma imagem e clique em<br>
                <strong style="color:#4CAF50;">IDENTIFICAR CULTURA</strong>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
