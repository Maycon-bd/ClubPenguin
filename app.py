# app.py — MatoScan
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
    page_title="MatoScan — Identificador de Culturas",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Tema matplotlib AgroTech ─────────────────────────────────────────────────
plt.rcParams['text.color']        = '#d4ecd4'
plt.rcParams['axes.labelcolor']   = '#d4ecd4'
plt.rcParams['xtick.color']       = '#7bab7b'
plt.rcParams['ytick.color']       = '#7bab7b'
plt.rcParams['figure.facecolor']  = '#0f1f0f'
plt.rcParams['axes.facecolor']    = '#0f1f0f'
plt.rcParams['axes.edgecolor']    = '#2d5a2d'
plt.rcParams['grid.color']        = '#1e3d1e'

# ─── CSS Personalizado — Tema AgroTech ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #d4ecd4 !important;
    font-size: 1.05rem;
}

[data-testid="stAppViewContainer"] {
    background-color: #0f1f0f !important;
}

[data-testid="stSidebar"] {
    background-color: #091409 !important;
    border-right: 2px solid rgba(76, 175, 80, 0.25);
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #c8e6c9 !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    color: #81c784 !important;
    letter-spacing: 1px;
}

.main-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #69f0ae 0%, #4caf50 45%, #d4a017 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 3px;
    margin-bottom: 0px;
}

.sub-title {
    color: #a5d6a7;
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 25px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.prediction-card {
    background: rgba(15, 40, 15, 0.95);
    border: 2px solid rgba(76, 175, 80, 0.4);
    border-radius: 10px;
    padding: 24px;
    margin-top: 15px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(76, 175, 80, 0.05);
    position: relative;
}

.prediction-card::before {
    content: '';
    position: absolute;
    top: 5px; left: 5px; right: 5px; bottom: 5px;
    border: 1px dashed rgba(76, 175, 80, 0.2);
    border-radius: 8px;
    pointer-events: none;
}

.prediction-culture {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #69f0ae;
    text-shadow: 0 0 15px rgba(105, 240, 174, 0.4);
    letter-spacing: 2px;
}

div.stButton > button {
    background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%) !important;
    color: #f1f8e9 !important;
    border: 2px solid rgba(76, 175, 80, 0.5) !important;
    padding: 12px 28px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    letter-spacing: 2px !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.25) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(105, 240, 174, 0.4) !important;
    background: linear-gradient(135deg, #69f0ae 0%, #4caf50 100%) !important;
    color: #0f1f0f !important;
}

table { font-size: 1.1rem !important; font-weight: 600 !important; color: #d4ecd4 !important; width: 100%; }
th { font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important; color: #81c784 !important; background-color: #1a2e1a !important; }
td { font-weight: 500 !important; }

div[data-testid="stTable"], div[data-testid="stDataFrame"] {
    background-color: #132013 !important;
    border: 2px solid #2d5a2d !important;
    border-radius: 8px;
}

.block-container { padding-top: 2.5rem; padding-bottom: 2.5rem; }

hr { border-color: rgba(76, 175, 80, 0.25) !important; border-width: 2px !important; }

div[data-testid="stFileUploader"] {
    background-color: #132013 !important;
    border: 2px dashed rgba(76, 175, 80, 0.35) !important;
    border-radius: 10px;
    padding: 20px;
}

/* Slider track verde */
[data-testid="stSlider"] > div > div > div { background: linear-gradient(90deg, #2e7d32, #4caf50) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Ícones por cultura ────────────────────────────────────────────────────────
CULTURA_ICONES = {
    'Milho':  '🌽',
    'Soja':   '🫘',
    'Arroz':  '🌾',
    'Trigo':  '🌾',
    'Feijao': '🫘',
}

CULTURA_NOME_PT = {
    'Milho':  'Milho',
    'Soja':   'Soja',
    'Arroz':  'Arroz',
    'Trigo':  'Trigo',
    'Feijao': 'Feijão',
}

# ─── Loaders ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_morphology_assets():
    try:
        model    = joblib.load('seed_classifier_model.pkl')
        features = joblib.load('model_features.pkl')
        classes  = joblib.load('model_classes.pkl')
        return model, features, classes
    except FileNotFoundError:
        st.error("❌ Modelo não encontrado. Execute 'python train_model.py' primeiro.")
        st.stop()

@st.cache_resource
def load_clip_assets():
    with st.spinner("🌿 Carregando sistema de visão computacional (CLIP)..."):
        model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor

@st.cache_data
def load_seeds_data():
    try:
        df = pd.read_csv('seeds_dataset.csv')
    except FileNotFoundError:
        st.warning("⚠️ 'seeds_dataset.csv' não encontrado. Execute 'train_model.py' primeiro.")
        return None
    return df

# ─── Cabeçalho ────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🌿 MATOSCAN</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Identificador de Culturas de Grãos por Inteligência Artificial</p>', unsafe_allow_html=True)

# ─── Navegação ────────────────────────────────────────────────────────────────
st.sidebar.markdown('### 🧭 MODO DE ANÁLISE')
modo = st.sidebar.radio(
    "Selecione o método:",
    ["⚙️ Morfometria da Semente", "📸 Varredura de Imagem"]
)

# ══════════════════════════════════════════════════════════════════════════════
# MODO 1 — MORFOMETRIA DA SEMENTE
# ══════════════════════════════════════════════════════════════════════════════
if modo == "⚙️ Morfometria da Semente":
    model, model_features, model_classes = load_morphology_assets()

    st.sidebar.markdown('### 📐 MEDIÇÕES DA SEMENTE')
    st.sidebar.write("Insira as medidas físicas da amostra:")

    comprimento_mm = st.sidebar.slider('Comprimento (mm)',      1.0, 25.0, 8.0,  step=0.1)
    largura_mm     = st.sidebar.slider('Largura (mm)',          1.0, 15.0, 6.0,  step=0.1)
    espessura_mm   = st.sidebar.slider('Espessura (mm)',        0.5, 12.0, 4.0,  step=0.1)
    massa_mg       = st.sidebar.slider('Massa (mg)',            5.0, 600.0, 200.0, step=1.0)

    # Features derivadas (calculadas automaticamente)
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

    seeds_df = load_seeds_data()

    col_left, col_right = st.columns([1.1, 1.2], gap="large")

    with col_left:
        st.markdown("### 📋 Medições Inseridas")
        st.markdown(f"""
        | Parâmetro | Valor |
        | :--- | :--- |
        | **Comprimento** | {comprimento_mm:.1f} mm |
        | **Largura** | {largura_mm:.1f} mm |
        | **Espessura** | {espessura_mm:.1f} mm |
        | **Massa** | {massa_mg:.0f} mg |
        | **Área (calculada)** | {area_mm2:.2f} mm² |
        | **Perímetro (calculado)** | {perimetro_mm:.2f} mm |
        | **Compacidade** | {compacidade:.3f} |
        | **Razão de Aspecto** | {razao_aspecto:.2f} |
        """)
        st.markdown("<br>", unsafe_allow_html=True)

        predict_clicked = st.sidebar.button("🌿 ANALISAR CULTURA")

        if predict_clicked or 'prediction_done' in st.session_state:
            st.session_state.prediction_done = True
            st.markdown("### 🔬 Resultado da Análise")

            try:
                prediction       = model.predict(user_input)
                prediction_proba = model.predict_proba(user_input)
                predicted        = prediction[0]
                icone            = CULTURA_ICONES.get(predicted, '🌱')
                nome_pt          = CULTURA_NOME_PT.get(predicted, predicted)

                st.markdown(f"""
                <div class="prediction-card">
                    <span style="font-size:0.85rem; color:#7bab7b; text-transform:uppercase; letter-spacing:1px;">Cultura Identificada</span>
                    <div class="prediction-culture">{icone} {nome_pt}</div>
                    <div style="margin-top:10px; font-size:0.95rem; color:#a5d6a7; line-height:1.5;">
                        O modelo identificou a semente com base nas assinaturas morfométricas fornecidas.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Tabela de probabilidades
                st.markdown("#### Probabilidades por Cultura")
                proba_df = pd.DataFrame(
                    prediction_proba,
                    columns=[CULTURA_NOME_PT.get(c, c) for c in model_classes]
                ).transpose()
                proba_df.columns = ['Confiança']
                st.dataframe(proba_df.style.format({"Confiança": "{:.2%}"}), use_container_width=True)

                # Gráfico de barras
                fig, ax = plt.subplots(figsize=(6, 2.5))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                labels_pt = [CULTURA_NOME_PT.get(c, c) for c in model_classes]
                colors    = ['#69f0ae' if c == predicted else '#2e5a2e' for c in model_classes]
                bars = ax.barh(labels_pt, prediction_proba[0], color=colors, height=0.5,
                               edgecolor='#2d5a2d', linewidth=1)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_color('#4caf50')
                ax.tick_params(axis='both', colors='#a5d6a7', labelsize=10)
                ax.xaxis.set_visible(False)
                for bar in bars:
                    w = bar.get_width()
                    ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{w:.1%}', va='center', ha='left',
                            color='#69f0ae', fontweight='bold', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Falha na análise: {e}")
        else:
            st.info("💡 Ajuste as medições e clique em **🌿 ANALISAR CULTURA** no painel lateral.")

    with col_right:
        st.markdown("### 📊 Distribuição Populacional")

        if seeds_df is not None:
            palette = {
                'Milho':  '#69f0ae',
                'Soja':   '#d4a017',
                'Arroz':  '#4fc3f7',
                'Trigo':  '#ffb74d',
                'Feijao': '#ef5350',
            }

            # Scatter — Comprimento vs Largura
            fig1, ax1 = plt.subplots(figsize=(7, 4.5))
            sns.scatterplot(
                data=seeds_df, x='comprimento_mm', y='largura_mm',
                hue='cultura', palette=palette, alpha=0.5, s=30, ax=ax1,
                edgecolor='#0f1f0f', linewidth=0.3
            )
            ax1.scatter(
                comprimento_mm, largura_mm, color='#ffffff',
                edgecolor='#69f0ae', s=280, marker='*',
                label='Amostra Atual', zorder=5, linewidths=1.5
            )
            ax1.set_title('Comprimento × Largura da Semente', fontsize=11,
                          fontweight='bold', pad=12, family='Rajdhani')
            ax1.set_xlabel('Comprimento (mm)', fontsize=9)
            ax1.set_ylabel('Largura (mm)', fontsize=9)
            ax1.legend(frameon=True, facecolor='#091409', edgecolor='#2d5a2d',
                       framealpha=0.9, fontsize=8)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.grid(True, linestyle=':', alpha=0.15)
            plt.tight_layout()
            st.pyplot(fig1)

            # Histograma — Massa por cultura
            fig2, ax2 = plt.subplots(figsize=(7, 4.5))
            sns.histplot(
                data=seeds_df, x='massa_mg', hue='cultura', palette=palette,
                kde=True, multiple="stack", alpha=0.4, ax=ax2,
                edgecolor='#0f1f0f', linewidth=0.3
            )
            ax2.axvline(
                massa_mg, color='#ffffff', linestyle='--', linewidth=2,
                label=f'Amostra ({massa_mg:.0f} mg)'
            )
            ax2.set_title('Distribuição de Massa por Cultura', fontsize=11,
                          fontweight='bold', pad=12, family='Rajdhani')
            ax2.set_xlabel('Massa (mg)', fontsize=9)
            ax2.set_ylabel('Frequência', fontsize=9)
            ax2.legend(frameon=True, facecolor='#091409', edgecolor='#2d5a2d',
                       framealpha=0.9, fontsize=8)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.grid(True, linestyle=':', alpha=0.15)
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("Execute 'train_model.py' para gerar os dados de referência.")


# ══════════════════════════════════════════════════════════════════════════════
# MODO 2 — VARREDURA DE IMAGEM (CLIP)
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("### 📸 Identificação por Imagem da Planta")
    st.write("Faça upload de uma foto da planta ou grão e o MatoScan utilizará Visão Computacional (CLIP) para identificar a cultura.")

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### 📂 Imagem da Planta")
        uploaded_file = st.file_uploader(
            "Arraste ou selecione a imagem:",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem Carregada", use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            diagnosticar = st.button("🌿 IDENTIFICAR CULTURA")
        else:
            st.info("💡 Envie uma foto da planta, folha ou grão para iniciar a identificação.")

    with col_result:
        st.markdown("#### 🔬 Resultado da Análise Visual")

        if uploaded_file is not None and (diagnosticar or 'vision_prediction_done' in st.session_state):
            st.session_state.vision_prediction_done = True

            clip_model, clip_processor = load_clip_assets()

            # Culturas e prompts descritivos para o CLIP
            culturas_clip = ['Milho', 'Soja', 'Arroz', 'Trigo', 'Feijao']
            prompts_clip  = [
                "a photo of a corn plant with large green leaves and tall stalk",
                "a photo of a soybean plant with small oval leaves and pods",
                "a photo of a rice plant growing in water with long thin leaves",
                "a photo of a wheat plant with golden grain spikes",
                "a photo of a bean plant with climbing vines and bean pods",
            ]

            with st.status("🌿 Processando Imagem...", expanded=True) as status:
                st.write("Preparando imagem para análise...")
                time.sleep(0.5)
                st.write("Extraindo características visuais (CLIP)...")
                time.sleep(0.5)
                st.write("Computando similaridade com culturas conhecidas...")

                inputs = clip_processor(
                    text=prompts_clip,
                    images=image,
                    return_tensors="pt",
                    padding=True
                )

                with torch.no_grad():
                    outputs = clip_model(**inputs)

                logits = outputs.logits_per_image
                probs  = logits.softmax(dim=1).numpy()[0]

                status.update(label="✅ Análise concluída!", state="complete", expanded=False)

            predicted_idx  = int(np.argmax(probs))
            predicted      = culturas_clip[predicted_idx]
            icone          = CULTURA_ICONES.get(predicted, '🌱')
            nome_pt        = CULTURA_NOME_PT.get(predicted, predicted)

            st.markdown(f"""
            <div class="prediction-card">
                <span style="font-size:0.85rem; color:#7bab7b; text-transform:uppercase; letter-spacing:1px;">Cultura Identificada por Imagem</span>
                <div class="prediction-culture">{icone} {nome_pt}</div>
                <div style="margin-top:10px; font-size:0.95rem; color:#a5d6a7; line-height:1.5;">
                    Análise realizada localmente via rede neural profunda (CLIP).
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Tabela de probabilidades
            st.markdown("#### Similaridade Visual por Cultura")
            nomes_pt  = [CULTURA_NOME_PT.get(c, c) for c in culturas_clip]
            proba_df  = pd.DataFrame(probs, index=nomes_pt, columns=['Confiança'])
            st.dataframe(proba_df.style.format({"Confiança": "{:.2%}"}), use_container_width=True)

            # Gráfico
            fig, ax = plt.subplots(figsize=(6, 2.5))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            colors = ['#69f0ae' if c == predicted else '#2e5a2e' for c in culturas_clip]
            bars   = ax.barh(nomes_pt, probs, color=colors, height=0.5,
                             edgecolor='#2d5a2d', linewidth=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_color('#4caf50')
            ax.tick_params(axis='both', colors='#a5d6a7', labelsize=10)
            ax.xaxis.set_visible(False)
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{w:.1%}', va='center', ha='left',
                        color='#69f0ae', fontweight='bold', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)

        else:
            st.info("📊 O resultado da análise visual aparecerá aqui após o envio da imagem.")
