# app.py
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

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Mecanismo de Classificação - Palmer Penguins",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar estilo global do matplotlib para combinar com o tema steampunk minimalista
plt.rcParams['text.color'] = '#dfd5c6'
plt.rcParams['axes.labelcolor'] = '#dfd5c6'
plt.rcParams['xtick.color'] = '#a49788'
plt.rcParams['ytick.color'] = '#a49788'
plt.rcParams['figure.facecolor'] = '#1a1615'
plt.rcParams['axes.facecolor'] = '#1a1615'
plt.rcParams['axes.edgecolor'] = '#3c3029'
plt.rcParams['grid.color'] = '#3c3029'

# Estilização CSS personalizada para tema Steampunk Minimalista com fontes evidentes
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800;900&family=Outfit:wght@400;500;600;700;800;900&display=swap');

/* Configurar fontes e cores de fundo com alta nitidez */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: #f5eedf !important;
    font-size: 1.05rem;
}

[data-testid="stAppViewContainer"] {
    background-color: #1a1615 !important;
}

[data-testid="stSidebar"] {
    background-color: #120f0e !important;
    border-right: 2px solid rgba(184, 115, 51, 0.3);
}

/* Colorir labels e textos do sidebar para ficarem muito evidentes */
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
    color: #f5eedf !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}

/* Estilizar títulos globais da página (markdown) */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Cinzel', serif !important;
    font-weight: 800 !important;
    color: #d4af37 !important;
    letter-spacing: 1px;
}

/* Título principal com gradiente metálico espesso */
.main-title {
    font-family: 'Cinzel', serif;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #f1c40f 0%, #d4af37 40%, #e67e22 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 2px;
    margin-bottom: 0px;
    padding-bottom: 5px;
}

.sub-title {
    color: #e5cda8;
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 25px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* Card de Resultado de Previsão */
.prediction-card {
    background: rgba(30, 26, 24, 0.9);
    border: 2px solid rgba(184, 115, 51, 0.45);
    border-radius: 8px;
    padding: 24px;
    margin-top: 15px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6), inset 0 0 15px rgba(184, 115, 51, 0.1);
    position: relative;
}

.prediction-card::before {
    content: '';
    position: absolute;
    top: 5px; left: 5px; right: 5px; bottom: 5px;
    border: 1px dashed rgba(184, 115, 51, 0.25);
    border-radius: 6px;
    pointer-events: none;
}

.prediction-species {
    font-family: 'Cinzel', serif;
    font-size: 2.6rem;
    font-weight: 900;
    color: #f1c40f;
    text-shadow: 0 0 12px rgba(241, 196, 15, 0.4);
    letter-spacing: 1.5px;
}

/* Botão de bronze robusto */
div.stButton > button {
    background: linear-gradient(135deg, #d4af37 0%, #b87333 100%) !important;
    color: #120f0e !important;
    border: 2px solid rgba(184, 115, 51, 0.6) !important;
    padding: 12px 28px !important;
    font-family: 'Cinzel', serif !important;
    font-weight: 900 !important;
    font-size: 1.1rem !important;
    letter-spacing: 1.5px !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 15px rgba(184, 115, 51, 0.3) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(241, 196, 15, 0.45) !important;
    background: linear-gradient(135deg, #f1c40f 0%, #d4af37 100%) !important;
    color: #120f0e !important;
}

/* Estilização de Tabelas de Medição e Dados */
table {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #f5eedf !important;
    width: 100%;
}
th {
    font-family: 'Cinzel', serif !important;
    font-weight: 800 !important;
    color: #d4af37 !important;
    background-color: #231f1d !important;
}
td {
    font-weight: 600 !important;
}

/* Customização de dataframes interativos */
div[data-testid="stTable"], div[data-testid="stDataFrame"] {
    background-color: #1e1a18 !important;
    border: 2px solid #3c3029 !important;
    border-radius: 6px;
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2.5rem;
}

hr {
    border-color: rgba(184, 115, 51, 0.3) !important;
    border-width: 2px !important;
}

/* Container de upload steampunk */
div[data-testid="stFileUploader"] {
    background-color: #1e1a18 !important;
    border: 2px dashed rgba(184, 115, 51, 0.4) !important;
    border-radius: 8px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# 1. Carregar Modelo do Tab 1 (Morfometria)
@st.cache_resource
def load_morphology_assets():
    try:
        model = joblib.load('penguin_species_predictor_model.pkl')
        features = joblib.load('model_features.pkl')
        classes = joblib.load('model_classes.pkl')
        return model, features, classes
    except FileNotFoundError:
        st.error("Erro: Arquivos do modelo não encontrados. Por favor, execute 'train_model.py' primeiro.")
        st.stop()

# 2. Carregar Modelo do Tab 2 (Processamento de Imagem - CLIP)
@st.cache_resource
def load_clip_assets():
    with st.spinner("🔧 Calibrando Óptica de Visão Computacional (Carregando CLIP local)..."):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor

# Cabeçalho global
st.markdown('<h1 class="main-title">⚙️ MECANISMO DE CLASSIFICAÇÃO</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Análise de Espécimes via Aprendizado de Máquina & Medições Físicas</p>', unsafe_allow_html=True)

# Menu de navegação no painel lateral
st.sidebar.markdown('### 🧭 NAVEGAÇÃO')
modo_diagnostico = st.sidebar.radio(
    "Selecione o método de análise:",
    ["⚙️ Parâmetros Morfométricos", "📸 Varredura de Imagem"]
)

# ----------------- MODO 1: PARÂMETROS MORFOMÉTRICOS -----------------
if modo_diagnostico == "⚙️ Parâmetros Morfométricos":
    model, model_features, model_classes = load_morphology_assets()

    st.sidebar.markdown('### ⚙️ PARÂMETROS DO ESPÉCIME')
    st.sidebar.write("Calibrar as medições físicas:")

    bill_length_mm = st.sidebar.slider('Comprimento do Bico (mm)', 30.0, 60.0, 45.0, step=0.1)
    bill_depth_mm = st.sidebar.slider('Profundidade do Bico (mm)', 10.0, 25.0, 17.0, step=0.1)
    flipper_length_mm = st.sidebar.slider('Comprimento da Nadadeira (mm)', 170.0, 240.0, 200.0, step=1.0)
    body_mass_g = st.sidebar.slider('Massa Corporal (g)', 2500.0, 6500.0, 4000.0, step=50.0)
    island = st.sidebar.selectbox('Ilha de Coleta', ['Torgersen', 'Biscoe', 'Dream'])
    sex = st.sidebar.selectbox('Sexo Biológico', ['Male', 'Female'])

    # Organizar a entrada
    user_input_df = pd.DataFrame([[bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, island, sex]], 
                                 columns=model_features)

    # Dados originais para visualização
    @st.cache_data
    def load_original_data():
        try:
            df = sns.load_dataset('penguins')
        except Exception:
            try:
                df = pd.read_csv('penguins.csv')
            except Exception:
                url = "https://github.com/allisonhorst/palmerpenguins/raw/master/inst/extdata/penguins.csv"
                df = pd.read_csv(url)
                df.to_csv('penguins.csv', index=False)
        df.dropna(inplace=True)
        if 'sex' in df.columns and '.' in df['sex'].unique():
            df = df[df['sex'] != '.']
        return df

    original_df = load_original_data()

    col_left, col_right = st.columns([1.1, 1.2], gap="large")

    with col_left:
        st.markdown("### 📋 Calibração Atual")
        st.markdown(
            f"""
            | Medição | Valor Calibrado |
            | :--- | :--- |
            | **Comprimento do Bico** | {bill_length_mm} mm |
            | **Profundidade do Bico** | {bill_depth_mm} mm |
            | **Comprimento da Nadadeira** | {flipper_length_mm} mm |
            | **Massa Corporal** | {body_mass_g} g |
            | **Ilha** | {island} |
            | **Sexo** | {sex} |
            """
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
        predict_clicked = st.sidebar.button('INICIAR DIAGNÓSTICO ⚙️')
        
        if predict_clicked or 'prediction_done' in st.session_state:
            st.session_state.prediction_done = True
            st.markdown("### 🔮 Resultado do Diagnóstico")
            
            try:
                prediction = model.predict(user_input_df)
                prediction_proba = model.predict_proba(user_input_df)
                predicted_species = prediction[0]
                
                # Card
                st.markdown(f"""
                <div class="prediction-card">
                    <span style="font-size: 0.9rem; color: #a49788; text-transform: uppercase; letter-spacing: 1px;">Espécie Diagnosticada</span>
                    <div class="prediction-species">🐧 {predicted_species}</div>
                    <div style="margin-top: 10px; font-size: 0.95rem; line-height: 1.4; color: #dfd5c6;">
                        O mecanismo identificou o espécime com base nas assinaturas morfométricas calibradas.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Tabela de Probabilidades
                st.markdown("#### Matriz de Probabilidades")
                proba_df = pd.DataFrame(prediction_proba, columns=model_classes).transpose()
                proba_df.columns = ['Confiabilidade']
                st.dataframe(proba_df.style.format({"Confiabilidade": "{:.2%}"}), use_container_width=True)
                
                # Gráfico
                fig, ax = plt.subplots(figsize=(6, 2.5))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                colors = ['#d4af37' if c == predicted_species else '#8c5c30' for c in model_classes]
                bars = ax.barh(model_classes, proba_df['Confiabilidade'], color=colors, height=0.5, edgecolor='#3c3029', linewidth=1)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_color('#8c5c30')
                ax.tick_params(axis='both', colors='#dfd5c6', labelsize=10)
                ax.xaxis.set_visible(False)
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.1%}', 
                            va='center', ha='left', color='#d4af37', fontweight='bold', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Falha no diagnóstico: {e}")
        else:
            st.info("💡 Acione o botão **INICIAR DIAGNÓSTICO** no painel lateral para rodar a inferência.")

    with col_right:
        st.markdown("### 📈 Mapeamento Populacional")
        st.write("Posicionamento do espécime calibrado em relação à distribuição histórica:")
        
        steampunk_palette = {"Adelie": "#a47551", "Chinstrap": "#d4af37", "Gentoo": "#5c4033"}
        
        fig_scatter, ax_scatter = plt.subplots(figsize=(7, 4.5))
        sns.scatterplot(
            data=original_df, x='bill_length_mm', y='bill_depth_mm', hue='species', 
            palette=steampunk_palette, alpha=0.6, s=40, ax=ax_scatter, edgecolor='#1a1615', linewidth=0.5
        )
        ax_scatter.scatter(
            bill_length_mm, bill_depth_mm, color='#ff7f50', edgecolor='white', s=250, 
            marker='*', label='Calibração Atual', zorder=5, linewidths=1.5
        )
        ax_scatter.set_title('Morfologia do Bico (Comprimento vs. Profundidade)', fontsize=11, fontweight='bold', pad=12, family='Cinzel')
        ax_scatter.set_xlabel('Comprimento do Bico (mm)', fontsize=9)
        ax_scatter.set_ylabel('Profundidade do Bico (mm)', fontsize=9)
        ax_scatter.legend(frameon=True, facecolor='#120f0e', edgecolor='#3c3029', framealpha=0.9)
        ax_scatter.spines['top'].set_visible(False)
        ax_scatter.spines['right'].set_visible(False)
        ax_scatter.grid(True, linestyle=':', alpha=0.1)
        plt.tight_layout()
        st.pyplot(fig_scatter)
        
        fig_hist, ax_hist = plt.subplots(figsize=(7, 4.5))
        sns.histplot(
            data=original_df, x='flipper_length_mm', hue='species', palette=steampunk_palette,
            kde=True, multiple="stack", alpha=0.4, ax=ax_hist, edgecolor='#1a1615', linewidth=0.5
        )
        ax_hist.axvline(
            flipper_length_mm, color='#ff7f50', linestyle='--', linewidth=2, 
            label=f'Atual ({int(flipper_length_mm)} mm)'
        )
        ax_hist.set_title('Distribuição da Nadadeira', fontsize=11, fontweight='bold', pad=12, family='Cinzel')
        ax_hist.set_xlabel('Comprimento da Nadadeira (mm)', fontsize=9)
        ax_hist.set_ylabel('Frequência', fontsize=9)
        ax_hist.legend(frameon=True, facecolor='#120f0e', edgecolor='#3c3029', framealpha=0.9)
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.grid(True, linestyle=':', alpha=0.1)
        plt.tight_layout()
        st.pyplot(fig_hist)


# ----------------- MODO 2: VARREDURA DE IMAGEM (CLIP) -----------------
else:
    st.markdown("### 📸 Diagnóstico por Análise de Imagem")
    st.write("Anexe a fotografia de um pinguim e o mecanismo utilizará Visão Computacional profunda (modelo local CLIP) para identificar a espécie.")

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### 📂 Entrada de Mídia")
        uploaded_file = st.file_uploader(
            "Arraste ou selecione a imagem do pinguim:",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Fotografia Carregada", use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            diagnosticar = st.button("INICIAR DIAGNÓSTICO VISUAL ⚙️")
        else:
            st.info("💡 Por favor, anexe uma imagem de pinguim para iniciar o mecanismo óptico.")

    with col_result:
        st.markdown("#### 🔮 Resultados da Análise Óptica")
        
        if uploaded_file is not None and (diagnosticar or 'vision_prediction_done' in st.session_state):
            st.session_state.vision_prediction_done = True
            
            # Carregar pesos locais de processamento de imagem
            clip_model, clip_processor = load_clip_assets()
            
            with st.status("⚙️ Executando Varredura Óptica Local...", expanded=True) as status:
                st.write("Focalizando lentes ópticas...")
                time.sleep(0.6)
                st.write("Processando dimensões dos pixels...")
                time.sleep(0.5)
                st.write("Computando vetor de proximidade visual (CLIP)...")
                
                # Executar classificação zero-shot
                labels = ["Adelie penguin", "Chinstrap penguin", "Gentoo penguin"]
                
                # O CLIP entende melhor com prompts contextualizados
                prompts = [f"a photo of an {label}" for label in labels]
                
                inputs = clip_processor(
                    text=prompts, 
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                
                # Extrair probabilidades
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).numpy()[0]
                
                status.update(label="Varredura concluída com sucesso!", state="complete", expanded=False)
            
            # Mapear resultados
            classes_mapeadas = ["Adelie", "Chinstrap", "Gentoo"]
            predicted_idx = np.argmax(probs)
            predicted_species = classes_mapeadas[predicted_idx]
            
            # Card
            st.markdown(f"""
            <div class="prediction-card">
                <span style="font-size: 0.9rem; color: #a49788; text-transform: uppercase; letter-spacing: 1px;">Espécie Diagnosticada por Imagem</span>
                <div class="prediction-species">🐧 {predicted_species}</div>
                <div style="margin-top: 10px; font-size: 0.95rem; line-height: 1.4; color: #dfd5c6;">
                    Varredura óptica realizada localmente via rede neural profunda.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Tabela de Confiança
            st.markdown("#### Matriz de Probabilidades Visuais")
            proba_df = pd.DataFrame(probs, index=classes_mapeadas, columns=['Confiabilidade Visual'])
            st.dataframe(proba_df.style.format({"Confiabilidade Visual": "{:.2%}"}), use_container_width=True)
            
            # Gráfico de barras horizontais personalizado (Steampunk tones)
            fig, ax = plt.subplots(figsize=(6, 2.5))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            
            colors = ['#d4af37' if c == predicted_species else '#8c5c30' for c in classes_mapeadas]
            bars = ax.barh(classes_mapeadas, proba_df['Confiabilidade Visual'], color=colors, height=0.5, edgecolor='#3c3029', linewidth=1)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_color('#8c5c30')
            ax.tick_params(axis='both', colors='#dfd5c6', labelsize=10)
            ax.xaxis.set_visible(False)
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.1%}', 
                        va='center', ha='left', color='#d4af37', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("📊 Os resultados do processamento visual e as porcentagens de confiança aparecerão nesta seção após o diagnóstico.")
