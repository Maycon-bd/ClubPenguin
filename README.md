# 🌿 MatoScan — Identificador Inteligente de Culturas por IA

> Dashboard interativo premium desenvolvido em Streamlit para identificação e classificação de culturas de grãos (Milho, Soja, Arroz, Trigo, Feijão) utilizando modelos híbridos de Machine Learning e Visão Computacional Zero-Shot.

[![Python](https://img.shields.io/badge/Python-3.9%2B-2E7D32?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-CLIP-yellow?style=flat-square)](https://huggingface.co/openai/clip-vit-base-patch32)

---

## 📌 Sobre o Projeto

O **MatoScan** é uma plataforma AgriTech projetada para análise e identificação instantânea de sementes e culturas agrícolas. A aplicação possui uma interface moderna e responsiva inspirada em cockpits de monitoramento agrícola (como *John Deere Operations Center* e *Climate FieldView*), operando em dois modos inteligentes:

1. **⚙️ Módulo Morfometria da Semente**: Utiliza algoritmos de **Machine Learning (Random Forest)** para predizer a cultura baseando-se em características físicas e morfológicas inseridas por controles deslizantes interativos.
2. **📸 Módulo Visão Computacional**: Utiliza a rede neural **CLIP (Contrastive Language-Image Pre-Training)** da OpenAI de forma 100% local para classificar imagens de plantas ou sementes por similaridade semântica contextual.

---

## 📐 Parâmetros Morfológicos e Fórmulas

No modo morfométrico, o sistema recebe quatro medidas primárias da semente e calcula automaticamente quatro métricas geométricas derivadas para enriquecer o vetor de características utilizado na predição:

### Características Primárias:
- **Comprimento ($L$)** (mm) — Medida do eixo maior.
- **Largura ($W$)** (mm) — Medida do eixo menor.
- **Espessura ($T$)** (mm) — Profundidade da semente.
- **Massa ($M$)** (mg) — Peso individual estimado.

### Características Derivadas:
- **Área Projetada ($A$)** (mm²): Calculada aproximando a projeção da semente a uma elipse:
  $$A = \frac{\pi \cdot L \cdot W}{4}$$
- **Perímetro ($P$)** (mm): Calculado utilizando a segunda aproximação de Ramanujan para a circunferência de uma elipse:
  $$P \approx \pi \left[ 3(a + b) - \sqrt{(3a + b)(a + 3b)} \right]$$
  *(onde $a = L/2$ e $b = W/2$)*
- **Compacidade ($C$)**: Razão de circularidade da forma projetada:
  $$C = \frac{4\pi \cdot A}{P^2}$$
  *(valores próximos a $1.0$ indicam formas perfeitamente circulares, como a soja)*
- **Razão de Aspecto ($AR$)**: Relação de alongamento da semente:
  $$AR = \frac{L}{W}$$

---

## 🌾 Guia de Valores de Teste Rápido (Modo Morfometria)

Para validar o funcionamento do classificador de forma imediata e obter **100% de confiança** no resultado, posicione os controles deslizantes da barra lateral com os valores padrão de referência descritos abaixo:

| Cultura Alvo | Comprimento | Largura | Espessura | Massa | Formato Esperado |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **🌽 Milho** | `10.0 mm` | `8.0 mm` | `5.0 mm` | `320 mg` | Oval achatado |
| **🫘 Soja** | `6.5 mm` | `6.0 mm` | `5.0 mm` | `195 mg` | Altamente esférico (compacidade $\approx 1$) |
| **🌾 Arroz** | `7.5 mm` | `2.4 mm` | `2.0 mm` | `26 mg` | Altamente alongado e leve |
| **🌾 Trigo** | `6.0 mm` | `3.5 mm` | `3.0 mm` | `42 mg` | Ovalado, pequeno e leve |
| **🫘 Feijão** | `15.0 mm` | `9.0 mm` | `7.0 mm` | `320 mg` | Reniforme e grande |

---

## 🚀 Como Instalar e Executar

### 1. Clonar o Repositório e Instalar Dependências
Certifique-se de possuir o Python 3.9+ instalado. Instale os pacotes necessários rodando:
```bash
pip install -r requirements.txt
```

### 2. Treinar o Modelo de Machine Learning
Para gerar a base de dados sintética morfométrica (1.750 amostras balanceadas) e treinar o classificador Random Forest, execute:
```bash
python train_model.py
```
*Este script exibirá o relatório de classificação no terminal com uma acurácia de **99.4%** no conjunto de teste e criará os arquivos de serialização necessários.*

### 3. Executar o Dashboard Streamlit
Com o modelo treinado, inicialize a interface web:
```bash
streamlit run app.py
```
O sistema abrirá automaticamente no navegador no endereço `http://localhost:8501`.

> ℹ️ **Nota sobre o Modo Visão Computacional (CLIP)**: Na primeira vez em que selecionar este modo e enviar uma imagem, o Streamlit fará o download local do modelo CLIP (`openai/clip-vit-base-patch32` — cerca de ~600 MB) do Hugging Face. As execuções subsequentes serão instantâneas e totalmente locais.

---

## 🗂️ Estrutura do Sistema

A varredura atualizada do diretório do projeto confirmou a seguinte organização de arquivos:

```
ClubPenguin/ (Futuro MatoScan)
├── app.py                         # Código da aplicação Streamlit (UI Premium + CLIP)
├── train_model.py                 # Script de simulação de dados e treinamento RF
├── requirements.txt               # Dependências do Python (Streamlit, PyTorch, etc.)
├── seed_classifier_model.pkl      # Modelo RandomForest serializado via Joblib
├── model_features.pkl             # Lista ordenada das features utilizadas pelo modelo
├── model_classes.pkl              # Lista ordenada das classes de culturas
├── seeds_dataset.csv              # Dataset de referência gerado com 1.750 amostras
├── .gitignore                     # Arquivos ignorados pelo controle de versão Git
└── README.md                      # Documentação do projeto (este arquivo)
```

---

## 🧠 Detalhes dos Modelos

### Módulo Morfometria — Random Forest
- **Dataset de Referência**: 1.750 amostras simulando distribuições morfológicas reais (350 amostras por cultura).
- **Algoritmo**: `Pipeline` contendo `StandardScaler` seguido de um `RandomForestClassifier` com 200 árvores de decisão.
- **Divisão do Dataset**: 80% Treino / 20% Teste (com estratificação de classes).
- **Acurácia obtida**: **99.43%**
- **Relatório de Classificação**:
  - **Arroz**: Precision: 1.00 \| Recall: 1.00
  - **Feijão**: Precision: 0.99 \| Recall: 0.99
  - **Milho**: Precision: 0.99 \| Recall: 0.99
  - **Soja**: Precision: 1.00 \| Recall: 1.00
  - **Trigo**: Precision: 1.00 \| Recall: 1.00

### Módulo Visão Computacional — CLIP (Zero-Shot)
- **Modelo**: `openai/clip-vit-base-patch32` (Transformer multimodal).
- **Abordagem**: Classificação baseada na proximidade do embedding da imagem em relação a prompts textuais específicos, traduzidos para o inglês de forma a otimizar a performance do modelo CLIP:
  - *Milho*: `"a close-up photo of corn plant or corn grain seeds"`
  - *Soja*: `"a close-up photo of soybean plant or soybean seeds"`
  - *Arroz*: `"a close-up photo of rice plant or rice grains"`
  - *Trigo*: `"a close-up photo of wheat plant or wheat grains"`
  - *Feijão*: `"a close-up photo of bean plant or bean seeds"`

---

## 🔭 Roadmap de Evolução

- [ ] **Integração de Base de Imagens Reais**: Substituir o classificador zero-shot por um modelo Fine-Tuned (ex: ResNet ou Vision Transformer) treinado em dataset próprio de sementes.
- [ ] **Análise de Qualidade de Grãos**: Detectar percentual de grãos quebrados, ardidos, avariados e presença de impurezas na amostra.
- [ ] **Exportação de Relatórios (Laudo Técnico)**: Permitir exportar as medições e gráficos plotados em formato PDF assinado digitalmente.
- [ ] **Suporte a Novas Culturas**: Expandir a identificação para café, girassol, cevada e canola.

---

## 👤 Autor

**Maycon Garcia Silva**

[![GitHub](https://img.shields.io/badge/GitHub-Maycon--bd-181717?style=flat-square&logo=github)](https://github.com/Maycon-bd)

---

## 📄 Licença

Este projeto está sob a licença **MIT**. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

> 🌿 *"Conhecer a semente é conhecer a safra."* 🌾
