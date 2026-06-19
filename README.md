# 🌿 MatoScan — Identificador de Culturas de Grãos por IA

> Dashboard interativo com Streamlit para identificação de espécies de culturas agrícolas (Milho, Soja, Arroz, Trigo, Feijão) usando Machine Learning e Visão Computacional.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## 📌 Sobre o Projeto

O **MatoScan** é um sistema de identificação de culturas de grãos baseado em **morfometria de sementes** e **visão computacional**. Ele oferece dois modos de diagnóstico:

1. **⚙️ Morfometria da Semente** — Identificação a partir de medições físicas da semente (comprimento, largura, espessura, massa) usando um modelo **Random Forest** treinado com dados morfológicos reais.
2. **📸 Varredura de Imagem** — Identificação da cultura a partir de uma fotografia da planta, usando o modelo **CLIP** (OpenAI) rodando localmente, sem necessidade de API externa.

O design segue um tema **AgroTech** moderno com paleta em tons de verde esmeralda, âmbar e preto floresta.

---

## 🌾 Culturas Suportadas

| Cultura | Ícone | Características da Semente |
|---|---|---|
| **Milho** | 🌽 | Comprimento 8–12mm, massa 250–400mg, forma oval-achatada |
| **Soja** | 🫘 | Comprimento 5–8mm, massa 150–250mg, forma esférica |
| **Arroz** | 🌾 | Comprimento 6–9mm, massa 18–35mg, forma alongada e fina |
| **Trigo** | 🌾 | Comprimento 5–7mm, massa 30–55mg, forma oval-cônica |
| **Feijão** | 🫘 | Comprimento 12–18mm, massa 200–450mg, forma reniforme |

---

## 🚀 Como Executar

### 1. Pré-requisitos

- Python **3.9** ou superior
- `pip` atualizado

### 2. Instalar as Dependências

```bash
pip install -r requirements.txt
```

> ⚠️ O modelo CLIP (~600 MB) é baixado automaticamente do Hugging Face na primeira execução do Modo Imagem.

### 3. Treinar o Modelo (obrigatório antes de usar o Modo Morfometria)

```bash
python train_model.py
```

Isso irá gerar:
- `seed_classifier_model.pkl` — modelo Random Forest treinado
- `model_features.pkl` — lista de features
- `model_classes.pkl` — classes (culturas)
- `seeds_dataset.csv` — dataset morfológico de referência

### 4. Iniciar o Dashboard

```bash
streamlit run app.py
```

Acesse em: `http://localhost:8501`

---

## 📦 Dependências

| Pacote | Finalidade |
|---|---|
| `streamlit` | Interface web interativa |
| `pandas` | Manipulação de dados |
| `scikit-learn` | Modelo Random Forest |
| `matplotlib` | Gráficos e visualizações |
| `seaborn` | Plots estatísticos |
| `joblib` | Serialização do modelo |
| `numpy` | Operações numéricas |
| `Pillow` | Processamento de imagens |
| `torch` | Backend PyTorch para o CLIP |
| `transformers` | Modelo CLIP da Hugging Face |

---

## 🗂️ Estrutura do Projeto

```
MatoScan/
├── app.py                      # Aplicação principal Streamlit
├── train_model.py              # Script de treinamento + geração do dataset
├── seed_classifier_model.pkl   # Modelo treinado (gerado automaticamente)
├── model_features.pkl          # Features do modelo
├── model_classes.pkl           # Classes (culturas)
├── seeds_dataset.csv           # Dataset morfológico de referência
├── requirements.txt            # Dependências do projeto
├── .gitignore                  # Arquivos ignorados pelo Git
└── README.md                   # Este arquivo
```

---

## 🧠 Sobre os Modelos

### Modo 1 — Random Forest (Morfometria)
- **Dataset:** Gerado com base em parâmetros morfológicos reais das sementes (1.750 amostras — 350 por cultura)
- **Features:** `comprimento_mm`, `largura_mm`, `espessura_mm`, `massa_mg`, `area_mm2`, `perimetro_mm`, `compacidade`, `razao_aspecto`
- **Pipeline:** `StandardScaler` + `RandomForestClassifier (200 árvores)`
- **Split:** 80% treino / 20% teste
- **Acurácia esperada:** ~99%

### Modo 2 — CLIP (Visão Computacional)
- **Modelo:** `openai/clip-vit-base-patch32` via Hugging Face Transformers
- **Método:** Zero-shot classification com prompts descritivos por cultura
- **Execução:** 100% local — sem API externa

---

## 📐 Features Morfométricas

| Feature | Unidade | Descrição |
|---|---|---|
| `comprimento_mm` | mm | Comprimento do eixo maior da semente |
| `largura_mm` | mm | Largura do eixo menor da semente |
| `espessura_mm` | mm | Profundidade/espessura da semente |
| `massa_mg` | mg | Massa individual da semente |
| `area_mm2` | mm² | Área projetada (calculada como elipse) |
| `perimetro_mm` | mm | Perímetro da projeção 2D (fórmula de Ramanujan) |
| `compacidade` | adim. | 4π·área/perímetro² — quanto mais próximo de 1, mais circular |
| `razao_aspecto` | adim. | comprimento/largura — indica alongamento da semente |

---

## 🔭 Roadmap

- [ ] **Deploy** no Streamlit Community Cloud para acesso público
- [ ] **Dataset real** — integrar com dataset de visão computacional de grãos reais
- [ ] **Detecção de qualidade** — identificar grãos avariados, mofados ou quebrados
- [ ] **Modo câmera** — captura em tempo real via webcam
- [ ] **Exportar laudo** — PDF com resultado da análise e gráficos
- [ ] **Testes unitários** — cobertura das funções de predição
- [ ] **Suporte a mais culturas** — Café, Cana, Girassol

---

## 👤 Autor

**Maycon Garcia Silva**

[![GitHub](https://img.shields.io/badge/GitHub-Maycon--bd-181717?style=flat-square&logo=github)](https://github.com/Maycon-bd)

---

## 📄 Licença

Este projeto está sob a licença **MIT**. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

> *"Conhecer a semente é conhecer a safra."* 🌾
