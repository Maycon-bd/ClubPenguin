# 🐧 ClubPenguin — Classificador de Espécies de Pinguins por IA

> Dashboard interativo com Streamlit para classificação de espécies de pinguins usando Machine Learning e Visão Computacional.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-✓-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## 📌 Sobre o Projeto

O **ClubPenguin** é um sistema de classificação de espécies de pinguins baseado no dataset **Palmer Penguins**. Ele oferece dois modos de diagnóstico:

1. **⚙️ Parâmetros Morfométricos** — Classificação por medições físicas (bico, nadadeira, massa, ilha, sexo) usando um modelo de Regressão Logística treinado com `scikit-learn`.
2. **📸 Varredura de Imagem** — Classificação por fotografia usando o modelo **CLIP** (OpenAI) rodando localmente, sem necessidade de API externa.

O design segue um tema **Steampunk Minimalista** com paleta em tons de bronze, dourado e carvão — proporcionando uma experiência visual única e imersiva.

---

## 🖼️ Interface

| Modo Morfométrico | Modo Imagem (CLIP) |
|---|---|
| Sliders para medições físicas | Upload de foto do pinguim |
| Gráfico de probabilidades | Resultado com confiança visual |
| Scatter plot e histograma populacionais | Barra de progresso animada |

---

## 🚀 Como Executar

### 1. Pré-requisitos

- Python **3.9** ou superior
- `pip` atualizado

### 2. Instalar as Dependências

```bash
pip install -r requirements.txt
```

> ⚠️ O modelo CLIP (≈ 600 MB) será baixado automaticamente do Hugging Face na primeira execução do Modo Imagem.

### 3. Treinar o Modelo de ML (obrigatório antes de usar o Modo Morfométrico)

```bash
python train_model.py
```

Isso irá gerar os arquivos:
- `penguin_species_predictor_model.pkl`
- `model_features.pkl`
- `model_classes.pkl`

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
| `scikit-learn` | Modelo de Regressão Logística |
| `matplotlib` | Gráficos e visualizações |
| `seaborn` | Dataset Palmer Penguins e plots |
| `joblib` | Serialização do modelo |
| `numpy` | Operações numéricas |
| `Pillow` | Processamento de imagens |
| `torch` | Backend PyTorch para o CLIP |
| `transformers` | Modelo CLIP da Hugging Face |

---

## 🗂️ Estrutura do Projeto

```
ClubPenguin/
├── app.py                              # Aplicação principal Streamlit
├── train_model.py                      # Script de treinamento do modelo ML
├── penguin_species_predictor_model.pkl # Modelo treinado (gerado pelo train_model.py)
├── model_features.pkl                  # Lista de features do modelo
├── model_classes.pkl                   # Classes de espécies do modelo
├── requirements.txt                    # Dependências do projeto
├── .gitignore                          # Arquivos ignorados pelo Git
└── README.md                           # Este arquivo
```

---

## 🧠 Sobre os Modelos

### Modo 1 — Regressão Logística (Morfometria)
- **Dataset:** Palmer Penguins (333 amostras após limpeza)
- **Features:** `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `island`, `sex`
- **Pipeline:** `StandardScaler` + `OneHotEncoder` + `LogisticRegression`
- **Split:** 80% treino / 20% teste
- **Acurácia esperada:** ~98%

### Modo 2 — CLIP (Visão Computacional)
- **Modelo:** `openai/clip-vit-base-patch32` via Hugging Face Transformers
- **Método:** Zero-shot classification — sem fine-tuning necessário
- **Espécies suportadas:** Adelie, Chinstrap, Gentoo

---

## 🐧 Espécies Classificadas

| Espécie | Características |
|---|---|
| **Adelie** | Bico curto e robusto, predominante na ilha Torgersen |
| **Chinstrap** | Bico fino, linha preta característica sob o queixo |
| **Gentoo** | Maior porte, nadadeiras longas, predominante em Biscoe |

---

## 🔭 Roadmap — O que ainda falta fazer

- [ ] **`requirements.txt`** — Criar o arquivo de dependências com versões fixadas
- [ ] **Testes unitários** — Adicionar testes para `train_model.py` e para as funções de predição
- [ ] **`LICENSE`** — Adicionar arquivo de licença (MIT recomendado)
- [ ] **Deploy online** — Publicar no Streamlit Community Cloud para acesso via URL pública
- [ ] **Cache do CLIP** — Persistir o modelo CLIP localmente para evitar redownload
- [ ] **Suporte a mais espécies** — Expandir o classificador para além das 3 espécies do Palmer dataset
- [ ] **Página "Sobre"** — Adicionar aba com informações sobre o projeto e o autor
- [ ] **Métricas de avaliação** — Exibir matriz de confusão e relatório de classificação na interface
- [ ] **Modo comparação** — Permitir comparar dois espécimes lado a lado
- [ ] **Exportar resultado** — Botão para baixar o resultado do diagnóstico em PDF/CSV

---

## 👤 Autor

**Maycon Garcia Silva**

[![GitHub](https://img.shields.io/badge/GitHub-Maycon--bd-181717?style=flat-square&logo=github)](https://github.com/Maycon-bd)

---

## 📄 Licença

Este projeto está sob a licença **MIT**. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

> *"Not all who wander are lost — some are just classifying penguins."* 🐧
