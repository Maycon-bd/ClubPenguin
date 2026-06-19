# train_model.py — MatoScan
# Gera dataset morfométrico sintético e realista de sementes de grãos
# e treina um modelo RandomForestClassifier para classificação.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("--- MatoScan: Iniciando Treinamento do Modelo de Sementes ---\n")

# =============================================================================
# 1. GERAR DATASET SINTÉTICO COM BASE EM PARÂMETROS REAIS DE SEMENTES
# =============================================================================
# Fontes de referência para os ranges morfológicos:
#   Milho  : comprimento 8-12mm, largura 6-10mm, espessura 4-6mm, massa 250-400mg
#   Soja   : comprimento 5-8mm,  largura 5-7mm,  espessura 4-6mm, massa 150-250mg
#   Arroz  : comprimento 6-9mm,  largura 2-3mm,  espessura 1.5-2.5mm, massa 18-35mg
#   Trigo  : comprimento 5-7mm,  largura 3-4mm,  espessura 2.5-3.5mm, massa 30-55mg
#   Feijão : comprimento 12-18mm, largura 7-11mm, espessura 6-8mm, massa 200-450mg

np.random.seed(42)
N_AMOSTRAS = 350  # por cultura

def gerar_sementes(cultura, n, comp_mu, comp_sigma, larg_mu, larg_sigma,
                   esp_mu, esp_sigma, massa_mu, massa_sigma):
    """Gera n amostras de semente com distribuição normal."""
    comprimento   = np.random.normal(comp_mu, comp_sigma, n)
    largura       = np.random.normal(larg_mu, larg_sigma, n)
    espessura     = np.random.normal(esp_mu,  esp_sigma,  n)
    massa         = np.random.normal(massa_mu, massa_sigma, n)

    # Features derivadas — análogas às usadas em visão computacional de sementes
    area          = comprimento * largura * np.pi / 4           # elipse aproximada
    perimetro     = np.pi * (3*(comprimento/2 + largura/2)
                              - np.sqrt((3*comprimento/2 + largura/2)
                                        * (comprimento/2 + 3*largura/2)))  # Ramanujan
    compacidade   = (4 * np.pi * area) / (perimetro ** 2)       # ~1 = circular
    razao_aspecto = comprimento / largura

    df = pd.DataFrame({
        'comprimento_mm': np.clip(comprimento, 1, 30),
        'largura_mm':     np.clip(largura,     1, 20),
        'espessura_mm':   np.clip(espessura,   0.5, 15),
        'massa_mg':       np.clip(massa,       5, 700),
        'area_mm2':       np.clip(area,        2, 250),
        'perimetro_mm':   np.clip(perimetro,   5, 80),
        'compacidade':    np.clip(compacidade, 0.1, 1.0),
        'razao_aspecto':  np.clip(razao_aspecto, 1, 6),
        'cultura': cultura
    })
    return df

# Parâmetros: (mu, sigma) para cada feature de cada cultura
culturas = pd.concat([
    gerar_sementes('Milho',  N_AMOSTRAS,
                   comp_mu=10.0, comp_sigma=0.8,
                   larg_mu=8.0,  larg_sigma=0.7,
                   esp_mu=5.0,   esp_sigma=0.5,
                   massa_mu=320, massa_sigma=35),
    gerar_sementes('Soja',   N_AMOSTRAS,
                   comp_mu=6.5,  comp_sigma=0.6,
                   larg_mu=6.0,  larg_sigma=0.5,
                   esp_mu=5.0,   esp_sigma=0.4,
                   massa_mu=195, massa_sigma=25),
    gerar_sementes('Arroz',  N_AMOSTRAS,
                   comp_mu=7.5,  comp_sigma=0.5,
                   larg_mu=2.4,  larg_sigma=0.2,
                   esp_mu=2.0,   esp_sigma=0.2,
                   massa_mu=26,  massa_sigma=4),
    gerar_sementes('Trigo',  N_AMOSTRAS,
                   comp_mu=6.0,  comp_sigma=0.5,
                   larg_mu=3.5,  larg_sigma=0.3,
                   esp_mu=3.0,   esp_sigma=0.25,
                   massa_mu=42,  massa_sigma=8),
    gerar_sementes('Feijao', N_AMOSTRAS,
                   comp_mu=15.0, comp_sigma=1.5,
                   larg_mu=9.0,  larg_sigma=1.0,
                   esp_mu=7.0,   esp_sigma=0.7,
                   massa_mu=320, massa_sigma=60),
], ignore_index=True)

culturas = culturas.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"1. Dataset gerado: {culturas.shape[0]} amostras | {culturas['cultura'].value_counts().to_dict()}\n")

# Salvar CSV para uso posterior no app
culturas.to_csv('seeds_dataset.csv', index=False)
print("   Dataset salvo em 'seeds_dataset.csv'\n")

# =============================================================================
# 2. PREPARAR FEATURES E ALVO
# =============================================================================
features = ['comprimento_mm', 'largura_mm', 'espessura_mm', 'massa_mg',
            'area_mm2', 'perimetro_mm', 'compacidade', 'razao_aspecto']
target   = 'cultura'

X = culturas[features]
y = culturas[target]

print(f"2. Features: {features}")
print(f"   Culturas: {sorted(y.unique())}\n")

# =============================================================================
# 3. DIVISÃO TREINO / TESTE
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"3. Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras\n")

# =============================================================================
# 4. PIPELINE: StandardScaler + RandomForestClassifier
# =============================================================================
model = Pipeline(steps=[
    ('scaler',     StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

print("4. Treinando RandomForest (200 árvores)...")
model.fit(X_train, y_train)
print("   Modelo treinado!\n")

# =============================================================================
# 5. AVALIAÇÃO
# =============================================================================
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"5. Acurácia no conjunto de teste: {accuracy:.4f} ({accuracy*100:.1f}%)\n")
print("   Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# =============================================================================
# 6. SALVAR ARTEFATOS
# =============================================================================
joblib.dump(model,    'seed_classifier_model.pkl')
joblib.dump(features, 'model_features.pkl')
joblib.dump(sorted(y.unique()), 'model_classes.pkl')

print("6. Artefatos salvos:")
print("   - seed_classifier_model.pkl")
print("   - model_features.pkl")
print("   - model_classes.pkl")
print("\n--- Treinamento Concluído com Sucesso! ---")
