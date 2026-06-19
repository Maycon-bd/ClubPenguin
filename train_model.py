# train_model.py
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("--- Iniciando Treinamento e Salvamento do Modelo ---\n")

# 1. Carregar o Dataset (Palmer Penguins)
try:
    df = sns.load_dataset('penguins')
    print("Dataset 'penguins' carregado com sucesso via seaborn.")
except Exception as e:
    print(f"Erro ao carregar o dataset via seaborn: {e}")
    print("Tentando carregar do CSV local ('penguins.csv').")
    try:
        df = pd.read_csv('penguins.csv')
    except Exception as e_local:
        print(f"Erro ao carregar CSV local: {e_local}")
        print("Baixando do link do GitHub...")
        url = "https://github.com/allisonhorst/palmerpenguins/raw/master/inst/extdata/penguins.csv"
        df = pd.read_csv(url)
        df.to_csv('penguins.csv', index=False)
        print("Dataset baixado e salvo localmente como 'penguins.csv'.")

# 2. Pré-processamento e Limpeza de Dados
print("2. Pré-processando e limpando dados...")
# Remover linhas com valores ausentes para simplificar
df.dropna(inplace=True)
# Remover a coluna 'sex' com valor '.' se houver (ocorre em alguns datasets antigos)
if 'sex' in df.columns and '.' in df['sex'].unique():
    df = df[df['sex'] != '.']

print(f"    Dados após limpeza: {df.shape[0]} linhas, {df.shape[1]} colunas.\n")

# 3. Definição das Features (X) e Variável Alvo (y)
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'island', 'sex']
target = 'species'

X = df[features]
y = df[target]

print(f"    Features selecionadas: {features}")
print(f"    Variável alvo: {target}\n")

# 4. Divisão dos Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"    Dados divididos: Treino ({X_train.shape[0]} amostras), Teste ({X_test.shape[0]} amostras).\n")

# 5. Pipeline de Pré-processamento e Modelo
numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
categorical_features = ['island', 'sex']

# Criando o pré-processador:
# - StandardScaler para as features numéricas
# - OneHotEncoder para as features categóricas (cria colunas binárias)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Criando o Pipeline: Pré-processador + Modelo de Regressão Logística
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

print("    Pipeline de pré-processamento e modelo configurado.\n")

# 6. Treinamento do Modelo
print("6. Treinando o modelo...")
model.fit(X_train, y_train)
print("    Modelo treinado com sucesso!\n")

# 7. Avaliação do Modelo
print("7. Avaliando o modelo no conjunto de teste...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"    Acurácia do modelo: {accuracy:.2f}\n")
print("    Relatório de Classificação:\n")
print(classification_report(y_test, y_pred))
print("\n    Modelo está pronto para ser salvo.\n")

# 8. Salvando o Modelo e o Pré-processador
# Salvar o pipeline completo
joblib.dump(model, 'penguin_species_predictor_model.pkl')
# Salvar as features e a lista de espécies
joblib.dump(features, 'model_features.pkl')
joblib.dump(model.classes_, 'model_classes.pkl')

print("8. Modelo, features e classes salvos com sucesso.")
print("--- Treinamento e Salvamento Concluídos ---")
