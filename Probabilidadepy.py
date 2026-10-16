import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. Coleta de Dados via NBA_API
# ---------------------------------------------------------
print('Coletando dados da NBA...')
# Busca jogos da temporada regular 2023-24 (league_id='00' indica a NBA)
gamefinder = leaguegamefinder.LeagueGameFinder(
    season_nullable='2023-24', league_id_nullable='00'
)
df_games = gamefinder.get_data_frames()[0]

# Filtrar apenas jogos de Temporada Regular (IDs iniciados em '002')
df_games = df_games[df_games['GAME_ID'].str.startswith('002')].copy()

# ---------------------------------------------------------
# 2. Tratamento de Dados e Engenharia de Atributos
# ---------------------------------------------------------
# Converte data e ordena por time e cronologia
df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'])
df_games = df_games.sort_values(by=['TEAM_ID', 'GAME_DATE'])

# Variável Alvo (Target): 1 para Vitória (W), 0 para Derrota (L)
df_games['TARGET_WIN'] = (df_games['WL'] == 'W').astype(int)

# Seleção das estatísticas principais da partida
base_stats = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 'TOV']

# Criação de médias móveis dos últimos 100 jogos (sem incluir a partida atual)
for stat in base_stats:
  df_games[f'ROLL_100_{stat}'] = df_games.groupby('TEAM_ID')[stat].transform(
      lambda x: x.shift(1).rolling(window=100, min_periods=3).mean()
  )

feature_cols = [f'ROLL_100_{stat}' for stat in base_stats]

# Remoção de valores nulos gerados pelas primeiras rodadas
df_clean = df_games.dropna(subset=feature_cols + ['TARGET_WIN']).copy()

# ---------------------------------------------------------
# 3. Divisão de Dados e Treinamento do Random Forest
# ---------------------------------------------------------
X = df_clean[feature_cols]
y = df_clean['TARGET_WIN']

# Separar 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Inicializar e treinar o modelo
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. Avaliação e Extração dos Resultados
# ---------------------------------------------------------
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('\n' + '=' * 40)
print(f'ACURÁCIA DO MODELO: {accuracy * 100:.2f}%')
print('=' * 40 + '\n')

# Importância dos Atributos
feature_importances = pd.Series(
    rf.feature_importances_, index=feature_cols
).sort_values(ascending=False)

print('--- IMPORTÂNCIA DOS ATRIBUTOS (FEATURE IMPORTANCE) ---')
for feature, val in feature_importances.items():
  print(f'{feature}: {val * 100:.2f}%')