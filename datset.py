import pandas as pd
import numpy as np

# dtype=str evita que o GAME_ID perca os zeros à esquerda
df = pd.read_csv('dados\dados_treinamento.csv', parse_dates=['GAME_DATE'],
                 dtype={'GAME_ID': str, 'SEASON_ID': str})

# --- 1. Pontos sofridos (PTS do adversário no mesmo jogo)
opp = df[['GAME_ID', 'TEAM_ID', 'PTS']].rename(columns={'TEAM_ID': 'OPP_ID', 'PTS': 'PTS_ALLOWED'})
df = df.merge(opp, on='GAME_ID')
df = df[df['TEAM_ID'] != df['OPP_ID']]
df = df.sort_values(['TEAM_ID', 'GAME_DATE', 'GAME_ID']).reset_index(drop=True)

# --- 2. Forma recente: média dos N jogos ANTERIORES (shift evita vazamento)
J = 10
cols = ['PTS', 'PTS_ALLOWED', 'FGM', 'FGA', 'FG3M', 'FG3A', 'REB', 'AST', 'STL', 'BLK', 'TOV']
roll = (df.groupby('TEAM_ID')[cols]
          .transform(lambda x: x.shift(1).rolling(J, min_periods=5).mean()))
roll.columns = [f'{c}_R' for c in cols]
df = pd.concat([df, roll], axis=1)

# percentuais recalculados a partir de acertos/tentativas (resolve o NaN do FT_PCT também)
df['FG_PCT_R']  = df['FGM_R'] / df['FGA_R']
df['FG3_PCT_R'] = df['FG3M_R'] / df['FG3A_R']

# --- 3. Aproveitamento na temporada até o jogo anterior
g = df.groupby(['TEAM_ID', 'SEASON_ID'])
jogos_ant = g.cumcount()
vit_ant = g['WL'].cumsum() - df['WL']
df['WIN_PCT'] = (vit_ant / jogos_ant).fillna(0.5)

# --- 4. Descanso (a data do jogo atual já é conhecida antes dele)
df['DESCANSO'] = df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.clip(upper=7).fillna(7)

# --- 5. Uma linha por jogo (mandante x visitante)
base = ['GAME_ID', 'TEAM_ID', 'PTS', 'WL', 'SEASON_ID', 'GAME_DATE']
feat = ['PTS_R', 'PTS_ALLOWED_R', 'FG_PCT_R', 'FG3_PCT_R', 'FG3A_R',
        'REB_R', 'AST_R', 'STL_R', 'BLK_R', 'TOV_R', 'WIN_PCT', 'DESCANSO']

casa = df[df['IS_HOME'] == 1][base + feat]
fora = df[df['IS_HOME'] == 0][base + feat]
jogos = casa.merge(fora, on='GAME_ID', suffixes=('_H', '_A'))
jogos = jogos.rename(columns={'SEASON_ID_H': 'SEASON_ID', 'GAME_DATE_H': 'GAME_DATE', 'WL_H': 'HOME_WIN'})
jogos = jogos.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)

# --- 6. Elo (adaptado do seu amigo)
def calcular_elo(j, k=20, vantagem=100, ini=1500):
    elo, ultima_temp = {}, {}
    pre_h, pre_a = [], []
    for r in j.itertuples():
        h, a = r.TEAM_ID_H, r.TEAM_ID_A
        for t in (h, a):  # regressão à média na virada de temporada
            if t in ultima_temp and ultima_temp[t] != r.SEASON_ID:
                elo[t] = 0.75 * elo[t] + 0.25 * ini
            ultima_temp[t] = r.SEASON_ID
        eh, ea = elo.get(h, ini), elo.get(a, ini)
        pre_h.append(eh); pre_a.append(ea)

        esperado = 1 / (10 ** (-((eh + vantagem) - ea) / 400) + 1)
        margem = abs(r.PTS_H - r.PTS_A)
        mult = (margem + 3) ** 0.8 / 7.5
        delta = k * mult * (r.HOME_WIN - esperado)
        elo[h], elo[a] = eh + delta, ea - delta
    return pre_h, pre_a

jogos['ELO_H'], jogos['ELO_A'] = calcular_elo(jogos)

# --- 7. Diferenças (mandante - visitante) como features finais
for f in feat + ['ELO']:
    jogos[f'DIF_{f}'] = jogos[f'{f}_H'] - jogos[f'{f}_A']

FEATURES = [c for c in jogos.columns if c.startswith('DIF_')] + ['ELO_H', 'ELO_A']
dataset = jogos.dropna(subset=FEATURES).reset_index(drop=True)
dataset[['GAME_ID', 'SEASON_ID', 'GAME_DATE', 'HOME_WIN'] + FEATURES].to_csv('dados\dataset_features.csv', index=False)

print(dataset['HOME_WIN'].mean())  # baseline: % de vitórias do mandante