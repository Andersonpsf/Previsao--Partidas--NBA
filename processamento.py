import pandas as pd
import numpy as np

df = pd.read_csv("dados\dados_treinamento.csv", dtype={'GAME_ID': str})
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

# --- Corrige IS_HOME usando a sigla do mandante contida no MATCHUP
mandante_abrev = np.where(
    df["MATCHUP"].str.contains(" vs. ", regex=False),
    df["MATCHUP"].str.split(" vs. ").str[0],
    df["MATCHUP"].str.split(" @ ").str[1],
)
df["IS_HOME"] = (df["TEAM_ABBREVIATION"] == mandante_abrev).astype(int)

# --- Checagens: cada jogo com 2 times e exatamente 1 mandante
assert not df.duplicated(["GAME_ID", "TEAM_ID"]).any()
chk = df.groupby("GAME_ID").agg(times=("TEAM_ID", "nunique"),
                                mandantes=("IS_HOME", "sum"),
                                registros=("TEAM_ID", "size"))
problemas = chk[(chk.times != 2) | (chk.mandantes != 1) | (chk.registros != 2)]
print("Jogos ainda problemáticos:", len(problemas))
print(df[df["GAME_ID"].isin(["22400147", "22400621", "22400633"])]
        [["GAME_ID", "TEAM_ABBREVIATION", "MATCHUP", "IS_HOME", "WL"]])

# Se sobrar algum jogo problemático, aí sim descarte só esses jogos (os dois lados)
df = df[~df["GAME_ID"].isin(problemas.index)]

# --- Médias dos 5 jogos ANTERIORES
stats = ["PTS", "FG_PCT", "FG3A", "FG3_PCT", "REB", "AST", "STL", "BLK", "TOV"]
df = df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

medias = []
for c in stats:
    nome = f"{c}_MEDIA_5J"
    medias.append(nome)
    df[nome] = (df.groupby("TEAM_ID")[c]
                  .transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean()))

# --- Uma linha por jogo
casa = df[df["IS_HOME"] == 1][["GAME_ID", "TEMPORADA", "GAME_DATE", "TEAM_NAME", "WL"] + medias]
fora = df[df["IS_HOME"] == 0][["GAME_ID", "TEAM_NAME"] + medias]

casa = casa.rename(columns={"TEAM_NAME": "CASA", "WL": "VITORIA_CASA",
                            **{m: f"CASA_{m}" for m in medias}})
fora = fora.rename(columns={"TEAM_NAME": "FORA", **{m: f"FORA_{m}" for m in medias}})

partidas = casa.merge(fora, on="GAME_ID", validate="one_to_one")

# --- Diferenças (mandante - visitante)
for m in medias:
    partidas[f"DIF_{m}"] = partidas[f"CASA_{m}"] - partidas[f"FORA_{m}"]

FEATURES = ([f"CASA_{m}" for m in medias] + [f"FORA_{m}" for m in medias]
            + [f"DIF_{m}" for m in medias])

partidas = partidas.dropna(subset=FEATURES + ["VITORIA_CASA"]).reset_index(drop=True)
partidas.to_csv("dados\dados_processados.csv", index=False)

print(partidas.shape)
print(partidas["TEMPORADA"].value_counts().sort_index())
print("% vitória do mandante:", partidas["VITORIA_CASA"].mean())