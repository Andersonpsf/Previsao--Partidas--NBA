"""
Previsor de partidas da NBA: Random Forest x Elo, lado a lado.

Entrada: dados_treinamento.csv (gerado pela sua coleta, com a coluna TEMPORADA,
WL já em 0/1 e as 5 temporadas, de 2020-21 a 2024-25).

Fluxo:
  1. Carrega os jogos e corrige o mandante pela sigla do MATCHUP
  2. Calcula o Elo (modelo simples, sem machine learning)
  3. Escolhe a janela de forma recente validando em 2023-24
  4. Avalia Baseline x Elo x Random Forest em 2024-25 (acurácia, log loss,
     Brier, tabela de confiabilidade, concordância entre os modelos)
  5. Treina o Random Forest final com tudo e prevê confrontos escolhidos
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# ------------------------------------------------------------------ CONFIG
BASE = ["PTS", "FG_PCT", "FG3A", "FG3_PCT", "REB", "AST", "STL", "BLK", "TOV"]
ESTATISTICAS = BASE + ["PTS_ALLOWED"]          # PTS_ALLOWED = pontos sofridos

TEMPORADA_TESTE = "2024-25"                    # só para a avaliação final
TEMPORADA_VALIDACAO = "2023-24"                # usada para escolher a janela
JANELAS_CANDIDATAS = (3, 5, 10, 15, 20)

ELO_INICIAL = 1500
ELO_K = 20
ELO_CASA = 100                                 # vantagem de jogar em casa (pontos de Elo)
ELO_REGRESSAO = 0.25                           # quanto o Elo volta à média a cada temporada


# ------------------------------------------------------------------ DADOS
def carregar_jogos(caminho="dados\dados_treinamento.csv"):
    """Uma linha por time por jogo, com IS_HOME corrigido e pontos sofridos."""
    df = pd.read_csv(caminho, dtype={"GAME_ID": str})
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # O mandante é identificado pela sigla no MATCHUP ("A vs. B" -> A; "A @ B" -> B)
    mandante = np.where(
        df["MATCHUP"].str.contains(" vs. ", regex=False),
        df["MATCHUP"].str.split(" vs. ").str[0],
        df["MATCHUP"].str.split(" @ ").str[1],
    )
    df["IS_HOME"] = (df["TEAM_ABBREVIATION"] == mandante).astype(int)

    # Jogos inconsistentes (não têm 2 times e exatamente 1 mandante) são descartados
    chk = df.groupby("GAME_ID").agg(times=("TEAM_ID", "nunique"),
                                    mandantes=("IS_HOME", "sum"),
                                    registros=("TEAM_ID", "size"))
    ruins = chk[(chk.times != 2) | (chk.mandantes != 1) | (chk.registros != 2)].index
    if len(ruins):
        print(f"Aviso: {len(ruins)} jogos inconsistentes descartados.")
    df = df[~df["GAME_ID"].isin(ruins)]

    # Pontos sofridos = pontos do adversário no mesmo jogo
    opp = df[["GAME_ID", "TEAM_ID", "PTS"]].rename(
        columns={"TEAM_ID": "OPP_ID", "PTS": "PTS_ALLOWED"})
    df = df.merge(opp, on="GAME_ID")
    df = df[df["TEAM_ID"] != df["OPP_ID"]]

    return df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def tabela_de_jogos(games):
    """Uma linha por JOGO (mandante x visitante), em ordem cronológica."""
    casa = games[games["IS_HOME"] == 1][
        ["GAME_ID", "TEMPORADA", "GAME_DATE", "TEAM_ID", "TEAM_NAME", "PTS", "WL"]
    ].rename(columns={"TEAM_ID": "CASA_ID", "TEAM_NAME": "CASA",
                      "PTS": "PTS_CASA", "WL": "VITORIA_CASA"})
    fora = games[games["IS_HOME"] == 0][
        ["GAME_ID", "TEAM_ID", "TEAM_NAME", "PTS"]
    ].rename(columns={"TEAM_ID": "FORA_ID", "TEAM_NAME": "FORA", "PTS": "PTS_FORA"})
    jogos = casa.merge(fora, on="GAME_ID", validate="one_to_one")
    return jogos.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


# ------------------------------------------------------------------ MODELO 1: ELO
def prob_elo(elo_casa, elo_fora):
    """Probabilidade de o mandante vencer, só com a fórmula do Elo."""
    return 1 / (1 + 10 ** (-((elo_casa + ELO_CASA) - elo_fora) / 400))


def calcular_elo(jogos):
    """
    Percorre os jogos em ordem. O Elo "pré-jogo" é registrado ANTES de atualizar,
    então não há vazamento. Retorna a tabela com PROB_ELO e os ratings finais.
    """
    elo, ultima_temp = {}, {}
    pre_casa, pre_fora = [], []

    for r in jogos.itertuples():
        c, f = r.CASA_ID, r.FORA_ID
        for t in (c, f):                       # volta parcialmente à média na virada de temporada
            if t in ultima_temp and ultima_temp[t] != r.TEMPORADA:
                elo[t] = (1 - ELO_REGRESSAO) * elo[t] + ELO_REGRESSAO * ELO_INICIAL
            ultima_temp[t] = r.TEMPORADA

        ec, ef = elo.get(c, ELO_INICIAL), elo.get(f, ELO_INICIAL)
        pre_casa.append(ec)
        pre_fora.append(ef)

        esperado = prob_elo(ec, ef)
        margem = abs(r.PTS_CASA - r.PTS_FORA)
        delta = ELO_K * ((margem + 3) ** 0.8 / 7.5) * (r.VITORIA_CASA - esperado)
        elo[c], elo[f] = ec + delta, ef - delta

    jogos = jogos.copy()
    jogos["ELO_CASA"], jogos["ELO_FORA"] = pre_casa, pre_fora
    jogos["PROB_ELO"] = prob_elo(jogos["ELO_CASA"], jogos["ELO_FORA"])
    return jogos, elo


# ------------------------------------------------------------------ MODELO 2: RANDOM FOREST
def montar_dataset(games, jogos_elo, janela):
    """Médias dos `janela` jogos ANTERIORES de cada time + diferenças."""
    g = games.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).copy()
    medias = []
    for c in ESTATISTICAS:
        nome = f"{c}_MEDIA"
        medias.append(nome)
        g[nome] = (g.groupby("TEAM_ID")[c]
                     .transform(lambda s: s.shift(1).rolling(janela, min_periods=janela).mean()))

    casa = g[g["IS_HOME"] == 1][["GAME_ID"] + medias].rename(
        columns={m: f"CASA_{m}" for m in medias})
    fora = g[g["IS_HOME"] == 0][["GAME_ID"] + medias].rename(
        columns={m: f"FORA_{m}" for m in medias})
    ds = jogos_elo.merge(casa, on="GAME_ID").merge(fora, on="GAME_ID")

    for m in medias:
        ds[f"DIF_{m}"] = ds[f"CASA_{m}"] - ds[f"FORA_{m}"]

    # Lista explícita (evita pegar CASA_ID/FORA_ID por engano). O Elo NÃO entra
    # nas features: assim os dois modelos são realmente independentes.
    features = ([f"CASA_{m}" for m in medias] + [f"FORA_{m}" for m in medias]
                + [f"DIF_{m}" for m in medias])
    ds = ds.dropna(subset=features + ["VITORIA_CASA"]).reset_index(drop=True)
    return ds, features


def novo_rf():
    return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
                                  random_state=42, n_jobs=-1)


def avaliar(ds, features, temporadas_treino, temporada_teste):
    """Treina o RF nas temporadas de treino e devolve o teste com as 3 probabilidades."""
    tr = ds[ds["TEMPORADA"].isin(temporadas_treino)]
    te = ds[ds["TEMPORADA"] == temporada_teste].copy()
    rf = novo_rf().fit(tr[features], tr["VITORIA_CASA"])
    te["PROB_RF"] = rf.predict_proba(te[features])[:, 1]
    te["PROB_BASE"] = tr["VITORIA_CASA"].mean()   # baseline: sempre a taxa do mandante
    return te, rf


# ------------------------------------------------------------------ ESCOLHA DA JANELA
def escolher_janela(games, jogos_elo, candidatas=JANELAS_CANDIDATAS):
    """Valida em 2023-24 (treino só nas anteriores). 2024-25 não é tocada aqui."""
    linhas = {}
    for J in candidatas:
        ds, feats = montar_dataset(games, jogos_elo, J)
        temporadas = sorted(ds["TEMPORADA"].unique())
        treino = [t for t in temporadas if t < TEMPORADA_VALIDACAO]
        te, _ = avaliar(ds, feats, treino, TEMPORADA_VALIDACAO)
        y = te["VITORIA_CASA"]
        linhas[J] = {
            "jogos": len(te),
            "Acurácia RF": accuracy_score(y, te["PROB_RF"] >= 0.5),
            "Log loss RF": log_loss(y, te["PROB_RF"]),
            "Log loss Elo": log_loss(y, te["PROB_ELO"]),
        }
    tabela = pd.DataFrame(linhas).T
    print(f"\n=== Escolha da janela (validação em {TEMPORADA_VALIDACAO}) ===")
    print(tabela.round(3))
    melhor = int(tabela["Log loss RF"].idxmin())
    print(f"Janela escolhida (menor log loss do RF): {melhor}")
    return melhor


# ------------------------------------------------------------------ RELATÓRIO
MODELOS = {"Baseline (mandante)": "PROB_BASE", "Elo (simples)": "PROB_ELO",
           "Random Forest": "PROB_RF"}


def confiabilidade(y, p, faixas=(0, 0.4, 0.5, 0.6, 0.7, 1.0)):
    """Em cada faixa de probabilidade prevista, quanto realmente aconteceu."""
    t = pd.DataFrame({"y": y.values, "p": p.values,
                      "faixa": pd.cut(p.values, list(faixas), include_lowest=True)})
    t = t.groupby("faixa", observed=True).agg(
        jogos=("y", "size"), previsto=("p", "mean"), observado=("y", "mean"))
    return t.round(3)


def relatorio(te, png="confiabilidade.png"):
    y = te["VITORIA_CASA"]
    print(f"\n=== Avaliação em {TEMPORADA_TESTE} ({len(te)} jogos) ===")
    linhas = {}
    for nome, col in MODELOS.items():
        p = te[col]
        linhas[nome] = {"Acurácia": accuracy_score(y, p >= 0.5),
                        "Log loss": log_loss(y, p),
                        "Brier": brier_score_loss(y, p)}
    print(pd.DataFrame(linhas).T.round(3))
    print("(acurácia: maior é melhor | log loss e Brier: menor é melhor)")

    print("\n=== Confiabilidade: probabilidade prevista x o que aconteceu ===")
    print("Um modelo confiável tem 'previsto' próximo de 'observado' em cada faixa.")
    print(pd.concat({"Elo": confiabilidade(y, te["PROB_ELO"]),
                     "Random Forest": confiabilidade(y, te["PROB_RF"])}, axis=1))

    print("\n=== Acerto conforme a confiança do modelo ===")
    tab = {}
    for nome in ("Elo (simples)", "Random Forest"):
        p = te[MODELOS[nome]]
        conf = (p - 0.5).abs()
        tab[nome] = {}
        for rotulo, lim in (("todos os jogos", 0), ("confiança >= 10 p.p.", 0.10),
                            ("confiança >= 20 p.p.", 0.20)):
            m = conf >= lim
            if m.sum():
                acc = accuracy_score(y[m], p[m] >= 0.5)
                tab[nome][rotulo] = f"{acc:.1%} ({m.sum()} jogos)"
    print(pd.DataFrame(tab))

    print("\n=== Concordância entre os modelos ===")
    acordo = (te["PROB_RF"] >= 0.5) == (te["PROB_ELO"] >= 0.5)
    print(f"Concordam sobre o vencedor em {acordo.mean():.1%} dos jogos")
    for rotulo, m in (("Quando concordam", acordo), ("Quando discordam", ~acordo)):
        if m.sum():
            print(f"  {rotulo} ({m.sum()} jogos): "
                  f"RF acerta {accuracy_score(y[m], te.loc[m, 'PROB_RF'] >= 0.5):.1%} | "
                  f"Elo acerta {accuracy_score(y[m], te.loc[m, 'PROB_ELO'] >= 0.5):.1%}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5.5, 5.5))
        plt.plot([0, 1], [0, 1], "--", color="gray", label="perfeito")
        for nome, col in (("Elo", "PROB_ELO"), ("Random Forest", "PROB_RF")):
            t = confiabilidade(y, te[col], faixas=(0, .3, .4, .5, .6, .7, 1.0))
            plt.plot(t["previsto"], t["observado"], "o-", label=nome)
        plt.xlabel("Probabilidade prevista (mandante vence)")
        plt.ylabel("Frequência observada")
        plt.title("Curva de confiabilidade")
        plt.legend()
        plt.tight_layout()
        plt.savefig(png, dpi=130)
        plt.close()
        print(f"\nGráfico salvo em {png}")
    except ImportError:
        print("\n(matplotlib não instalado: gráfico não gerado)")


# ------------------------------------------------------------------ PREVISÃO DE UM CONFRONTO
def medias_recentes(games, team_id, janela):
    ultimos = games[games["TEAM_ID"] == team_id].tail(janela)
    return ultimos[ESTATISTICAS].mean()


def prever(info_casa, info_fora, games, rf, features, elo_final, janela):
    m = medias_recentes(games, info_casa["id"], janela)
    v = medias_recentes(games, info_fora["id"], janela)

    linha = {}
    for c in ESTATISTICAS:
        nome = f"{c}_MEDIA"
        linha[f"CASA_{nome}"] = m[c]
        linha[f"FORA_{nome}"] = v[c]
        linha[f"DIF_{nome}"] = m[c] - v[c]
    X = pd.DataFrame([linha])[features]

    p_rf = rf.predict_proba(X)[0, 1]
    p_elo = float(prob_elo(elo_final.get(info_casa["id"], ELO_INICIAL),
                           elo_final.get(info_fora["id"], ELO_INICIAL)))
    p_med = (p_rf + p_elo) / 2

    tabela = pd.DataFrame(
        {"Random Forest": [p_rf, 1 - p_rf], "Elo": [p_elo, 1 - p_elo],
         "Média": [p_med, 1 - p_med]},
        index=[f"{info_casa['full_name']} (casa)", f"{info_fora['full_name']} (fora)"])
    print("\n===== PREVISÃO =====")
    print((tabela * 100).round(1).astype(str) + "%")

    concordam = (p_rf >= 0.5) == (p_elo >= 0.5)
    diferenca = abs(p_rf - p_elo)
    print(f"\nDiferença entre os modelos: {diferenca * 100:.1f} p.p.")
    if concordam and diferenca < 0.10:
        print("Os modelos concordam: previsão mais confiável.")
    elif concordam:
        print("Concordam no vencedor, mas divergem na intensidade da probabilidade.")
    else:
        print("Os modelos DISCORDAM no vencedor: jogo incerto, confie pouco em qualquer um.")
    if max(abs(p_rf - 0.5), abs(p_elo - 0.5)) < 0.05:
        print("Probabilidades próximas de 50%: jogo equilibrado.")


# ------------------------------------------------------------------ INTERFACE
def pedir_time(mensagem):
    from nba_api.stats.static.teams import find_team_by_abbreviation
    while True:
        info = find_team_by_abbreviation(input(mensagem).strip().upper())
        if info is not None:
            return info
        print("Time não encontrado, tente novamente.")


def escolher_times():
    from nba_api.stats.static.teams import get_teams
    print("Times da NBA:")
    for t in get_teams():
        print(t["full_name"], "-", t["abbreviation"])
    casa = pedir_time("Abreviação do time MANDANTE (joga em casa): ")
    fora = pedir_time("Abreviação do time VISITANTE: ")
    return casa, fora


def main():
    games = carregar_jogos()
    print("Último jogo nos dados:", games["GAME_DATE"].max().date())

    jogos = tabela_de_jogos(games)
    jogos_elo, elo_final = calcular_elo(jogos)

    janela = escolher_janela(games, jogos_elo)

    ds, features = montar_dataset(games, jogos_elo, janela)
    temporadas = sorted(ds["TEMPORADA"].unique())
    treino = [t for t in temporadas if t != TEMPORADA_TESTE]
    te, _ = avaliar(ds, features, treino, TEMPORADA_TESTE)
    relatorio(te)

    # Modelo final para uso real: treina com TODAS as temporadas
    rf_final = novo_rf().fit(ds[features], ds["VITORIA_CASA"])

    while True:
        casa, fora = escolher_times()
        prever(casa, fora, games, rf_final, features, elo_final, janela)
        if input("\nPrever outro confronto? (s/n): ").strip().lower() != "s":
            break


if __name__ == "__main__":
    main()