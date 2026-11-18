import time

from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static.teams import find_team_by_abbreviation, get_teams
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints.scoreboardv2 import ScoreboardV2
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

# # Nikola Jokić
# career = playercareerstats.PlayerCareerStats(player_id='203999')

# Find team by abbreviation
# get_team = find_team_by_abbreviation('DEN')



teams =get_teams()

# series = pd.Series([i['full_name'] for i in teams], index=[i['abbreviation'] for i in teams])
# for i in teams:
#     print(i['full_name'], i['abbreviation'])

# if find_team_by_abbreviation("AVA"):
#     print(find_team_by_abbreviation("SAS"))
# else: 
#     print("Team not found")

# pandas data frames (optional: pip install pandas)
# career.season_totals_regular_season.get_data_frame()
# print(series)

series = pd.Series(teams)
print("Time Escolhido: " + series.iloc[1]['full_name'])

df = pd.DataFrame(teams)
print(df)
print("================================")
# # json
# career.get_json()



# # json
# career.get_json()


# Today's Score Board
# games = scoreboard.ScoreBoard()

# # json
# games.get_json()

# # dictionary
# games.get_dict()
# print(games.get_dict())
# board = ScoreboardV2()

# # 2. Extrai os dados no formato de dicionário
# board_dict = board.get_dict()

# # 3. Isola apenas a lista de jogos dentro do dicionário
# # A estrutura de resposta da API agrupa os jogos neste caminho:
# lista_de_jogos = board_dict['scoreboard']['games']

# # 4. Cria o DataFrame passando a lista de dicionários (os jogos)
# df_games = pd.DataFrame(lista_de_jogos)

# # Exibe o resultado
# print(df_games)

temporadas_usadas = [
    '2020-21',
    '2021-22',
    '2022-23',
    '2023-24'
]

temporada_teste = ['2024-25']


df_temporadas_treinamento = []
for temporada in temporadas_usadas + temporada_teste:
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=temporada, league_id_nullable='00', season_type_nullable='Regular Season')
    games = gamefinder.get_data_frames()[0]
    games['TEMPORADA'] = temporada
    df_temporadas_treinamento.append(games)
    print(f"Temporada: {temporada}")
    time.sleep(1)  # Pausa de 1 segundo entre as chamadas para evitar sobrecarga na API
df_treinamento = pd.concat(df_temporadas_treinamento, ignore_index=True)
print(df_treinamento)
print(df_treinamento['TEMPORADA'].value_counts())

group = df_treinamento.groupby('TEAM_NAME')
print("================================")
print(group["WL"].value_counts())
print("================================")
print(group["WL"].count())
print("================================")


df_treinamento['WL'] = (df_treinamento['WL']=='W').astype(int)
df_treinamento['IS_HOME'] = df_treinamento['MATCHUP'].str.contains('vs.').astype(int)
df_treinamento.to_csv('dados\dados_treinamento.csv', index=False)
print("Dados de treinamento salvos em 'dados\dados_treinamento.csv'.")
print(df_treinamento)
print(df_treinamento['IS_HOME'])

infos = [
    'TEAM_ID',
    'GAME_ID',
    'TEAM_NAME',
    'GAME_DATE',
    'MATCHUP',
    'WL',
    'PTS',
    'IS_HOME',
]

print(df_treinamento[infos].head(10))

# Estatísticas que vamos usar para calcular o desempenho recente
estatisticas = [
    'PTS',
    'FG_PCT',
    'FG3A',
    'FG3_PCT',
    'REB',
    'AST',
    'STL',
    'BLK',
    'TOV'
]
# print(df_treinamento)
# Busca jogos das últimas temporadas
# gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2023-24', league_id_nullable='00')
# print(gamefinder.get_json())
# games = gamefinder.get_data_frames()[0].set_index('TEAM_NAME')
# print("================================")
# # Filtra apenas jogos de temporada regular / playoffs de times
# print(games)
# dictionary
# career.get_dict()






