from nba_api.stats.static.teams import find_team_by_abbreviation, get_teams, teams
from nba_api.stats.endpoints import leaguegamefinder
import main
import pandas as pd
teams =get_teams()
def times():
    print("Escolha um time da NBA:")
    for i in teams:
        print(i['full_name'], " - ", i['abbreviation'])
    time_Escolhido = input("Digite a abreviação do time escolhido: ").upper()
    time_Escolhido2 = input("Digite a abreviação do time escolhido: ").upper()

    info_time = find_team_by_abbreviation(time_Escolhido)
    info_time2 = find_team_by_abbreviation(time_Escolhido2)
    print(info_time)
    times = [info_time, info_time2]
    for time in times:
        if time is not None:
            print("Informações do time escolhido:")
            print("Nome completo:", time['full_name'])
            print("Abreviação:", time['abbreviation'])
            print("ID do time:", time['id'])
        else:
            print("Time não encontrado.")
    return info_time, info_time2

def calc(info_time, info_time2, def_treinamento):
    games = def_treinamento
    info_gerais = games[games['TEAM_ID'] == info_time['id']]
    info_gerais2 = games[games['TEAM_ID'] == info_time2['id']]
    print("Informações gerais do time escolhido:")
    print(info_gerais)
    print("Informações gerais do segundo time escolhido:D")
    print(info_gerais2)
    print(games.columns)
    print("Shape:")
    print(games.shape)
    print("Cabeça: ")
    print(games.head())


    
    House_matches = info_gerais[info_gerais['MATCHUP'].str.contains('vs.')]
    Away_matches = info_gerais[info_gerais['MATCHUP'].str.contains('@')]
    vitorias_time1 = info_gerais[info_gerais['WL'] == 'W'].shape[0]
    derrotas_time1 = info_gerais[info_gerais['WL'] == 'L'].shape[0]

    print(House_matches)
    print(Away_matches)

    House_matches2 = info_gerais2[info_gerais2['MATCHUP'].str.contains('vs.')].shape[0]
    Away_matches2 = info_gerais2[info_gerais2['MATCHUP'].str.contains('@')].shape[0]
 
    print(House_matches2)
    print(Away_matches2)

    vitorias_time2 = info_gerais2[info_gerais2['WL'] == 'W'].shape[0]
    derrotas_time2 = info_gerais2[info_gerais2['WL'] == 'L'].shape[0]
    print(f'O time {info_time["full_name"]} tem {vitorias_time1} vitórias e {derrotas_time1} derrotas na temporada 2023-24.')
    print(f'O time {info_time2["full_name"]} tem {vitorias_time2} vitórias e {derrotas_time2} derrotas na temporada 2023-24.')

info_time, info_time2 = times()
calc(info_time, info_time2, def_treinamento=main.df_treinamento)

