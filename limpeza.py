import main
df_limpeza= main.df_treinamento
print("Hora de Limpar os dados do DataFrame de treinamento:")
print("Dimensões:", df_limpeza.shape)
print("\nColunas:")
print(df_limpeza.columns.tolist())

print("\nTipos de dados:")
print(df_limpeza.dtypes)

print("\nValores ausentes:")
print(df_limpeza.isna().sum())

print("\nLinhas totalmente duplicadas:")
print(df_limpeza.duplicated().sum())

print("\nResultados:")
print(df_limpeza['WL'].value_counts(dropna=False))

print("\nPartidas distintas:")
print(df_limpeza['GAME_ID'].nunique())

print(
    df_limpeza[df_limpeza['FT_PCT'].isna()]
)

print(
    df_limpeza.loc[
        df_limpeza['FT_PCT'].isna(),
        ['GAME_ID', 'TEAM_NAME', 'FTM', 'FTA', 'FT_PCT']
    ]
)