import pandas as pd 

pokemon = ["Bulbasaur", "Ivysaur", "Venusaur", "Charmander", "Charmeleon", "Charizard", "Squirtle", "Wartortle", "Blastoise"]
series = pd.Series(pokemon)
print(series[series != "Squirtle"])