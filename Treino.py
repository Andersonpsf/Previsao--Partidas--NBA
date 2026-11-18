import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ds = pd.read_csv("dados\dados_processados.csv")
FEATURES = [c for c in ds.columns if c.startswith(("CASA_", "FORA_", "DIF_"))
            and c not in ("CASA", "FORA")]

treino = ds[ds["TEMPORADA"] != "2024-25"]
teste  = ds[ds["TEMPORADA"] == "2024-25"]
X_train, y_train = treino[FEATURES], treino["VITORIA_CASA"]
X_test,  y_test  = teste[FEATURES],  teste["VITORIA_CASA"]
print(len(treino), "treino |", len(teste), "teste")

# Baseline: sempre chutar o mandante
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("Baseline:", round(accuracy_score(y_test, base.predict(X_test)), 3))

# Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print("Random Forest:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred, target_names=["Visitante", "Mandante"]))
print(confusion_matrix(y_test, pred))

# Quais features pesam mais
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp.head(10))