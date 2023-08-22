
import numpy as np
from sklearn.utils import resample
from logistic_regression import model,X_test, X_train, y_train, y_test
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


n_bootstrap = 100  # Nombre de bootstrap
n_iterations = 10  # Nombre d'itérations

scores = []  # Liste pour stocker les scores de chaque itération

for _ in range(n_iterations):
    bootstrap_scores = []  # Liste pour stocker les scores bootstrap
    
    for _ in range(n_bootstrap):
        # Créez un échantillon bootstrap
        X_boot, y_boot = resample(X_train, y_train, random_state=42)
        
        # Créez et entraînez votre modèle sur l'échantillon bootstrap
        
        model.fit(X_boot, y_boot)
        
        # Faites des prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)
        
        # Calculez le score de précision et ajoutez-le à la liste bootstrap_scores
        bootstrap_scores.append(accuracy_score(y_test, y_pred))
    
    # Calculez le score moyen des scores bootstrap et ajoutez-le à la liste scores
    scores.append(np.mean(bootstrap_scores))


import matplotlib.pyplot as plt

# Tracer l'histogramme des scores
plt.hist(scores, bins=10, edgecolor='black')
plt.xlabel("Scores")
plt.ylabel("Fréquence")
plt.title("Distribution des scores")
plt.show()
