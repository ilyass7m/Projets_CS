import numpy as np
from sklearn.utils import resample
from logistic_regression import model,X_test, X_train, y_train, y_test
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


# Évaluer la performance du modèle sur l'ensemble de test original
original_score = model.score(X_test, y_test)

# Définir le nombre d'itérations et l'écart type du bruit gaussien
n_iterations = 10
std_dev = 0.1

# Initialiser une liste pour stocker les scores bruités
noisy_scores = []

# Boucle pour évaluer la stabilité avec différents bruits gaussiens
for i in range(n_iterations):
    # Générer le bruit gaussien
    noise = np.random.normal(0, std_dev, X_train.shape)

    # Ajouter le bruit aux données d'entraînement
    X_train_noisy = X_train + noise

    # Ré-entraîner le modèle sur les données bruitées
    model.fit(X_train_noisy, y_train)

    # Évaluer la performance du modèle ré-entraîné sur l'ensemble de test original
    noisy_score = model.score(X_test, y_test)
    noisy_scores.append(noisy_score)

# Créer un graphique des scores bruités par rapport au score original
plt.plot(range(n_iterations), noisy_scores, 'bo-', label='Noisy Scores')
plt.axhline(original_score, color='r', linestyle='--', label='Original Score')
plt.xlabel('Iterations')
plt.ylabel('Model Score')
plt.title('Model Stability')
plt.legend()
plt.show()


