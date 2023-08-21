import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, roc_auc_score


data = pd.read_csv(r'C:\Users\HOME\Desktop\EI exposome\Projets_CS\data\new_data.csv')  
print(data)

cols_to_drop = [col for col in data.columns if 'string' in col]
data = data.drop(cols_to_drop, axis=1)
numeric_columns = data.select_dtypes(include='number').columns.tolist()
data = data[numeric_columns]

X = data.drop('hs_asthma', axis=1) 

#data=pd.read_excel('dataset_vf.xlsx')


X=pd.read_excel(r'C:\Users\HOME\Desktop\EI exposome\Projets_CS\data\data_compo3.xlsx')
y = data['hs_asthma']

y=np.where(y<0, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # ajustez la taille du jeu de test selon vos besoins

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Exactitude (accuracy) du modèle : {:.2f}".format(accuracy))

from sklearn.metrics import f1_score, confusion_matrix

# Calcul du F1-score
f1 = f1_score(y_test, y_pred)

# Calcul de la matrice de confusion
confusion = confusion_matrix(y_test, y_pred)

# Affichage du F1-score
print("F1-score:", f1)

# Affichage de la matrice de confusion
print("Matrice de confusion:")
print(confusion)


y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculer le taux de faux positifs (FPR), le taux de vrais positifs (TPR) et les seuils
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculer l'aire sous la courbe ROC (AUC-ROC)
auc_score = roc_auc_score(y_test, y_pred_proba)

# Tracer la courbe ROC
plt.plot(fpr, tpr, label='Courbe ROC (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--')  # Ligne en pointillés représentant le hasard
plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC')
plt.legend(loc='lower right')
plt.show()
