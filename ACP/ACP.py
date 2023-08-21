import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Charger les données dans un DataFrame panda 
data = pd.read_excel('dataset_vf.xlsx')

# se débarasser des colonnes non numériques 
cols_to_drop = [col for col in data.columns if 'string' in col]
data = data.drop(cols_to_drop, axis=1)


numeric_columns = data.select_dtypes(include='number').columns.tolist()
data = data[numeric_columns]
#data=data.drop('ID', axis=1)

# Séparer les variables explicatives (X) et la variable cible (y) si nécessaire
X = data.drop('hs_asthma', axis=1)
#our target !
y = data['hs_asthma'] 


# Standardiser les variables explicatives
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Effectuer l'ACP
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Variance expliquée par chaque composante principale
explained_variance_ratio = pca.explained_variance_ratio_

# Afficher la variance expliquée cumulée
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
print("Variance expliquée cumulée : ", cumulative_variance_ratio)

# le nbre de compos a garder en fct de la variance cumulé
k = len(cumulative_variance_ratio[cumulative_variance_ratio <= 0.70])

# Sélectionner les k premières composantes principales
X_selected = X_pca[:, :k]
data=pd.DataFrame(X_selected)
data.to_excel('data_compo4', engine='xlsxwriter',index=False)

# Affiche les var les plus importantes pour chaque composante principale
feature_names = X.columns
component_names = ['Component {}'.format(i+1) for i in range(k)]

components_df = pd.DataFrame(pca.components_[:k, :], columns=feature_names, index=component_names)
components_df.to_csv('components.csv',index=False)
print(components_df)

# Utiliser les covariables sélectionnées (X_selected) dans votre modèle

# Parcourir chaque composante principale
occ={}
for component in component_names:
    # Sélectionner les variables les plus importantes pour la composante principale
    important_variables = components_df.loc[component].abs().nlargest(3).index
    print("Variables importantes pour", component, ":", important_variables.tolist())
    for element in important_variables:
        if element not in occ:
            occ[element]=1

        else:
            occ[element]+=1

print(occ)

# Obtenir les 5 clés ayant les plus grandes valeurs
top_10_cles = sorted(occ, key=occ.get, reverse=True)[:10]

# Afficher les 5 clés
print("Les 10 facteurs ayant plus d'impact sur l'asthme :", top_10_cles)










