import pandas as pd

# Charger le dataset original
df_components= pd.read_csv(r'C:\Users\HOME\Desktop\EI exposome\Projets_CS\data\components.csv')

print(df_components.shape)


df=pd.read_excel(r'C:\Users\HOME\Desktop\EI exposome\Projets_CS\data\dataset_vf.xlsx')
df=df.drop('Unnamed: 0', axis=1)

print(df.shape)



#print(df_original.shape[0])


#Définir les poids pour chaque colonne
poids = [[ df_components.iloc[i, df_components.columns.get_loc(column)] for column in df_components.columns] for i in range(df_components.shape[0]) ]

# Créer un nouveau DataFrame pour stocker les combinaisons linéaires
df_combinaisons = pd.DataFrame()

# Calculer les combinaisons linéaires pour chaque colonne
for i in range(df.shape[0]):
    for j in range(df_components.shape[0]):
        df_combinaisons.iloc[i,f'composante {j+1}']=poids[j]*df.loc[i]
    


# Afficher le nouveau dataset avec les colonnes transformées
print(df_combinaisons) 