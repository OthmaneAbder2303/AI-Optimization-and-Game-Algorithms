import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

# Chargement des données
df = pd.read_csv('possum.csv')

# Nettoyage des données
# Suppression des lignes avec des valeurs manquantes
df_clean = df.dropna()

# Sélection des colonnes d'intérêt
data = df_clean[['footlgth', 'earconch']]

# Affichage des données
plt.figure(figsize=(10, 6))
plt.scatter(data['footlgth'], data['earconch'], alpha=0.7)
plt.xlabel('Longueur d\'empreinte')
plt.ylabel('Taille pavillon de l\'oreille')
plt.title('Relation entre la longueur d\'empreinte et la taille du pavillon de l\'oreille')
plt.grid(True, alpha=0.3)
plt.show()

# Préparation des données pour la régression
X = data['footlgth'].values
y = data['earconch'].values