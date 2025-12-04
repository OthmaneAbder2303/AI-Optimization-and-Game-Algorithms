import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('possum.csv')
print(data.shape)
print(data.head())

data.dropna(inplace=True)

plt.scatter(data['footlgth'], data['earconch'], c='blue', label='Données réelles')
plt.xlabel('Longueur d\'empreinte')
plt.ylabel('Taille pavillon d\'oreille')
plt.title('Données de Possums')
plt.legend()
plt.show()
#print(data.shape)

def predict_model(alpha, beta, x):
    return alpha * x + beta

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

x = data['footlgth'].values
y = data['earconch'].values

y_pred_initial = predict_model(0, 0, x)
initial_mse = mse(y, y_pred_initial)
print(f'MSE initial (alpha=0, beta=0): {initial_mse}')
alpha = 0.75
beta = 1
y_pred = predict_model(alpha, beta, x)
current_mse = mse(y, y_pred)
print(f'MSE avec alpha={alpha}, beta={beta}: {current_mse}')

plt.scatter(x, y, c='blue', label='Données réelles')
plt.plot(x, y_pred, color='red', label='Modèle de régression linéaire')

alpha_optimal = (x - np.mean(x)) @ (y - np.mean(y)) / np.sum((x - np.mean(x)) ** 2)
beta_optimal = np.mean(y) - alpha_optimal * np.mean(x)
y_pred_optimal = predict_model(alpha_optimal, beta_optimal, x)
optimal_mse = mse(y, y_pred_optimal)
print(f'Alpha optimal: {alpha_optimal}, Beta optimal: {beta_optimal}')
print(f'MSE optimal: {optimal_mse}')  

plt.plot(x, y_pred_optimal, color='green', label='Modèle optimal de régression linéaire')


plt.xlabel('Longueur d\'empreinte')
plt.ylabel('Taille pavillon d\'oreille')
plt.title('Régression Linéaire sur les Données de Possums')
plt.legend()
plt.show()



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

linear_model = LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(
    data[['footlgth']], data['earconch'], test_size=0.2, random_state=42
)
linear_model.fit(x_train, y_train)
y_pred_sklearn = linear_model.predict(x_test)
sklearn_mse = mse(y_test.values, y_pred_sklearn)
print(f'MSE avec LinearRegression de scikit-learn: {sklearn_mse}')
print(linear_model.coef_, linear_model.intercept_)
print(linear_model.score(x_test, y_test))

plt.scatter(x_test, y_test, c='blue', label='Données réelles')
plt.plot(x_test, y_pred_sklearn, color='orange', label='Modèle LinearRegression de scikit-learn')
plt.xlabel('Longueur d\'empreinte')
plt.ylabel('Taille pavillon d\'oreille')
plt.title('Régression Linéaire avec scikit-learn')
plt.legend()
plt.show()