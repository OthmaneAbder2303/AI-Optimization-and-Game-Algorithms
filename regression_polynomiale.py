import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def f(x):
    return 12*(np.sin(x)/x + np.finfo(np.float32).eps)

x = np.linspace(-3, 10, 100)
y = f(x) + np.random.normal(0, 1, size=100)

plt.scatter(x, y, color='blue', label='Données bruyantes')
plt.plot(x, f(x), color='red', label='Fonction réelle f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Données bruyantes et fonction réelle')
plt.legend()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

for d in range(1,10):
    poly_features = PolynomialFeatures(degree=d)
    X_train_poly = poly_features.fit_transform(x_train.reshape(-1, 1))
    X_test_poly = poly_features.transform(x_test.reshape(-1, 1))

    model = Ridge(alpha=1.0)
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f'Degree: {d}, Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')

    x_range = np.linspace(-3, 10, 200)
    X_range_poly = poly_features.transform(x_range.reshape(-1, 1))
    y_range_pred = model.predict(X_range_poly)

    plt.scatter(x, y, color='blue', label='Données bruyantes')
    plt.plot(x_range, f(x_range), color='red', label='Fonction réelle f(x)')
    plt.plot(x_range, y_range_pred, color='green', label=f'Prédiction degré {d}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Régression Polynomiale de degré {d}')
    plt.legend()
    plt.show()

