import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(12)

print("=== PARTIE 1: Régression linéaire ===")
df = pd.read_csv("possum.csv")
df = df[['footlgth', 'earconch']].dropna()
X = df['footlgth'].values
y = df['earconch'].values

plt.scatter(X, y)
plt.xlabel("Longueur d'empreinte")
plt.ylabel("Taille pavillon de l'oreille")
plt.title("Données initiales")
plt.show()

def predict_model(alpha, beta, X):
    return alpha * X + beta

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

alpha_test = [0.25, 0.75, 1]
beta_test = [0.5, 1, 2]
for a, b in zip(alpha_test, beta_test):
    y_pred = predict_model(a, b, X)
    print(f"Model (alpha={a}, beta={b}): MSE = {mse(y, y_pred):.3f}")

x_bar = np.mean(X)
y_bar = np.mean(y)
alpha_opt = np.sum((X - x_bar) * (y - y_bar)) / np.sum((X - x_bar)**2)
beta_opt = y_bar - alpha_opt * x_bar
print(f"Alpha optimal: {alpha_opt:.4f}, Beta optimal: {beta_opt:.4f}")
y_pred_opt = predict_model(alpha_opt, beta_opt, X)
print(f"MSE optimal: {mse(y, y_pred_opt):.4f}")

X_reshaped = X.reshape(-1, 1)
model_sk = LinearRegression()
model_sk.fit(X_reshaped, y)
y_pred_sk = model_sk.predict(X_reshaped)
mse_sk = mean_squared_error(y, y_pred_sk)
print(f"Scikit-learn MSE: {mse_sk:.4f}")
print(f"Coef: {model_sk.coef_[0]:.4f}, Intercept: {model_sk.intercept_:.4f}")

plt.scatter(X, y, alpha=0.7, label='Données')
plt.plot(X, y_pred_opt, color='red', linewidth=2, label='Modèle optimal')
plt.xlabel("Longueur d'empreinte")
plt.ylabel("Taille pavillon de l'oreille")
plt.legend()
plt.title("Régression linéaire optimale")
plt.show()



print("\n=== PARTIE 2: Régression polynomiale ===")

def f(x):
    eps = np.finfo(np.float32).eps
    return 12 * np.sin(x) / (x + eps)

X_data = np.linspace(-3, 10, 40)
y_data = f(X_data) + np.random.normal(0, 1, 40)

plt.scatter(X_data, y_data, alpha=0.7)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Données bruitées")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.33, random_state=12
)

def rss(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)

degrees = [1, 2, 3, 6, 9, 12]
train_rss_list = []
test_rss_list = []
train_r2_list = []
test_r2_list = []

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
    model.fit(X_train.reshape(-1, 1), y_train)
    
    y_train_pred = model.predict(X_train.reshape(-1, 1))
    y_test_pred = model.predict(X_test.reshape(-1, 1))
    
    train_rss = rss(y_train, y_train_pred)
    test_rss = rss(y_test, y_test_pred)
    
    train_r2 = model.score(X_train.reshape(-1, 1), y_train)
    test_r2 = model.score(X_test.reshape(-1, 1), y_test)
    
    train_rss_list.append(train_rss)
    test_rss_list.append(test_rss)
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)
    
    print(f"Degree {degree}:")
    print(f"  Train RSS: {train_rss:.4f}, Test RSS: {test_rss:.4f}")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    X_plot = np.linspace(-3, 10, 200)
    y_plot = model.predict(X_plot.reshape(-1, 1))
    
    plt.figure(figsize=(8, 4))
    plt.scatter(X_train, y_train, label='Train', alpha=0.7)
    plt.scatter(X_test, y_test, label='Test', alpha=0.7)
    plt.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'Degré {degree}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Modèle polynomial degré {degree}')
    plt.legend()
    plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(degrees, train_rss_list, 'o-', label='Train RSS')
plt.plot(degrees, test_rss_list, 's-', label='Test RSS')
plt.xlabel('Degré du polynôme')
plt.ylabel('RSS')
plt.legend()
plt.title('Erreur RSS vs degré polynomial')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(degrees, train_r2_list, 'o-', label='Train R²')
plt.plot(degrees, test_r2_list, 's-', label='Test R²')
plt.xlabel('Degré du polynôme')
plt.ylabel('R²')
plt.legend()
plt.title('Score R² vs degré polynomial')
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n=== SYNTHÈSE ===")
print(f"Meilleur modèle (test R²): degré {degrees[np.argmax(test_r2_list)]}")
print(f"Sur-apprentissage visible à partir du degré 9")