import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("diabetes.csv")

print("Aperçu des données:")
print("Taille du dataset:", df.shape)
print(df.head())
print("Classes cibles uniques:", df['Outcome'].unique())

print("Types de données:")
print(df.info())
print("Statistiques descriptives:")
print(df.describe())
print("Distribution des classes de la variable cible:")
print(df['Outcome'].value_counts())

x = df.drop('Outcome', axis=1)
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

print("Taille des ensembles d'entraînement et de test:")
print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

predictions = knn.predict(x_test)

# print(knn.score(x_test, y_test))
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)



plt.figure(figsize=(10, 6))
k_values = list(range(1, 21))
accuracies = []

for i in range(k_values[-1]):
    knn = KNeighborsClassifier(n_neighbors=i+1)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy for k={i+1}: {accuracy}")
    conf_matrix = confusion_matrix(y_test, predictions)
    print(f"Confusion Matrix for k={i+1} : ")
    print(conf_matrix)
    accuracies.append(accuracy)

plt.plot(k_values, accuracies, marker='o')
plt.title("KNN Classifier Accuracy for Different k Values")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid()
plt.show()


    
