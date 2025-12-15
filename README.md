# AI Algorithms for Problem Solving, Games and Optimization

Ce dépôt regroupe plusieurs **algorithmes fondamentaux en Intelligence Artificielle, optimisation et machine learning**, implémentés en Python dans un cadre académique (TP / expérimentations).

Il couvre des méthodes de **recherche**, **jeux**, **optimisation**, **apprentissage supervisé**, ainsi que la **manipulation de jeux de données réels**.

---

## 📂 Structure du dépôt

```
.
├── TP1/
├── TP2/
├── genetics.py
├── sim_annealing.py
├── regression_lineaire.py
├── regression_polynomiale.py
├── knn_diabetes_classification.py
├── diabetes.csv
├── housing.csv
├── possum.csv
├── .gitignore
└── README.md
```

---

## 🔍 Algorithmes implémentés

### ⭐ A* (A-Star Search)

* Algorithme de recherche informée pour le **plus court chemin**.
* Utilise une fonction heuristique pour guider l'exploration.
* Implémentation basée sur les concepts de **Node**, **coût**, et **fonction f(n) = g(n) + h(n)**.

**Applications :**

* Recherche de chemin
* Résolution de problèmes de graphes

---

### 🎮 Alpha-Beta Pruning

* Optimisation de l’algorithme **Minimax** pour les jeux à somme nulle.
* Implémenté pour le jeu **Power 3**.
* Réduction du nombre de nœuds explorés grâce à l’élagage Alpha-Beta.

**Applications :**

* Jeux adversariaux
* Intelligence artificielle pour jeux

---

### 🔥 Simulated Annealing

* Algorithme d’optimisation stochastique inspiré du recuit thermique.
* Permet d’échapper aux **optima locaux**.
* Contrôlé par une fonction de température décroissante.

**Applications :**

* Optimisation combinatoire
* Problèmes NP-difficiles

---

### 🧬 Algorithme Génétique

* Métaheuristique basée sur l’évolution naturelle.
* Utilise :

  * Sélection
  * Croisement
  * Mutation
* Optimisation itérative d’une population de solutions.

**Applications :**

* Optimisation globale
* Recherche de solutions approchées

---

## 📊 Machine Learning

### 📈 Régression Linéaire

* Implémentation avec **scikit-learn**.
* Validation des modèles.
* Apprentissage supervisé pour variables continues.

Fichier : `regression_lineaire.py`

---

### 📉 Régression Polynomiale & Ridge

* Extension de la régression linéaire.
* Gestion du **sur-apprentissage** avec la régularisation Ridge.

Fichier : `regression_polynomiale.py`

---

### 🧠 Classification KNN

* Algorithme **K-Nearest Neighbors**.
* Standardisation des données.
* Application sur le dataset **diabetes**.

Fichier : `knn_diabetes_classification.py`

---

## 🗃️ Jeux de données

Les datasets utilisés sont fournis au format `.csv` :

* `diabetes.csv` – Classification médicale
* `housing.csv` – Régression (prix de logements)
* `possum.csv` – Données biologiques

---

## ⚙️ Prérequis

* Python ≥ 3.8
* Bibliothèques principales :

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## ▶️ Exécution

Exemple :

```bash
python regression_lineaire.py
python knn_diabetes_classification.py
python genetics.py
```

---

## 🎯 Objectifs pédagogiques

* Comprendre les algorithmes classiques de l’IA
* Manipuler des datasets réels
* Comparer différentes approches d’optimisation
* Appliquer des modèles de machine learning

---

## 👤 Auteur

**Othmane Abderrazik**
Étudiant en Génie Informatique / IA

---

## 📜 Licence

Projet à but **éducatif**.

Libre à utiliser pour l’apprentissage et l’expérimentation.
