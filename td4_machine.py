import random
import math

def etatInitial(N):
    etat =[]
    for i in range(N) :
        etat.append(random.randint(0, N - 1))
    return etat

def afficher(etat):
    N = len(etat)
    for i in range(N):
        for j in range(N):
            if(etat[j]==i):
                print ("|Q", end="")
            else:
                print("| ", end="")
        print("|")


def evaluer(etat):
    eval = 0
    N = len(etat)
    for i in range(N):
        for j in range(i+1, N):
            if etat[i] == etat[j] or abs(i - j) == abs(etat[i] - etat[j]):
                eval += 1
    return eval


def fitness(etat):
    return 1/(1+ evaluer(etat))


def population_initiale(taille_population, N):
    population = []
    for i in range(taille_population):
        population.append(etatInitial(N))
    return population


def selection(population):
    total = 0
    for solution in population :
        # print(solution)
        # print(fitness(solution))

        total +=fitness(solution)

    r = random.uniform(0, total)
    # print("Random value for selection:", r)

    cumulative_fitness = 0
    for solution in population:
        cumulative_fitness += fitness(solution)
        if cumulative_fitness >= r:
            # print("Selected solution:", solution)
            # print("Selected solution fitness:", fitness(solution))
            return solution
        

def croisement(parent1, parent2):
    point_croisement = random.randint(1, len(parent1) - 1)
    enfant = parent1[:point_croisement] + parent2[point_croisement:]
    # print("Parent 1:", parent1)
    # print("Parent 2:", parent2)
    # print("Point de croisement:", point_croisement)
    # print("Enfant:", enfant)
    return enfant

def croisement_guide(parent1, parent2):
    N = len(parent1)
    enfant = [-1] * N

    for col in range(N):
        choix1 = parent1[col]
        choix2 = parent2[col]

        # On essaye chaque candidat
        conflits_choix1 = 0
        conflits_choix2 = 0

        # Conflits si on met choix1 à la colonne `col`
        for c in range(col):
            if enfant[c] == choix1:
                conflits_choix1 += 1

        # Conflits si on met choix2 à la colonne `col`
        for c in range(col):
            if enfant[c] == choix2:
                conflits_choix2 += 1

        # Choix du meilleur
        if conflits_choix1 < conflits_choix2:
            enfant[col] = choix1
        elif conflits_choix2 < conflits_choix1:
            enfant[col] = choix2
        else:
            enfant[col] = random.choice([choix1, choix2])

    return enfant



def mutation(etat):
    # print("Avant mutation:", etat)
    i = random.randint(0, len(etat) - 1)
    etat[i] = random.randint(0, len(etat) - 1)
    # print("Après mutation:", etat)
    return etat

def algorithme_genetique(N, taille_population, nombre_generations, taux_mutation):
    population = population_initiale(taille_population, N)

    for generation in range(nombre_generations):
        nouvelle_population = []

        for _ in range(taille_population):
            parent1 = selection(population)
            parent2 = selection(population)
            enfant = croisement_guide(parent1, parent2)

            if random.random() < taux_mutation:
                enfant = mutation(enfant)

            nouvelle_population.append(enfant)
            if(fitness(enfant) == 1.0):
                print("Solution optimale trouvée dans la génération", generation+1)
                return enfant

        population = nouvelle_population

        meilleur = max(population, key=fitness)
        if(generation % 50 == 0):
            print(f"Génération {generation+1}, meilleure fitness = {fitness(meilleur)}")

    return max(population, key=fitness)


# etat_init = etatInitial(4)
# afficher(etat_init)
# print("Fitness initiale:", fitness(etat_init))

etatFinal = algorithme_genetique(N=16, taille_population=100, nombre_generations=1000, taux_mutation=0.03)
afficher(etatFinal)
print("Fitness finale:", fitness(etatFinal))

