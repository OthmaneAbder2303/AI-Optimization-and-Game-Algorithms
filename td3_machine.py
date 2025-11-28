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

def recuitSimule(etatInitial, temperatureInitiale, tauxRefroidissement, iter_max):
    etatCourant = etatInitial[:]
    temperature = temperatureInitiale
    evalCourante = evaluer(etatCourant)
    while temperature > 1e-3:
        for _ in range(iter_max):
            voisin = etatCourant[:]
            i = random.randint(0, len(voisin) - 1)
            voisin[i] = random.randint(0, len(voisin) - 1)

            evalVoisin = evaluer(voisin)
            deltaEval = evalVoisin - evalCourante

            if deltaEval < 0 or random.uniform(0, 1) < math.exp(-deltaEval / temperature):
                etatCourant = voisin
                evalCourante = evalVoisin

        temperature *= tauxRefroidissement
    return etatCourant


etat_init = etatInitial(4)
afficher(etat_init)
print("Evaluation initiale:", evaluer(etat_init))

etatFinal = recuitSimule(etat_init, temperatureInitiale=1000, tauxRefroidissement=0.95, iter_max=100)
afficher(etatFinal)
print("Evaluation finale:", evaluer(etatFinal))

