import copy

# Constantes
PION_VIDE = 0
PION_MAX = 1   # IA (joueur maximisant)
PION_MIN = 2   # Humain (joueur minimisant)
TAILLE = 5

class Noeud:
    def __init__(self, grille=None, isMax=True):
        if grille is None:
            self.grille = [[PION_VIDE for _ in range(TAILLE)] for _ in range(TAILLE)]
        else:
            self.grille = copy.deepcopy(grille)
        self.isMax = isMax
        self.h = 0  # Évaluation heuristique

    def __str__(self):
        s = ""
        for ligne in self.grille:
            s += "|".join(str(x) for x in ligne) + "\n"
        s += "------------\n"
        s += f"evaluation = {self.h}\n"
        return s

    def troisPionsAlignesLignes(self, pion):
        score = 0
        for i in range(TAILLE):
            for j in range(TAILLE - 2):  # 3 cases consécutives
                window = self.grille[i][j:j+3]
                count_pion = window.count(pion)
                count_vide = window.count(PION_VIDE)

                if count_pion == 3:
                    score += 10000
                elif count_pion == 2 and count_vide == 1:
                    score += 200
                elif count_pion == 1 and count_vide == 2:
                    score += 30
        return score

    def troisPionsAlignesColonnes(self, pion):
        score = 0
        for j in range(TAILLE):
            for i in range(TAILLE - 2):
                a = self.grille[i][j]
                b = self.grille[i+1][j]
                c = self.grille[i+2][j]
                window = [a, b, c]
                count_pion = window.count(pion)
                count_vide = window.count(PION_VIDE)

                if count_pion == 3:
                    score += 10000
                elif count_pion == 2 and count_vide == 1:
                    score += 200
                elif count_pion == 1 and count_vide == 2:
                    score += 30
        return score

    def evaluer(self):
        self.h = 0
        poids = -1
        self.h += self.troisPionsAlignesLignes(PION_MAX)
        self.h += poids * self.troisPionsAlignesLignes(PION_MIN)
        self.h += self.troisPionsAlignesColonnes(PION_MAX)
        self.h += poids * self.troisPionsAlignesColonnes(PION_MIN)

    def finJeu(self):
        # Victoire MAX ou MIN ?
        if (self.troisPionsAlignesLignes(PION_MAX) >= 10000 or
            self.troisPionsAlignesColonnes(PION_MAX) >= 10000):
            self.h = 100000  # Victoire quasi-certaine pour MAX
            return True
        if (self.troisPionsAlignesLignes(PION_MIN) >= 10000 or
            self.troisPionsAlignesColonnes(PION_MIN) >= 10000):
            self.h = -100000
            return True

        # Grille pleine ?
        pleine = all(self.grille[i][j] != PION_VIDE for i in range(TAILLE) for j in range(TAILLE))
        if pleine:
            self.h = 0
            return True
        return False


class Puissance3:
    def __init__(self):
        self.WIDTH = TAILLE
        self.HEIGHT = TAILLE
        self.grilleJeu = [[PION_VIDE for _ in range(TAILLE)] for _ in range(TAILLE)]

    def __str__(self):
        s = ""
        for ligne in self.grilleJeu:
            s += "|".join(str(x) for x in ligne) + "\n"
        s += "------------\n"
        return s

    def jouerColonne(self, pion, c, grille=None):
        if grille is None:
            grille = self.grilleJeu
        if c < 0 or c >= TAILLE:
            return False
        # Trouver la première ligne vide dans la colonne c (en partant du bas)
        for i in range(TAILLE-1, -1, -1):
            if grille[i][c] == PION_VIDE:
                grille[i][c] = pion
                return True
        return False  # Colonne pleine

    def alpha_beta(self, noeud, alpha, beta, profondeur):
        if profondeur == 0 or noeud.finJeu():
            noeud.evaluer()
            return noeud.h, -1

        meilleur_coup = -1

        if noeud.isMax:
            max_eval = float('-inf')
            for c in range(TAILLE):
                if self.colonneValide(c, noeud.grille):
                    copie_grille = copy.deepcopy(noeud.grille)
                    self.jouerColonne(PION_MAX, c, copie_grille)
                    successeur = Noeud(copie_grille, False)
                    eval_coup, _ = self.alpha_beta(successeur, alpha, beta, profondeur - 1)
                    if eval_coup > max_eval:
                        max_eval = eval_coup
                        meilleur_coup = c
                    alpha = max(alpha, eval_coup)
                    if beta <= alpha:
                        break  # Coupure beta
            return max_eval, meilleur_coup
        else:  # MIN
            min_eval = float('inf')
            for c in range(TAILLE):
                if self.colonneValide(c, noeud.grille):
                    copie_grille = copy.deepcopy(noeud.grille)
                    self.jouerColonne(PION_MIN, c, copie_grille)
                    successeur = Noeud(copie_grille, True)
                    eval_coup, _ = self.alpha_beta(successeur, alpha, beta, profondeur - 1)
                    if eval_coup < min_eval:
                        min_eval = eval_coup
                        meilleur_coup = c
                    beta = min(beta, eval_coup)
                    if beta <= alpha:
                        break  # Coupure alpha
            return min_eval, meilleur_coup

    def colonneValide(self, c, grille):
        return grille[0][c] == PION_VIDE  # Si le haut est vide, on peut jouer





def main():
    jeu = Puissance3()
    profondeur = 6  # Profondeur raisonnable pour 5x5

    print("=== PUISSANCE 3 : Humain (2) vs IA (1) ===")
    print("Choisissez qui commence :")
    print("1 - Humain commence")
    print("2 - IA commence")
    choix = input("Votre choix (1 ou 2) : ")

    humain_commence = (choix == "1")

    print(jeu)

    while True:
        if humain_commence:
            # Tour humain
            while True:
                try:
                    col = int(input("Votre coup (colonne 0-4) : "))
                    if 0 <= col <= 4 and jeu.jouerColonne(PION_MIN, col):
                        break
                    else:
                        print("Coup invalide !")
                except:
                    print("Entrez un nombre entre 0 et 4")

            print(jeu)
            noeud = Noeud(jeu.grilleJeu, False)
            if noeud.finJeu():
                print("Vous avez gagné !" if noeud.h < 0 else "Match nul !")
                break

            # Tour IA
            print("IA réfléchit...")
            _, coup_ia = jeu.alpha_beta(Noeud(jeu.grilleJeu, True), float('-inf'), float('inf'), profondeur)
            jeu.jouerColonne(PION_MAX, coup_ia)
            print(f"IA joue en colonne {coup_ia}")
            print(jeu)

            noeud = Noeud(jeu.grilleJeu, True)
            if noeud.finJeu():
                print("L'IA a gagné !" if noeud.h > 0 else "Match nul !")
                break

        else:
            # IA commence
            print("IA commence...")
            _, coup_ia = jeu.alpha_beta(Noeud(jeu.grilleJeu, True), float('-inf'), float('inf'), profondeur)
            jeu.jouerColonne(PION_MAX, coup_ia)
            print(f"IA joue en colonne {coup_ia}")
            print(jeu)

            noeud = Noeud(jeu.grilleJeu, True)
            if noeud.finJeu():
                print("L'IA a gagné !")
                break

            # Tour humain
            while True:
                try:
                    col = int(input("Votre coup (colonne 0-4) : "))
                    if 0 <= col <= 4 and jeu.jouerColonne(PION_MIN, col):
                        break
                    else:
                        print("Coup invalide !")
                except:
                    print("Entrez un nombre valide")

            print(jeu)
            noeud = Noeud(jeu.grilleJeu, False)
            if noeud.finJeu():
                print("Vous avez gagné !" if noeud.h < 0 else "Match nul !")
                break

        humain_commence = not humain_commence  # Alternance


# Test de l'évaluation (comme dans le sujet)
def test_evaluation():
    grille_test = [
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,1,0,0,0],
        [1,2,0,2,0]
    ]
    n = Noeud(grille_test, True)
    n.evaluer()
    print(n)
    print("Attendu : evaluation = -140")

if __name__ == "__main__":
    test_evaluation()  # Décommentez pour tester l'évaluation
    main()