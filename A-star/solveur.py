from noeud import Noeud

class Solveur:
    def astar(self, grille, heur, size):
        root = Noeud(grille, pere=None, g=0, heur=heur, size=size)
        if root.estUnEtatFinal():
            return root
        openList = [root]
        closedList = []
        while openList != []:
            print(len(openList))
            current = min(openList, key=lambda x: x.f())
            openList.remove(current)
            closedList.append(current.grille)
            if current.estUnEtatFinal():
                return current
            for succ in current.successeurs():
                if succ.grille in closedList:
                    continue
                found = False
                for n in openList:
                    if n == succ:
                        found = True
                        if succ.f() < n.f():
                            openList.remove(n)
                            openList.append(succ)
                        break
                if not found:
                    openList.append(succ)
        return None

    def afficherChemin(self, noeud):
        if noeud is None:
            print("Pas de solution")
            return
        path = []
        current = noeud
        while current:
            path.append(current)
            current = current.pere
        path.reverse()
        for i, n in enumerate(path):
            print(f"Étape {i}:")
            print(n)
            print()
        print(f"Nombre de mouvements: {len(path) - 1}")


grille = [
     [8, 6, 7],
     [2, 5, 4],
     [1, 3, 0]
]
#grille = [[3, 2], [1, 0]]
size = 3

solveur = Solveur()
# print("Résolution avec h1:")
# solution_h1 = solveur.astar(grille, 'h1', size=size)
# solveur.afficherChemin(solution_h1)

print("\nRésolution avec h2:")
solution_h2 = solveur.astar(grille, 'h2', size=size)
solveur.afficherChemin(solution_h2)