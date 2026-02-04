import copy

class Noeud:
    SIZE = 3
    GOAL = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    GOAL_POS = {}
    for i in range(SIZE):
        for j in range(SIZE):
            val = GOAL[i][j]
            if val != 0:
                GOAL_POS[val] = (i, j)

    def __init__(self, grille, pere=None, g=0, heur='h1'):
        self.grille = [row[:] for row in grille]
        self.pere = pere
        self.g = g
        self.heur = heur

    def __str__(self):
        return '\n'.join(' '.join(str(cell) if cell != 0 else ' ' for cell in row) for row in self.grille)

    def __eq__(self, other):
        if not isinstance(other, Noeud):
            return False
        return self.grille == other.grille

    def h(self, heur=None):
        if heur is None:
            heur = self.heur
        if heur == 'h1':
            count = 0
            for i in range(self.SIZE):
                for j in range(self.SIZE):
                    if self.grille[i][j] != 0 and self.grille[i][j] != self.GOAL[i][j]:
                        count += 1
            return count
        elif heur == 'h2':
            sumd = 0
            for i in range(self.SIZE):
                for j in range(self.SIZE):
                    val = self.grille[i][j]
                    if val != 0:
                        gi, gj = self.GOAL_POS[val]
                        sumd += abs(i - gi) + abs(j - gj)
            return sumd
        else:
            raise ValueError("Heuristique inconnue")

    def f(self):
        return self.g + self.h()

    def estUnEtatFinal(self):
        return self.grille == self.GOAL

    def find_blank(self):
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if self.grille[i][j] == 0:
                    return i, j
        raise ValueError("Pas de case vide")

    def successeurs(self):
        i, j = self.find_blank()
        succs = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.SIZE and 0 <= nj < self.SIZE:
                new_grille = copy.deepcopy(self.grille)
                new_grille[i][j], new_grille[ni][nj] = new_grille[ni][nj], new_grille[i][j]
                succs.append(Noeud(new_grille, self, self.g + 1, self.heur))
        return succs

class Solveur:
    def astar(self, initial_grille, heur='h1'):
        root = Noeud(initial_grille, None, 0, heur)
        if root.estUnEtatFinal():
            return root
        open_list = [root]
        closed = set()
        while open_list:
            current = min(open_list, key=lambda n: n.f())
            open_list.remove(current)
            current_tuple = tuple(tuple(row) for row in current.grille)
            closed.add(current_tuple)
            if current.estUnEtatFinal():
                return current
            for succ in current.successeurs():
                succ_tuple = tuple(tuple(row) for row in succ.grille)
                if succ_tuple in closed:
                    continue
                found = False
                for n in open_list:
                    if n == succ:
                        found = True
                        if succ.f() < n.f():
                            open_list.remove(n)
                            open_list.append(succ)
                        break
                if not found:
                    open_list.append(succ)
        return None

    def afficher_chemin(self, node):
        if node is None:
            print("Pas de solution")
            return
        path = []
        current = node
        while current:
            path.append(current)
            current = current.pere
        path.reverse()
        for i, n in enumerate(path):
            print(f"Étape {i}:")
            print(n)
            print()
        print(f"Nombre de mouvements: {len(path) - 1}")

# Tests
initial_grille = [[7, 2, 4], [5, 0, 6], [8, 3, 1]]

# Test classe Noeud
node = Noeud(initial_grille)
print("Affichage du noeud:")
print(node)
print("\nValeur h avec h1:", node.h('h1'))  # Devrait être 6
print("Valeur h avec h2:", node.h('h2'))  # Devrait être 14

print("\nSuccesseurs:")
succs = node.successeurs()
for s in succs:
    print(s)
    print()

# Test Solveur avec h1
solveur = Solveur()
print("Résolution avec h1:")
solution_h1 = solveur.astar(initial_grille, 'h1')
solveur.afficher_chemin(solution_h1)

# Test Solveur avec h2
print("\nRésolution avec h2:")
solution_h2 = solveur.astar(initial_grille, 'h2')
solveur.afficher_chemin(solution_h2)