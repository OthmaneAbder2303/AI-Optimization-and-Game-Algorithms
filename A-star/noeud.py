import copy

class Noeud:
    def __init__(self, grille, size, pere=None, g=0, heur='h1'):
        self.grille = copy.deepcopy(grille)
        self.size = size
        self.pere = pere
        self.g = g
        self.heur = heur

        # on calcule le goal pour la taille donnee
        self.GOAL = []
        num = 1
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if i == self.size - 1 and j == self.size - 1:
                    row.append(0)
                else:
                    row.append(num)
                    num += 1
            self.GOAL.append(row)

        # calcul de GOAL_POS
        self.GOAL_POS = {}
        for i in range(self.size):
            for j in range(self.size):
                val = self.GOAL[i][j]
                if val != 0:
                    self.GOAL_POS[val] = (i, j)

    def __str__(self):
        return '\n'.join(' '.join(str(cell) if cell != 0 else ' ' for cell in row) for row in self.grille)

    def __eq__(self, other):
        return self.grille == other.grille

    def h(self):
        if self.heur == 'h1':
            count = 0
            for i in range(self.size):
                for j in range(self.size):
                    if self.grille[i][j] != 0 and self.grille[i][j] != self.GOAL[i][j]:
                        count += 1
            return count
        elif self.heur == 'h2':
            dist = 0
            for i in range(self.size):
                for j in range(self.size):
                    val = self.grille[i][j]
                    if val != 0:
                        target_i, target_j = self.GOAL_POS[val]
                        dist += abs(i - target_i) + abs(j - target_j)
            return dist
        else:
            raise ValueError(self.heur)

    def f(self):
        return self.g + self.h()

    def successeurs(self):
        global v_i, v_j
        for i in range(self.size):
            for j in range(self.size):
                if self.grille[i][j] == 0:
                    v_i = i
                    v_j = j
        succ = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for di, dj in directions:
            n_i, n_j = v_i + di, v_j + dj
            if 0 <= n_i < self.size and 0 <= n_j < self.size:
                new_grille = copy.deepcopy(self.grille)
                new_grille[v_i][v_j], new_grille[n_i][n_j] = new_grille[n_i][n_j], new_grille[v_i][v_j]
                succ.append(Noeud(new_grille, size=self.size, pere=self, g=self.g + 1, heur=self.heur))
        return succ

    def estUnEtatFinal(self):
        return self.grille == self.GOAL

#
# grille1 = [
#     [7, 2, 4],
#     [5, 0, 6],
#     [8, 3, 1]
# ]
# grille2 = [
#     [1, 2, 3],
#     [4, 5, 6],
#     [8, 7, 0]
# ]
#
# noeud1 = Noeud(grille1, size=3, heur='h1')
# print(noeud1)
#
# noeud2 = Noeud(grille2, size=3, heur='h2')
# print(noeud2)
#
# print(noeud1 == noeud2)
#
# print(noeud1.h())
# print(noeud2.h())
#
# for succ in noeud2.successeurs():
#     print(succ)
#     print(succ.f())