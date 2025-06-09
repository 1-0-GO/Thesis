from bisect import bisect_left
import pandas as pd

class ParetoFront:
    """
    Maintain a Pareto front of Solutions (minimize complexity and loss).
    Stores self.front sorted by ascending complexity.  Invariants:
      - front[i].fitness = (c_i, l_i)
      - c_0 < c_1 < … < c_{m-1}
      - l_0 > l_1 > … > l_{m-1}
    """

    def __init__(self):
        # the list of non-dominated solutions
        self.front = []

    def _values(self, sol):
        """Extract (complexity, loss) from your Solution object."""
        return sol.fitness  # assumed to be a tuple (complexity, loss)

    def check_dominated(self, sol):
        """
        Return the index where sol would be inserted if it is non-dominated;
        or -1 if sol is dominated by someone on the front.
        """
        c_new, l_new = self._values(sol)
        # build a temporary list of complexities for bisect
        idx = bisect_left(self.front, sol)

        # 1) Check left neighbor: if it has loss <= l_new, it dominates sol
        if idx > 0 and self.front[idx-1].fitness[1] <= l_new:
            return -1

        # 2) Also check any existing with exactly equal complexity
        if idx < len(self.front) and self.front[idx].fitness[0] == c_new and self.front[idx].fitness[1] <= l_new:
            return -1

        # sol is not dominated; idx is correct insertion position
        return idx

    def update_one(self, sol):
        """
        Try to insert sol into the front.
        Returns True if inserted (and may have removed dominated points),
        or False if sol was dominated.
        """
        idx = self.check_dominated(sol)
        if idx < 0:
            return False

        # Remove any existing solutions to the right that are dominated by sol:
        # while their loss >= sol.loss
        _, l_new = self._values(sol)
        j = idx
        while j < len(self.front) and self.front[j].fitness[1] >= l_new:
            del self.front[j]

        # Insert sol at the correct place by complexity
        self.front.insert(idx, sol)
        return True

    def update(self, solutions):
        """Batch-update: insert each solution in turn."""
        for sol in solutions:
            self.update_one(sol)

    def __iter__(self):
        return iter(self.front)

    def __len__(self):
        return len(self.front)
    
    def to_df(self):
        data = []
        for sol in self:
            data.append({"equation": sol.equation, "complexity": sol.fitness[0], "loss": sol.fitness[1]})
        return pd.DataFrame(data)

class Solution:
    def __init__(self, equation, complexity, loss):
        self.equation = equation
        self.fitness = (round(complexity), loss)
    def __repr__(self):
        c, l = self.fitness
        return f"<Sol eq={self.equation!r} c={c} l={l}>"
    def __lt__(self, other):
        return self.fitness < other.fitness

########
# Test #
########

pf = ParetoFront()

solutions = [
    {"complexity": 5.0, "loss": 0.1, "data": "Solution A"},
    {"complexity": 5.05, "loss": 0.1, "data": "Solution B"},
    {"complexity": 10.0, "loss": 0.05, "data": "Solution C"},
    {"complexity": 15, "loss": 1e-6, "data": "Solution Z"},  
    {"complexity": 10.1, "loss": 0.05-1e-6, "data": "Solution D"},
    {"complexity": 7.0, "loss": 0.2, "data": "Solution E"},  
    {"complexity": 2.0, "loss": 1.5, "data": "Solution F"},
]
solutions = [Solution(sol['data'], sol['complexity'], sol['loss']) for sol in solutions]
pf.update(solutions)
print("Pareto Front:")
for sol in pf:
    print(sol)

solutions = [  
    {"complexity": 2.0, "loss": 1.4, "data": "Solution F"},  
    {"complexity": 3.0, "loss": 1.5, "data": "Solution F"},  
    {"complexity": 16, "loss": 1e-6, "data": "Solution Y"},  
]
solutions = [Solution(sol['data'], sol['complexity'], sol['loss']) for sol in solutions]
pf.update(solutions)
print("Pareto Front:")
for sol in pf:
    print(sol)

pf.update_one(Solution('Last', 16, 1e-9))

print("Final Pareto Front:")
print(pf.to_df())
print(list(pf))