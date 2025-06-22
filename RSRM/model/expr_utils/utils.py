import _thread
import threading
from contextlib import contextmanager
from typing import Dict
from sympy import simplify, sympify, Function
from model.expr_utils.exp_tree import PreTree
from bisect import bisect_left
import pandas as pd
from sympy import Array, NDimArray

import numpy as np

from model.expr_utils.exp_tree_node import Expression

class If(Function):
    @classmethod
    def eval(cls, x, y):
        if isinstance(x, NDimArray):
            return Array(np.where(np.array(x.tolist())>0, np.array(y.tolist()), 0))

def get_expression(strs: str) -> Expression:
    """
    Get the expression variable for the corresponding expression.
    :param strs: expression string
    :return: the corresponding expression
    """
    exp_dict = {
        "Id": Expression(1, np.array, lambda x: f"{x}"),
        "Add": Expression(2, np.add, lambda x, y: f"({x}) + ({y})"),
        "Sub": Expression(2, np.subtract, lambda x, y: f"({x})-({y})"),
        "Mul": Expression(2, np.multiply, lambda x, y: f"({x})*({y})"),
        "Div": Expression(2, np.divide, lambda x, y: f"({x})/({y})"),
        "Pow": Expression(2, np.power, lambda x, y: f"({x})**({y})"),
        "Dec": Expression(1, lambda x: x - 1, lambda x: f"({x})+1"),
        "Inc": Expression(1, lambda x: x + 1, lambda x: f"({x})-1"),
        "Neg": Expression(1, np.negative, lambda x: f"-({x})"),
        "Exp": Expression(1, np.exp, lambda x: f"exp(C*({x}))"),
        "Log": Expression(1, np.log, lambda x: f"log({x})"),
        "Sin": Expression(1, np.sin, lambda x: f"sin({x})"),
        "Cos": Expression(1, np.cos, lambda x: f"cos({x})"),
        "Asin": Expression(1, np.arcsin, lambda x: f"arcsin({x})"),
        "Atan": Expression(1, np.arctan, lambda x: f"arctan({x})"),
        "Sqrt": Expression(1, np.sqrt, lambda x: f"({x})**0.5"),
        "N2": Expression(1, np.square, lambda x: f"({x})**2"),
        "Pi": Expression(0, np.pi, 'pi'),
        "One": Expression(0, 1, '1'),   
        # New functions
        "Tanh": Expression(1, np.tanh, lambda x: f"tanh({x} - C)"),
        "Denom": Expression(1, np.tanh, lambda x: f"1/( C*({x}) - 1)"),
        "Inv": Expression(1, lambda x: 1 / (x + 1e-6), lambda x: f"({x})**(-1)"),
        "Ond": Expression(2, lambda x, y: np.where(x>0, y, 0), lambda x, y: f""),
        "Harmonic": Expression(2, lambda x, y: 2 * x * y / (x + y), lambda x, y: f"2*({x})*({y})/(({x})+({y}))"),
        "Saturation": Expression(2, lambda x, k: k * x / (k + x), lambda x, k: f"({k})*({x})/(({k})+({x}))"),
        "Damping": Expression(2, lambda x, k: x / (1 + k * x), lambda x, k: f"({x})/(1+({k})*({x}))"),
        "Threshold": Expression(2, lambda x, k: np.maximum(0, x - k), lambda x, k: f"max(0, ({x})-({k}))"),
        "Logistic": Expression(1, lambda x: 1 / (1 + np.exp(-x)), lambda x: f"1/(1+exp(-({x})))")
    }

    if strs in exp_dict:
        return exp_dict[strs]
    return Expression(0, None if strs == 'C' else int(strs[1:]), strs)

def format_numbers_in_string(text, precision=2):
    pattern = r'(?<![a-zA-Z_.])-?(?:\d*\.\d+|\d+\.\d+|\d+[eE][+-]?\d+)(?![a-zA-Z_.])'
    def format_match(match):
        return f"{float(match.group()):.{precision}f}"
    
    return re.sub(pattern, format_match, text)

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

    def __repr__(self):
        return str(self.front)

    def __iter__(self):
        return iter(self.front)

    def __len__(self):
        return len(self.front)
    
    def clear(self):
        self.front = []
    
    def to_df(self):
        data = []
        for sol in self:
            equation_ = format_numbers_in_string(sol.equation)
            data.append({"equation": equation_, "complexity": sol.fitness[0], "loss": sol.fitness[1]})
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


def expression_dict(tokens, num_of_var, const) -> Dict[int, Expression]:
    """
    Create expression dictionary key is index, value is Expression class Number of parameters,
     numpy computes expression, string computes expression
    :return: dictionary: key is index, value is Expression
    """

    def generate_expression_dict(expression_list) -> Dict[int, Expression]:
        exp_dict = {}
        for i, expression in enumerate(expression_list):
            exp = get_expression(expression)
            exp.type = i
            exp_dict[i] = exp
            exp.type_name = expression
        return exp_dict

    return generate_expression_dict(
        [f'X{i}' for i in range(1, 1 + num_of_var)] +
        (["C"] if const else []) +
        tokens
    )


class FinishException(Exception):
    """
    Exceptions when getting the correct expression
    """
    pass


import re
def count_weighted_operations(expression):
    try:
        expression = str(simplify(sympify(expression)))

        # Define a regex pattern for the operations
        pattern = r'max|min|log|sin|cos|exp|\+|\-|\*\*|\*|\/|\bd+\b'
        
        # Define weights for each operation
        weights = {
            '+': 0,
            '-': 0,
            '*': 0,
            '/': 1,
            '**': 1,
            'log': 1,
            'sin': 1,
            'cos': 1,
            'exp': 1,
            'min': 1,
            'max': 1
        }
        
        # Find all operations in the expression
        operations = re.findall(pattern, expression)

        # Find all variables in the expression
        vars_and_consts = re.findall(r'X\d+|C', expression)

        # Exclude digits that are part of variables
        for vc in vars_and_consts:
            expression = expression.replace(vc, '')
        
        standalone_digits = re.findall(r'[1-9]\d*', expression)

        # Calculate the total weight
        total_weight = sum(weights[op] for op in operations)
        
        return total_weight + len(vars_and_consts) + len(standalone_digits)
    except Exception:
        return np.iinfo(np.int32).max
    
def calculate_formula_complexity(expression, tree):
    try:
        expression = str(simplify(sympify(expression)))

        # Define regex patterns for different components
        operation_pattern = r'max|min|log|sin|cos|exp|if|\+|\-|\*|\^|\/'
        non_arithmetic_pattern = r'max|min|log|sin|cos|exp|if'
        
        # Find all matches for operations
        operations = re.findall(operation_pattern, expression)
        non_arithmetic_ops = re.findall(non_arithmetic_pattern, expression)
        
        # Calculate components
        l = len(re.findall(r'\w+|[-+*/^]', expression))  # Count variables, constants, and operations
        no = len(operations)  # Number of operations
        nnao = len(non_arithmetic_ops)  # Non-arithmetic operations
        nnaoc = calculate_total_nesting(tree)  # Consecutive compositions
        
        # Apply the formula
        complexity = 0.2 * l + 0.5 * no + 3.4 * nnao + 4.5 * nnaoc
        
        return complexity
    except Exception as e:
        print(f"Error: {e}")
        return 0

def calculate_total_nesting(tree: PreTree):
    total_nesting = 0

    def dfs(node, nnao=0):
        nonlocal total_nesting
        if not node._children:  # Leaf node
            total_nesting += nnao
        if node._expr.type_name not in ['Add', 'Sub', 'Mul', 'Div']:
            nnao += 1
        for child in node._children:
            dfs(child, nnao = nnao)

    dfs(tree.root, nnao=0)
    return total_nesting

def calculate_loss(loss, complexity, parsimony = 1 / 1000):
    return loss * (1 + complexity * parsimony)

@contextmanager
def time_limit(seconds: float, msg=''):
    """
    Timing class Multi-threaded run throws TimeoutError error after seconds
    """
    if len(threading.enumerate()) >= 2:
        for th in threading.enumerate():
            if th.name.count('Timer') > 0:
                th._stop()
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    try:
        timer.start()
        yield
    except KeyboardInterrupt:
        raise TimeoutError(msg)
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()