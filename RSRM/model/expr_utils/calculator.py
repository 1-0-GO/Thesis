import math
import warnings
from typing import Tuple, Dict

import numpy as np
import sympy as sp
from sympy import Add, Mul, Pow, Expr
import numexpr as ne
from scipy.optimize import minimize
import warnings
import math
from typing import Optional, Tuple
from functools import lru_cache

ne.set_num_threads(1)

# === Updated Library Code with LRU Cache and Resiliency ===

# TODO: Have math_funcs be dynamic
_MATH_FUNCS = {
               'exp', 'log', 
            #    'sqrt', 
            #    'tanh', 'cosh', 'sinh',
            #    'sin', 'cos', 'tan', 'arcsin', 'arctan'
               }

from model.config import Config
from model.expr_utils.utils import time_limit, FinishException, Solution, complexity_calculation


def process_symbol_with_C(symbols: str, c: np.ndarray) -> str:
    
    """
    Replacing parameter placeholders with real parameters
    :param symbols: expressions
    :param c: parameter
    :return: Converted expression
    >>>process_symbol_with_C('C1*X1+C2*X2',np.array([2.1,3.3])) -> '2.1*X1+3.3*X2'
    """
    for idx, val in enumerate(c):
        symbols = symbols.replace(f"C{idx + 1}", str(val))
    return symbols

import re

_pattern_funcs = re.compile(
    r"(?:"
    + "|".join(
        rf"{fn}\(\s*C\s*\)"   # for each fn, match fn(   C   )
        for fn in sorted(_MATH_FUNCS)
    )
    + r")"
)
_pattern_mul   = re.compile(r"\b[0-9]+(?:\.[0-9]+)?\*C\b|C\*[0-9]+(?:\.[0-9]+)?|C\*C|C\*\*[0-9]+(?:\.[0-9]+)?")  
_pattern_dup = re.compile(r'C\s*\+\s*C(?![\*/])')

def prune_poly_c(eq: str) -> str:
    s = eq
    prev = None
    while s != prev:
        prev = s
        # 1) Collapse functions
        s = _pattern_funcs.sub("C", s)
        # 2) Collapse numeric multipliers and C*C
        s = _pattern_mul.sub("C", s)
        # 3) Collapse duplicate C + C until stable
        s = _pattern_dup.sub("C", s)
    return s


def _get_numexpr_model(symbols: str):
    """
    Turn an infix string like "C1*X1 + exp(C2*X2)" into a real
    def model(X, c): ...  that calls np.exp etc.
    """

    return ne.NumExpr(symbols)


def cal_expression_single(model, 
                          t: np.ndarray,
                          params: list,
                          loss_fn) -> Tuple[np.ndarray,float]:
    """
    Evaluate `symbols` on X, c via an numexpr-compiled model,
    then compute loss_fn(pred, t).  Returns (pred, loss).
    """
    try:
        pred = model.run(*params)
        err = loss_fn(pred, t)
        new_err = np.asarray(err).astype(float)
        if math.isinf(new_err) or math.isnan(new_err):
            return None, 1e999
    except OverflowError:
        return None, 1.23e99
    except ValueError:
        return None, 2.34e99
    except NameError:
        return None, 3.45e99
    except ArithmeticError:
        return None, 4.56e99

    return pred, err


def make_loss_fn(model: ne.NumExpr, x, t, loss_fn):
    inames = model.input_names
    # identify which positions in the arg‐tuple are X's vs C's
    x = np.ascontiguousarray(x, dtype=np.float64)
    t = np.ascontiguousarray(t, dtype=np.float64)
    x_slots = []
    c_slots = []
    for pos, name in enumerate(inames):
        if name.startswith("X"):
            idx = int(name[1:]) - 1
            x_slots.append((pos, x[idx]))
        elif name.startswith("C"):
            c_slots.append(pos)
        else:
            raise ValueError("Unknown variable name")
    # build a single list with X's prefilled and dummy zeros for C's
    args_list = [None] * len(inames)
    for pos, x_arr in x_slots:
        args_list[pos] = x_arr
    for pos in c_slots:
        args_list[pos] = 0.0   
    def loss_only(c_vec):
        # overwrite only the C slots
        for j, pos in enumerate(c_slots):
            args_list[pos] = c_vec[j]
        return cal_expression_single(model, t, args_list, loss_fn)[1]
    return loss_only


class EarlyStopper(Exception): 
    def __init__(self, xk, loss):
        self.xk = xk
        self.loss = loss
        super().__init__("Early stopping triggered")

def make_callback(threshold, func):
    def callback(xk):
        loss = func(xk)
        if loss < threshold:
            raise EarlyStopper(xk, loss)
    return callback


def replace_parameter_and_calculate(symbols: str,
                                    x: np.ndarray,
                                    t: np.ndarray,
                                    config_s: Config,
                                    target_loss: float,
                                    x0: float = None) -> Tuple[float,str]:
    """
    Fits constants in `symbols` (if config_s.const_optimize),
    then returns (best_loss, symbols_with_values).
    """
    minim_kwargs = {}
    c_len = symbols.count("C")
    model = _get_numexpr_model(symbols)
    loss_only = make_loss_fn(model, x, t, config_s.loss)
    # Only worth it if we're using groups (only way for it to save time)
    if config_s.group:
        callback = make_callback(threshold=target_loss, func=loss_only)
        minim_kwargs["callback"] = callback

    if config_s.const_optimize and c_len > 0:
        # start from random, optimize up to 10 iters
        lower_bound = 1e-2/config_s.maxim  # Maximum X sets the scale. c*X > 0.01 for all features, smaller than that is too small a c
        x0 = x0 if x0 is not None else lower_bound + np.abs(np.random.randn(c_len))
        minim_kwargs.update({
            "method": "Powell",
            "bounds": [(lower_bound, None)] * c_len,      # enforce c_j > lower_bound
            "options": {"maxiter": 10, "ftol": 1e-3}
        })
        loss_only(x0)
        try:
            opt = minimize(loss_only,
                        x0=x0,
                        **minim_kwargs)
            best_c = opt.x
            best_loss = opt.fun
        except EarlyStopper as e:
            best_c = e.xk
            best_loss = e.loss
    else:
       return 1e999, x0 

    return best_loss, best_c


def expr_to_canonical(expr: Expr):
    """Recursively convert a sympy expression to a canonical tuple."""
    if expr.is_Atom:
        return (str(expr),)
    elif isinstance(expr, Add):
        args = frozenset(expr_to_canonical(arg) for arg in expr.args)
        return ('add', args)
    elif isinstance(expr, Mul):
        args = sorted(expr_to_canonical(arg) for arg in expr.args)
        return ('mul', tuple(args))
    elif isinstance(expr, Pow):
        base_can = expr_to_canonical(expr.base)
        exp_can = expr_to_canonical(expr.exp)
        return ('Pow', (base_can, exp_can))
    else:
        # Other functions like exp(...)
        func_name = expr.func.__name__
        args = tuple(expr_to_canonical(arg) for arg in expr.args)
        return (func_name, args)


class CalculatorArgs:
    def __init__(self, symbols: str, config_s, t_limit):
        self.symbols = symbols
        self.config_s = config_s
        self.t_limit = t_limit
        self.canonical_expression = expr_to_canonical(sp.sympify(symbols))
    def __hash__(self):
        return hash(self.canonical_expression)
    def __eq__(self, other):
        if not isinstance(other, CalculatorArgs):
            return NotImplemented
        return self.canonical_expression == other.canonical_expression
    def __repr__(self):
        return str(self.canonical_expression)


@lru_cache(maxsize=10_000)
def cached_cal_expression(cal_args: CalculatorArgs) -> Tuple[float, Dict[str,str]]:
    symbols = cal_args.symbols
    config_s = cal_args.config_s
    t_limit = cal_args.t_limit
    config_s.counter[0] += 1
    # fitted_expressions_per_group = {}
    loss = config_s.target_loss
    worst_group = None
    for group in sorted(config_s.worst_group_counts, key=config_s.worst_group_counts.get, reverse=True):
        x_train = config_s.x[group]
        t_train = config_s.t[group]
        with time_limit(t_limit):
            loss_train, best_c = replace_parameter_and_calculate(symbols, x_train, t_train, config_s, loss)
            eq_replaced_C = process_symbol_with_C(symbols, best_c)
            # fitted_expressions_per_group[group] = eq_replaced_C
            if loss_train > 1e-10 + loss:
                loss = loss_train
                worst_group = group
    config_s.worst_group_counts[worst_group] += 1
    if config_s.best_exp[1] > 1e-10 + loss:
        config_s.best_exp = eq_replaced_C, loss
    complexity = complexity_calculation(symbols)
    # print(eq_replaced_C, loss)
    config_s.pf.update_one(Solution(eq_replaced_C, complexity, loss))
    if loss <= config_s.target_loss:
        raise FinishException
    config_s.counter[2] += 1
    return loss


def cal_expression(symbols: str, config_s: Config, t_limit: float = 0.2) -> Tuple[float, Dict[str,str]]:
    """
    Calculate the value of the expression in train and test dataset
    :param symbols: target expression, contains parameter C
    :param config_s: config file, used to determine whether to optimize parameters or not and store independent variable
     and result
    :param t_limit: time limit of calculation in because of time in optimization
    :return: sum of the error of expression in train and test dataset
    """
    warnings.filterwarnings('ignore')
    try:
        symbols = sp.sympify(symbols)
        symbols_xpand = str(sp.expand(symbols))
        symbols = str(symbols)
        if symbols.count('zoo') or symbols.count('nan') or symbols.count('I'):
            return 1e999, symbols
        symbols_xpand = 'C*' + symbols_xpand.replace(' + ', ' + C*').replace(' - ', ' - C*')
        c_len = symbols_xpand.count('C')
        symbols_xpand = prune_poly_c(symbols_xpand)
        symbols_xpand = symbols_xpand.replace('C', 'PPP')  # replace C with C1,C2...
        for i in range(1, c_len + 1):
            symbols_xpand = symbols_xpand.replace('PPP', f'C{i}', 1)
        cal_args = CalculatorArgs(symbols_xpand, config_s, t_limit)
        return cached_cal_expression(cal_args), symbols
    except TimeoutError as e:
        config_s.counter[1] += 1
        print("**Timed out** ", end="")
    except RuntimeError as e:
        print("**Runtime Error** ", end="")
    except OverflowError as e:
        print("**Overflow Error** ", end="")
    except ValueError as e:
        print("**Value Error** ", end="")
    except MemoryError as e:
        print("**Memory Error** ", end="")
    except SyntaxError as e:
        print("**Syntax Error** ", end="")
    except Exception as e:
        pass
    print(f"Equation = {symbols}.")
    return 1e999, symbols