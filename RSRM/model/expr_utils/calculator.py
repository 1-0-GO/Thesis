import math
import warnings
from typing import Tuple, Dict

import numpy as np
import sympy as sp
import ast
from scipy.optimize import minimize
import warnings
import math
from typing import Optional, Tuple

warnings.filterwarnings("error")

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
_pattern_mul   = re.compile(r"\b[0-9]+(?:\.[0-9]+)?\*C\b|C\*[0-9]+(?:\.[0-9]+)?|C\*C")  
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

from functools import lru_cache

# 1) Cached model‐builder
@lru_cache(maxsize=10_000)
def _get_ast_model(expr):
    expr_ast = ast.parse(expr, mode='eval').body
    class ReplaceNames(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id.startswith('X'):
                idx = int(node.id[1:]) - 1
                return ast.Subscript(
                    value=ast.Name(id='X', ctx=ast.Load()),
                    slice=ast.Index(ast.Constant(idx)), ctx=node.ctx)
            if node.id.startswith('C'):
                idx = int(node.id[1:]) - 1
                return ast.Subscript(
                    value=ast.Name(id='c', ctx=ast.Load()),
                    slice=ast.Index(ast.Constant(idx)), ctx=node.ctx)
            # Map function names to numpy.<name>
            if node.id in _MATH_FUNCS:
                return ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
                                     attr=node.id, ctx=node.ctx)
            return node

    transformer = ReplaceNames()
    new_body = transformer.visit(expr_ast)
    ast.fix_missing_locations(new_body)

    func_def = ast.FunctionDef(
        name='f_ext',
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='X', annotation=None), ast.arg(arg='c', annotation=None)],
            vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=[ast.Return(new_body)],
        decorator_list=[]
    )
    module = ast.Module(body=[func_def], type_ignores=[])
    ast.fix_missing_locations(module)

    env = {'np': np}
    exec(compile(module, '<ast_ext>', 'exec'), env, env)
    return env['f_ext']


def cal_expression_single(model, 
                          t: np.ndarray,
                          x,
                          c_vec,
                          loss_fn) -> Tuple[np.ndarray,float]:
    """
    Evaluate `symbols` on X, c via an Autograd-compiled model,
    then compute loss_fn(pred, t).  Returns (pred, loss).
    """
    try:
        pred = model(x, c_vec)
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


def make_loss_fn(model, x, t, loss_fn):
    def loss_only(c_vec):
        # overwrite only the C slots
        return cal_expression_single(model, t, x, c_vec, loss_fn)[1]
    return loss_only


def replace_parameter_and_calculate(symbols: str,
                                    x: np.ndarray,
                                    t: np.ndarray,
                                    config_s) -> Tuple[float,str]:
    """
    Fits constants in `symbols` (if config_s.const_optimize),
    then returns (best_loss, symbols_with_values).
    """
    c_len = symbols.count("C")
    model = _get_ast_model(symbols)
    loss_only = make_loss_fn(model, x, t, config_s.loss)

    if config_s.const_optimize and c_len > 0:
        # start from random, optimize up to 16 iters
        step = 1e-2/config_s.maxim  # So that c*X > 0.01 for all features, smaller than that is too small a bump
        ftol = 1e-2 * config_s.best_exp[1]  # 0.01 * loss is our tolerance
        x0 = step + np.abs(np.random.randn(c_len))
        return loss_only(x0), process_symbol_with_C(symbols, x0)
        minim_kwargs = {
            "method": "Powell",
            "bounds": [(step, None)] * c_len,      # enforce c_j > 0
            "options": {"maxiter":10,"ftol":ftol,"xtol":step}
        }
        opt = minimize(loss_only,
                       x0=x0,
                       **minim_kwargs)
        best_c = opt.x
    else:
       return 1e999, symbols 

    best_loss = loss_only(opt.x)
    filled_expr = process_symbol_with_C(symbols, best_c)
    return best_loss, filled_expr


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
    config_s.symbol_tol_num += 1
    config_s.count[0] += 1
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
        fitted_expressions_per_group = {}
        total_loss = 0
        num_elements_dataset = 0
        # Train
        for group in config_s.x:
            x_train = config_s.x[group]
            t_train = config_s.t[group]
            with time_limit(t_limit):
                loss_train, eq_replaced_C = replace_parameter_and_calculate(symbols_xpand, x_train, t_train, config_s)
                fitted_expressions_per_group[group] = eq_replaced_C
                # Losses are already weighted by size of dataset by using sum instead of mean
                total_loss += loss_train
                num_elements_dataset += t_train.shape[0]
        # Test
        # for group in config_s.x_:
        #     x_test = config_s.x[group]
        #     t_test = config_s.t[group]
        #     eq_replaced_C = fitted_expressions_per_group[group]
        #     with time_limit(t_limit):
        #         loss_test, eq_replaced_C = replace_parameter_and_calculate(eq_replaced_C, x_test, t_test, config_s)
        #         total_loss += loss_test
        #         num_elements_dataset += t_test.shape[0]
        # Loss
        loss = total_loss / num_elements_dataset
        if config_s.best_exp[1] > 1e-10 + loss:
            config_s.best_exp = eq_replaced_C, loss
        complexity = complexity_calculation(symbols_xpand)
        # print(eq_replaced_C, loss)
        config_s.pf.update_one(Solution(eq_replaced_C, complexity, loss))
        if loss <= config_s.reward_end_threshold:
            raise FinishException
        config_s.count[2] += 1
        return loss, fitted_expressions_per_group
    except TimeoutError as e:
        config_s.count[1] += 1
        print("**Timed out** ", end="")
    # except RuntimeError as e:
    #     print("**Runtime Error** ", end="")
    # except OverflowError as e:
    #     print("**Overflow Error** ", end="")
    # except ValueError as e:
    #     print("**Value Error** ", end="")
    # except MemoryError as e:
    #     print("**Memory Error** ", end="")
    # except SyntaxError as e:
    #     print("**Syntax Error** ", end="")
    # except Exception as e:
    #     pass
    print(f"Equation = {symbols}.")
    return 1e999, None
