import math
import warnings
from typing import Tuple, Optional, Dict, List

import numpy as np
import sympy as sp
from autograd import numpy as anp
from autograd import grad
from scipy.optimize import minimize
import warnings
import math
from typing import Optional, Tuple
from functools import lru_cache

# === Updated Library Code with LRU Cache and Resiliency ===

_MATH_FUNCS = {'exp', 'log', 'sin', 'cos', 'sqrt', 'tan', 'cosh', 'sinh',
               'tanh', 'arcsin', 'arctan'}

from model.config import Config
from model.expr_utils.utils import time_limit, FinishException, Solution, complexity_calculation


def prune_poly_c(eq: str) -> str:
    """
    Reducing multiple parameters in a parameterized expression to a single parameter
    :param eq: expression string
    :return: the modified expression
    >>> prune_poly_c('C*C+C+X1*C')->"C+C*X1"
    """
    for i in range(5):
        eq_l = eq
        c_poly = ['C**' + str(i) + ".5" for i in range(1, 4)]
        c_poly += ['C**' + str(i) + ".25" for i in range(1, 4)]
        c_poly += ['C**' + str(i) for i in range(1, 4)]
        c_poly += [' ' + str(i) + "*C" for i in range(1, 4)]
        for c in c_poly:
            if c in eq:
                eq = eq.replace(c, 'C')
        eq = eq.replace('-C', 'C')
        eq = eq.replace('C*C', 'C')
        eq = eq.replace('exp(C)', 'C')
        eq = str(sp.sympify(eq))
        if eq == eq_l:
            break
    return eq

@lru_cache(maxsize=None)
def _make_func_from_expr_ast(symbols: str):
    """
    Compile a symbolic expression into a fast Python function f(X, c) -> array.
    Supports X1...Xn, C1...Cm, and numpy math funcs in _MATH_FUNCS.
    """
    expr_ast = ast.parse(symbols, mode='eval').body

    class ReplaceNames(ast.NodeTransformer):
        def visit_Name(self, node):
            # Map X1... -> X[idx]
            if node.id.startswith('X'):
                idx = int(node.id[1:]) - 1
                return ast.Subscript(
                    value=ast.Name(id='X', ctx=ast.Load()),
                    slice=ast.Index(ast.Constant(idx)), ctx=node.ctx)
            # Map C1... -> c[idx]
            if node.id.startswith('C'):
                idx = int(node.id[1:]) - 1
                return ast.Subscript(
                    value=ast.Name(id='c', ctx=ast.Load()),
                    slice=ast.Index(ast.Constant(idx)), ctx=node.ctx)
            # Map math functions to numpy
            if node.id in _MATH_FUNCS:
                return ast.Attribute(
                    value=ast.Name(id='np', ctx=ast.Load()),
                    attr=node.id,
                    ctx=node.ctx
                )
            return node

    transformer = ReplaceNames()
    new_body = transformer.visit(expr_ast)
    ast.fix_missing_locations(new_body)

    func_def = ast.FunctionDef(
        name='f_expr',
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg='X', annotation=None), ast.arg(arg='c', annotation=None)],
            vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
        ),
        body=[ast.Return(new_body)],
        decorator_list=[]
    )
    module = ast.Module(body=[func_def], type_ignores=[])
    ast.fix_missing_locations(module)
    env = {'np': np}
    exec(compile(module, '<ast_expr>', 'exec'), env, env)
    return env['f_expr']

def cal_expression_single(symbols: str,
                          x: np.ndarray,
                          t: np.ndarray,
                          c: Optional[np.ndarray],
                          loss) -> Tuple[Optional[np.ndarray], float]:
    """
    Compute model output and loss for one expression at parameters c.
    Uses an AST-compiled function for speed instead of eval.
    Returns (values, loss) or (None, large_error) on failure.
    """
    X = x
    if c is None:
        c = np.zeros(0)
    func = _make_func_from_expr_ast(symbols)
    with warnings.catch_warnings(record=False) as caught_warnings:
        try:
            cal = func(X, c)
            ans = loss(cal, t)
            if math.isinf(ans) or math.isnan(ans) or caught_warnings:
                return None, 1e999
        except OverflowError:  # if error occurs, return --e9 as high error.
            return None, 1.23e9
        except ValueError:
            return None, 2.34e9
        except NameError:
            return None, 3.45e9
        except ArithmeticError:
            return None, 4.56e9
    return cal, ans

def cal_loss_expression_single(symbols: str,
                               x: np.ndarray,
                               t: np.ndarray,
                               c: Optional[np.ndarray],
                               loss) -> float:
    _, val = cal_expression_single(symbols, x, t, c, loss)
    return val

def replace_parameter_and_calculate(symbols: str,
                                    gradient_symbols: str,
                                    x: np.ndarray,
                                    t: np.ndarray,
                                    config_s) -> Tuple[float, str]:
    """
    Optimize parameters in-symbols expression and return (best_loss, filled_expression).
    """
    # determine count of C placeholders
    c_len = symbols.count('C')
    if config_s.const_optimize:
        x0 = np.random.randn(c_len)
        res = minimize(lambda c: cal_loss_expression_single(symbols, x, t, c, config_s.loss),
                       x0=x0,
                       method='Powell',
                       options={'maxiter': 10,
                                'ftol': 1e-3,
                                'xtol': 1e-3})
        c_opt = res.x
    else:
        c_opt = np.ones(c_len)

    best_loss = cal_loss_expression_single(symbols, x, t, c_opt, config_s.loss)
    # fill expression
    filled = symbols
    for idx, val in enumerate(c_opt):
        filled = filled.replace(f"C{idx+1}", str(val))
    return best_loss, filled


def cal_expression(symbols: str, config_s: Config, t_limit: float = 0.2) -> Tuple[float, Dict[str, str]]:
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
        if symbols.count('zoo') or symbols.count('nan'):
            return 1e999, symbols
        symbols_xpand = 'C*' + ' + C*'.join(symbols_xpand.split(' + '))
        c_len = symbols_xpand.count('C')
        symbols_xpand = prune_poly_c(symbols_xpand)
        C = [f'C{i}' for i in range(1, c_len + 1)]
        symbols_xpand = symbols_xpand.replace('C', 'PPP')  # replace C with C1,C2...
        for i in range(1, c_len + 1):
            symbols_xpand = symbols_xpand.replace('PPP', f'C{i}', 1)
        gradient_expr = None
        fitted_expressions_per_group = {}
        total_loss = 0
        num_elements_dataset = 0
        # Train
        for group in config_s.x:
            x_train = config_s.x[group]
            t_train = config_s.t[group]
            with time_limit(t_limit):
                loss_train, eq_replaced_C = replace_parameter_and_calculate(symbols_xpand, gradient_expr, x_train, t_train, config_s)
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
        # print("**Timed out** ", end="")
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
    # print(f"Equation = {symbols}.")
    return 1e999, None
