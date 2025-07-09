import math
import warnings
from typing import Tuple, Dict

import numpy as np
import sympy as sp
from autograd import numpy as anp
from autograd import grad
from scipy.optimize import basinhopping
import warnings
import math
from typing import Optional, Tuple
from functools import lru_cache

# === Updated Library Code with LRU Cache and Resiliency ===

# TODO: Have it be dynamic
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
        c_poly += [str(i) + "*C" for i in range(1, 4)]
        c_poly += ["C*" + str(i) for i in range(1, 4)]
        for c in c_poly:
            if c in eq:
                eq = eq.replace(c, 'C')
        eq = eq.replace('C*C', 'C')
        for func in _MATH_FUNCS:
            eq = eq.replace(f'{func}(C)', 'C')
        eq = str(sp.sympify(eq))
        if eq == eq_l:
            break
    return eq

# 1) Cached model‐builder
@lru_cache(maxsize=10_000)
def _get_autograd_model(symbols: str, n_features: int, n_params: int):
    """
    Turn an infix string like "C1*X1 + exp(C2*X2)" into a real
    def model(X, c): ...  that calls anp.exp etc.
    """
    # Step 1: rewrite placeholders
    code = symbols
    # X1 → X[0], X2 → X[1], …
    for i in range(n_features):
        code = code.replace(f"X{i+1}", f"X[{i}]")
    # C1 → c[0], …
    for j in range(n_params):
        code = code.replace(f"C{j+1}", f"c[{j}]")
    # Step 2: map math funcs to anp.
    for fn in _MATH_FUNCS:
        code = code.replace(f"{fn}(", f"np.{fn}(")
    # Step 3: wrap into a real def
    fn_src = "def model(X, c):\n    return " + code
    ns = {"np": anp}
    exec(compile(fn_src, "<model>", "exec"), ns)
    model_py = ns["model"]

    return model_py


def cal_expression_single(symbols: str,
                          x: np.ndarray,
                          t: np.ndarray,
                          c: np.ndarray,
                          model,
                          loss_fn) -> Tuple[np.ndarray,float]:
    """
    Evaluate `symbols` on X, c via an Autograd-compiled model,
    then compute loss_fn(pred, t).  Returns (pred, loss).
    """

    try:
        # run model (suppress numpy warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = model(x, c) if c is not None else model(x, np.zeros(0))
        err = loss_fn(pred, t)
        new_err = np.asarray(err).astype(float)
        if math.isinf(new_err) or math.isnan(new_err):
            return None, 1e999
    except OverflowError:
        return None, 1.23e9
    except ValueError:
        return None, 2.34e9
    except NameError:
        return None, 3.45e9
    except ArithmeticError:
        return None, 4.56e9

    return pred, err


def cal_loss_expression_single(symbols: str,
                               x: np.ndarray,
                               t: np.ndarray,
                               c: np.ndarray,
                               model,
                               loss_fn) -> float:
    """Just the loss portion of cal_expression_single."""
    _, err = cal_expression_single(symbols, x, t, c, model, loss_fn)
    return err


def replace_parameter_and_calculate(symbols: str,
                                    gradient_symbols: str,
                                    x: np.ndarray,
                                    t: np.ndarray,
                                    config_s) -> Tuple[float,str]:
    """
    Fits constants in `symbols` (if config_s.const_optimize),
    then returns (best_loss, symbols_with_values).
    """
    c_len = symbols.count("C")
    n_features = x.shape[0]

    # Loss-only wrapper for Autograd
    model = _get_autograd_model(symbols, n_features, c_len)
    def loss_only(c_vec):
        return cal_loss_expression_single(symbols, x, t, c_vec, model, config_s.loss)

    # Autograd gradient of the loss
    grad_loss = grad(loss_only)

    if config_s.const_optimize and c_len > 0:
        # start from random, optimize up to 10 iters
        x0 = np.abs(np.random.randn(c_len))
         # Pass bounds and jacobian to the local minimizer
        minim_kwargs = {
            "method": "L-BFGS-B",
            "bounds": [(1e-8, None)] * c_len,      # enforce c_j > 0
            "jac": lambda c: np.array(grad_loss(c)),
            "options": {"maxiter":100,"ftol":1e-4}
        }
        opt = basinhopping(
            func=loss_only,
            x0=x0,
            niter=16,
            minimizer_kwargs=minim_kwargs,
            disp=False
        )
        best_c = opt.x
    else:
       return 1e999, symbols 

    best_loss = cal_loss_expression_single(symbols, x, t, best_c, model, config_s.loss)
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
    return 1e999, None
