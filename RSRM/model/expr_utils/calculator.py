import math
import warnings
from typing import Tuple, Optional, Dict, List

import numpy as np
import sympy as sp
from numpy import sqrt, e as E, exp, sin, cos, log, inf, pi, tan, cosh, sinh, tanh, nan, seterr, arcsin, arctan
from scipy.optimize import minimize
import warnings

from model.config import Config
from model.expr_utils.utils import time_limit, FinishException, Solution, complexity_calculation

math_namespace = {
    "sqrt": sqrt,
    "E": E,
    "exp": exp,
    "sin": sin,
    "cos": cos,
    "log": log,
    "inf": inf,
    "pi": pi,
    "tan": tan,
    "cosh": cosh,
    "sinh": sinh,
    "tanh": tanh,
    "nan": nan,
    "arcsin": arcsin,
    "arctan": arctan,
}


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


def cal_expression_single(symbols: str, x: np.ndarray, t: np.ndarray, c: Optional[np.ndarray], loss) -> float:
    """
    Calculate the value of an expression with  `once` and compute the error rmse
    :param symbols: target expressions
    :param x: independent variable
    :param t: result or dependent variable
    :param c: parameter or None if there is no paramter
    :return: Calculated value of function at all points and its associated loss or 1e999 if error occurs
    """
    from numpy import inf, seterr
    zoo = inf
    seterr(all="ignore")
    I = complex(0, 1)
    for idx, val in enumerate(x):
        locals()[f'X{idx + 1}'] = val
    with warnings.catch_warnings(record=False) as caught_warnings:
        try:
            if c is not None:
                target = process_symbol_with_C(symbols, c)
            else:
                target = symbols
            cal = eval(target)
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


def cal_loss_expression_single(symbols: str, x: np.ndarray, t: np.ndarray, c: Optional[np.ndarray], loss) -> float:
    """
    :param symbols: target expressions
    :param x: independent variable
    :param t: result or dependent variable
    :param c: parameter or None if there is no paramter
    :return: Loss of function or 1e999 if error occurs
    """
    values, loss = cal_expression_single(symbols, x, t, c, loss)
    return loss


def replace_parameter_and_calculate(symbols: str, gradient_symbols: str, x: np.ndarray, t: np.ndarray, config_s: Config) -> Tuple[float, str]:
    """
    Calculate the value of the expression, the process is as follows
    1. Determine whether the parameter is included or not, if not, calculate directly
    2. Replace the parameter C in the expression with C1,C2.... CN
    3. Initialize the parameters
    4. Optimize the parameters if config_s.const_optimize, calculate the best parameters
    5. Fill the expression with the best parameters and return the expression with RMSE

    :param symbols: target expression, contains parameter C
    :param x: independent variable
    :param t: result or dependent variable
    :param config_s: config file, used to determine whether to optimize parameters or not
    :return: error, the expression containing the best parameters
    """
    c_len = symbols.count('C')

    if config_s.const_optimize:  # const optimize
        x0 = np.random.randn(c_len)
        x_ans = minimize(lambda c: cal_loss_expression_single(symbols, x, t, c, loss=config_s.loss),
                         x0=x0, 
                         method='Powell', 
                         options={'maxiter': 10, 'ftol': 1e-3, 'xtol': 1e-3})
        x0 = x_ans.x
    else:
        x0 = np.ones(c_len)
    val = cal_loss_expression_single(symbols, x, t, x0, loss=config_s.loss)
    eq_replaced_c = process_symbol_with_C(symbols, x0)
    return val, eq_replaced_c


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
