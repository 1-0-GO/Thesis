import pandas as pd
import sympy as sp
import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC
import re
import ast
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize, basinhopping, dual_annealing
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut

# Read and process dataset

def _identify_header(path, n=5, th=0.9):
    df1 = pd.read_csv(path, header='infer', nrows=n)
    df2 = pd.read_csv(path, header=None, nrows=n)
    sim = (df1.dtypes.values == df2.dtypes.values).mean()
    return 'infer' if sim < th else None

def load_dataset(path):
    header = _identify_header(f"{path}")
    csv = pd.read_csv(f"{path}", header=header)
    x, t = csv.iloc[:, :-1], csv.iloc[:, -1]
    return x, t

def read_train_test_indices(filename):
    """Reads train and test indices from a file."""
    with open(filename, 'r') as file:
        data = file.read()
    
    lines = data.strip().split('\n')
    tuples = [ast.literal_eval(line) for line in lines]
    
    return tuples

def get_fold_wrapper(X, y, folds):
    def get_fold(run, preprocess=normalize_by_iqr):
        fold = folds[run-1]
        train_index, test_index = fold
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if preprocess:
            X_train_s = preprocess(X_train)
            X_test_s = preprocess(X_test, X_train)
        return X_train_s, y_train, X_test_s, y_test
    return get_fold

def compute_iqr(X):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    range_ = np.max(X, axis=0) - np.min(X, axis=0)
    # Replace zero IQR values with corresponding range values to avoid division by zero
    scaling_factor = np.where(IQR == 0, range_, IQR)
    return scaling_factor

def gini_mean_difference(col):
    n = col.size
    diffs = np.abs(col[:, None] - col)  # Pairwise absolute differences
    gmd = np.sum(diffs) / (n * (n - 1))  # Average over all pairs
    return gmd

def compute_gmd(X):
    return X.apply(lambda col: gini_mean_difference(col.values), axis=0)

def normalize_by(X, scaling_factor_function, X_train=None):
    """
    Normalize X using a sclaing factor computed from X_train.
    
    Parameters:
        X: np.ndarray
            The data to be normalized.
        scaling_factor_function: function
            How do compute the scaling factor from a dataset
        X_train: np.ndarray, optional
            The reference dataset for computing IQR (default is X itself).
    
    Returns:
        np.ndarray: The normalized data.
    """
    if X_train is None:
        X_train = X
    scaling_factor = scaling_factor_function(X_train)
    return X / scaling_factor

def normalize_by_iqr(X, X_train=None):
    return normalize_by(X, compute_iqr, X_train)

def normalize_by_gmd(X, X_train=None):
    return normalize_by(X, compute_gmd, X_train)


# Custom functions

def harmonic(x, y):
    return 2 * x * y / (x + y)

def saturation(x, k):
    return k * x / (k + x)

def damping(x, k):
    return x / (1.0 + k * x)

def threshold(x, k):
    return max(0, x - k)

def inverse(x):
    return 1.0/(x+1e-6)

def sigmoid(x):
    return 1.0 / (1.0 + safe_exp(-x))

def square(x):
    return x * x

def cond(x, y):
    return np.where(x > 0, y, 0) 

def safe_exp(x):
    return np.exp(np.clip(x, -30, 30))

def safe_power(x, y):
    neg_mask = (x < 0) & (y % 2 == 1)
    x = np.abs(x)
    return neg_mask * safe_exp(y * np.log(x))


# Equation utilities

def custom_round(x):
    if x == 0:
        return "0"
    elif x < 0.1:
        return "{:.2e}".format(x)
    else:
        return str(round(x, 2))

def round_constants_in_equation(equation):
    # Ensure the equation is treated as a single string
    if isinstance(equation, list):
        equation = ''.join(map(str, equation))
    
    # Use regex to find all numbers in the equation and round them
    def round_match(match):
        value = float(match.group())
        return f"{custom_round(value)}"
    
    # Match numbers including those in scientific notation
    rounded_equation = re.sub(r'-?\d+\.\d+(e[+-]?\d+)?', round_match, equation)
    return rounded_equation

class Equation(ABC):
    def __init__(self, eq_str, variables, use_glm=False):
        self.equation = eq_str
        self.skeleton = replace_numeric_parameters(eq_str)
        if use_glm:
            glm_variables = [var for var in variables if var in self.skeleton]
            self.skeleton = 'C + C*' + ' + C*'.join(glm_variables)
        self.original_Cs = extract_numeric_parameters(eq_str)
        self.fitted_Cs = None
        self.variables = variables
        self.lambda_function = None
        self.use_glm = use_glm
    
    def __str__(self):
        return self.skeleton

    def get_original_equation_string(self):
        return round_constants_in_equation(self.equation)

    def get_fitted_equation_string(self):
        Cs = self.fitted_Cs
        if Cs is None:
            return self.get_original_equation_string()
        return round_constants_in_equation(replace_optimized_parameters(self.skeleton, Cs))
    
    @property
    def fitted_params(self):
        return {'Cs': self.fitted_Cs}
    
    def fit(self, X, y, basinhop=False, **minimizer_kwargs):
        Cs, min_loss = optimize_equation(self.skeleton, X, y, self.variables, self.loss, self.original_Cs, basinhop, self.use_glm, **minimizer_kwargs)
        self.fit_with_params({'Cs': Cs})
    
    def fit_with_params(self, params):
        Cs = params['Cs']  # Extract Cs from params
        equation_str_fitted = replace_optimized_parameters(self.skeleton, Cs)
        self.lambda_function = lambdify_equations([equation_str_fitted], variables=self.variables)[0]
        self.fitted_Cs = Cs
    
    def predict(self, X):
        if self.lambda_function is None:
            raise ValueError("Equation not fitted. Call fit() before predict().")
        return self.lambda_function(X)
    
    def __getstate__(self):
        # Return a dictionary of attributes to pickle, excluding 'lambda_func'
        state = self.__dict__.copy()
        del state['lambda_func']
        return state
    
    def __setstate__(self, state):
        # Restore the object's state and reinitialize 'lambda_func'
        self.__dict__.update(state)
        if self.fitted_params is not None:
            self.fit_with_params(self.fitted_params)


class PoissonEquation(Equation):
    def __init__(self, eq_str, variables, use_glm=False):
        super().__init__(eq_str, variables, use_glm)
        self.loss = 'poisson'
        if use_glm:
            glm_variables = [var for var in variables if var in self.skeleton]
            self.skeleton = 'exp(C + ' + 'C*' + ' + C*'.join(glm_variables) + ')'

class BernoulliEquation(Equation):
    def __init__(self, eq_str, variables, use_glm=False):
        super().__init__(eq_str, variables, use_glm)
        self.loss = 'bce'
        self.threshold = 0.5
        if use_glm:
            glm_variables = [var for var in variables if var in self.skeleton]
            self.skeleton = 'C + ' + 'C*' + ' + C*'.join(glm_variables)
    
    @property
    def fitted_params(self):
        return {'Cs': self.fitted_Cs, 'threshold': self.threshold}
    
    def fit(self, X, y, basinhop=False, k_fit_threshold=3, **minimizer_kwargs):
        Cs, min_loss = optimize_equation(self.skeleton, X, y, self.variables, self.loss, self.original_Cs, basinhop, self.use_glm, **minimizer_kwargs)
        if k_fit_threshold:
            self.fit_threshold(X, y, k=k_fit_threshold)
        self.fit_with_params({'Cs': Cs, 'threshold': self.threshold})

    def fit_with_params(self, params):
        super().fit_with_params(params)
        self.threshold = params['threshold']
    
    def predict_probas(self, X):
        return sigmoid(super().predict(X))

    def predict(self, X, threshold=None):
        return binarize_predictions(self.predict_probas(X), threshold or self.threshold)
    
    def fit_threshold(self, X, y, k=5):
        """
        Perform k-fold cross-validation to find the best threshold.
        
        Parameters:
            X: Features dataset (numpy array).
            y: True labels (numpy array).
            k: Number of folds for cross-validation.

        Returns:
            best_threshold: The threshold that maximizes the TSS metric across folds.
        """
        original_lambda = self.lambda_function
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        thresholds = []

        for train_idx, val_idx in skf.split(X, y):
            # Split the data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self.fit(X_train, y_train, k_fit_threshold=None)

            # get_threshold function finds the best threshold for this fold
            threshold = get_threshold(self.lambda_function, X_val, y_val)
            thresholds.append(threshold)

        # Average thresholds across all folds
        best_threshold = np.mean(thresholds)
        self.lambda_function = original_lambda
        self.threshold = best_threshold


def extract_numeric_parameters(equation):
    """
    Extract all numeric parameters (integers, reals, positive, or negative) from an equation string.

    Args:
        equation (str): The equation string (e.g., "-1.510*D + 5e-3*cond(D - 0.54, 5 - T)")

    Returns:
        list: A list of numeric parameters as floats or integers.
    """
    # Regex to match numeric parameters
    pattern = r'(?<![a-zA-Z_.])-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?![a-zA-Z_.])'
    matches = re.findall(pattern, equation)
    
    # Convert matches to numeric types (int or float)
    numeric_params = []
    for match in matches:
        if '.' in match or 'e' in match.lower():
            numeric_params.append(float(match))
        else:
            numeric_params.append(int(match))
    
    return numeric_params

def replace_placeholders(equation_skeleton):
    params = []
    parts = []
    c_count = 0
    pattern = re.compile(r'\bC\b')
    last_end = 0
    
    for match in pattern.finditer(equation_skeleton):
        start, end = match.start(), match.end()
        parts.append(equation_skeleton[last_end:start])
        parts.append(f'C{c_count}')
        params.append(sp.symbols(f'C{c_count}'))
        c_count += 1
        last_end = end
    parts.append(equation_skeleton[last_end:])
    new_eq_str = ''.join(parts)
    expr = sp.sympify(new_eq_str)
    return expr, params

def remove_sigmoid(equation_str):
    """
    Removes the sigmoid function and its parentheses from a string.

    Args:
        equation_str (str): The equation string (e.g., "sigmoid(0.98 - 2.05*D)")

    Returns:
        str: The equation without the sigmoid function and parentheses.
    """
    # Use regex to match and remove the sigmoid function and its parentheses
    match = re.match(r'sigmoid\((.*)\)', equation_str)
    if match:
        return match.group(1)
    else:
        return equation_str  # Return the original string if no sigmoid is found


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def poisson_deviance(y_true, y_pred):
    eps = 1e-10
    y_pred = np.asarray(y_pred).astype(float)
    prediction_nonpositive_mask = y_pred < eps
    prediction_positive_mask = ~prediction_nonpositive_mask

    target_np = y_true[prediction_nonpositive_mask]
    pred_np = y_pred[prediction_nonpositive_mask]

    target_p = y_true[prediction_positive_mask]
    pred_p = y_pred[prediction_positive_mask]

    # Continuous Piecewise Loss function
    loss_np = np.sum(target_np * np.log((target_np + eps) / eps) - (target_np + pred_np - 2 * eps)) # Decreases as the prediction becomes more positive, continuous with the other one at eps, just do pred_np = eps
    loss_p = np.sum(target_p * np.log((target_p + eps) / pred_p) - (target_p - pred_p))
    total_loss = loss_np + loss_p

    return total_loss

def binary_cross_entropy(y_true, y_pred):
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # Calculate binary cross-entropy loss
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def correlation_loss(y_true, y_pred):
    covariance = np.sum((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred))) / len(y_true)
    std_y_true = np.std(y_true)
    std_y_pred = np.std(y_pred)
    correlation = covariance / (std_y_true * std_y_pred)
    return -np.abs(correlation)

def optimize_equation(equation_skeleton, X, y, variables, loss='mse', initial_guess=None, basinhop=False, use_glm = False, minimizer_kwargs={}):
    X = X.values if hasattr(X, 'values') else np.asarray(X)
    y = np.asarray(y)
    var_columns = [X[:, i] for i in range(len(variables))]

    expr, params = replace_placeholders(equation_skeleton)
    if len(params) == 0:
        return [], 0

    # ================== GLM Optimization ==================
    if use_glm:
        # Check GLM compatibility: number of parameters = variables + 1 (intercept + coeffs)
        glm_variables = [var for var in variables if var in equation_skeleton]
        if len(params) != len(glm_variables) + 1:
            print(f"Parameter count mismatch (Num params: {len(params)}, Num vars: {len(glm_variables)}). Falling back to Powell.")
            use_glm = False
        else:
            family = None
            if loss == 'poisson':
                family = sm.families.Poisson()
            elif loss == 'bce':
                family = sm.families.Binomial()
            elif loss == 'mse':
                family = sm.families.Gaussian()
            else:
                print(f"GLM not supported for {loss} loss. Using Powell.")
                use_glm = False

        if use_glm:
            try:
                # Build formula: y ~ var1 + var2 + ...
                formula = 'y ~ ' + ' + '.join(glm_variables)
                
                # Create DataFrame with variables and target
                df = np.concatenate([X, y[:, np.newaxis]], axis=1)
                df = pd.DataFrame(df, columns=variables + ['y'])
                
                # Fit GLM
                model = smf.glm(formula=formula, data=df, family=family).fit()
                
                if loss == 'poisson' or loss == 'bce':
                    # For Poisson and BCE losses: intercept is the first parameter
                    intercept = model.params['Intercept']
                    coefficients = [model.params[var] for var in glm_variables]
                    glm_params = [intercept] + coefficients
                else:
                    # For other losses: intercept position depends on equation skeleton
                    intercept_pos = _find_intercept_position(equation_skeleton)
                    coefficients = [model.params[var] for var in glm_variables]
                    intercept = model.params['Intercept']
                    
                    # Insert intercept at the correct position
                    glm_params = coefficients.copy()
                    glm_params.insert(intercept_pos, intercept)
                
                return np.array(glm_params), model.deviance
                
            except Exception as e:
                print(f"GLM failed. Error: {e}. Formula: {formula}.\nFalling back to Powell.")
                use_glm = False

    # ================== Powell Optimization ==================

    var_symbols = sp.symbols(variables)
    
    modules = {
        '**': safe_power,
        'exp': safe_exp,
        'log': np.log,
        'sigmoid': sigmoid,
        'logistic': sigmoid,
        'sqrt': np.sqrt,
        'inv': inverse,
        'harmonic': harmonic,
        'saturation': saturation,
        'damping': damping,
        'threshold': threshold,
        'square': square,
        'cond': cond
    }
    
    args = var_symbols + params
    lambdified_func = sp.lambdify(args, expr, modules=modules)

    if loss == 'mse':
        loss_func = mse
    elif loss == 'poisson':
        loss_func = poisson_deviance
    elif loss == 'bce': 
        loss_func = lambda y_true, y_pred: binary_cross_entropy(y_true, sigmoid(y_pred))
    elif loss == 'exp': 
        loss_func = lambda y_true, y_pred: np.mean(np.exp(- y_true * y_pred))
    elif loss =='cov':
        loss_func = correlation_loss
    else:
        raise ValueError(f"Unsupported loss function: {loss}")
    
    def compute_loss(params_vals):
        # try:
            y_pred = lambdified_func(*var_columns, *params_vals)
        
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return np.inf
            
            return loss_func(y, y_pred)
        # except Exception as e:
            # return np.inf
    
    if initial_guess is None:
        temp = dual_annealing(compute_loss, [[-10,10]] * len(params))
        initial_guess = temp.x
    
    if basinhop: 
        result = basinhopping(compute_loss, initial_guess, minimizer_kwargs=minimizer_kwargs)
    else: 
        result = minimize(compute_loss, initial_guess, **minimizer_kwargs)
    return result.x, result.fun

def _find_intercept_position(equation_skeleton):
    """
    Find the position of the intercept placeholder (C) in the equation skeleton.
    Returns the index where the intercept should be inserted.
    """
    def remove_chars(string, chars_to_remove):
        for char in chars_to_remove:
            string = string.replace(char, "")
        return string

    # Remove function names and unnecessary characters
    cleaned_eq = remove_chars(equation_skeleton, ['exp', 'sigmoid', ' ', '(', ')'])

    # Split equation by addition and subtraction
    terms = re.split(r'[+-]', cleaned_eq)

    for i, term in enumerate(terms):
        if term == 'C':  
            return i  # Return the position of 'C'
    
    # If no standalone 'C' is found, assume it's at the end
    return len(terms)

def lambdify_equations(equations, variables):
    lambdified_eqs = []
    modules = {'**': safe_power, 'exp': safe_exp, 'log': np.log, 'sigmoid': sigmoid, 'logistic': sigmoid, 'sqrt': np.sqrt, 'inv': inverse, 
               'harmonic': harmonic, 'saturation': saturation, 'damping': damping, 'threshold': threshold,
               'square': square, 'cond': cond}

    for eq in equations:
        sym_expr = sp.sympify(eq, locals=locals())

        lambdified_eq = sp.lambdify(tuple(variables), sym_expr, modules)
        lambdified_eqs.append(wrapped_ll(lambdified_eq, len(variables)))

    return lambdified_eqs

def wrapped_ll(ll, var_num):
    def wrapper(data, threshold=None):
        # Ensure data is a 2D array
        data = np.asarray(data)
        if data.shape[1] != var_num:
            raise ValueError("Input data must have exactly 2 columns (observations by features)")
        result = ll(*[data[:, i] for i in range(var_num)]) * np.ones(data.shape[0])
        if threshold:
            return (sigmoid(result) >= threshold).astype(int)
        return result
    return wrapper


def replace_variables(expression, var_map):
    """Replaces variables in an expression"""
    for key, value in var_map.items():
        expression = expression.replace(key, value)
    return expression


def count_weighted_operations(expression):
    """Computes the complexity of an expression"""
    try:
        expression = str(sp.simplify(sp.sympify(expression)))

        # Define weights for each operation
        weights = {
            'const': 1,
            'vars': 1,
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
            'max': 1,
            'cond': 1,
            'inv': 1,
            'sigmoid': 1
        }
        
        # Find all variables in the expression
        vars = re.findall(r'X\d+|[A-Z]', expression)

        # Find all float constants in the expression
        consts = re.findall(r'[-+]?\d*\.\d+', expression)

        for v in vars:
            expression = expression.replace(v, '')

        for c in consts:
            expression = expression.replace(c, 'C')
        
        # Define a regex pattern for the operations
        pattern = r'cond|max|min|log|sin|cos|exp|\+|\-|\*\*|\*|\/|\bd+\b|inv|sigmoid'
        # Find all operations in the expression
        operations = re.findall(pattern, expression)

        # Calculate the total weight
        total_weight = sum(weights[op] for op in operations)
        
        standalone_digits = re.findall(r'[1-9]\d*', expression)
        
        return total_weight + len(vars) * weights['vars'] + len(consts) * weights['const'] + len(standalone_digits)
    except Exception as e:
        return np.iinfo(np.int32).max

def binarize_predictions(y_probas, threshold=0.5):
    """Binarize probabilities between 0 and 1 based on threshold"""
    return (y_probas >= threshold).astype(int)    

def get_threshold(lambda_function, X_val, y_val):
    """Find the threshold that maximizes the TSS metric"""
    y_probas = sigmoid(lambda_function(X_val))
    P = np.sum(y_val)
    N = len(y_val) - P

    # Sort probas and y in descending order
    sort_idx = np.argsort(y_probas)[::-1]
    sorted_probas = y_probas[sort_idx]
    sorted_y = y_val[sort_idx]
    # Number of positive predictions at each threshold
    cum_count_positive_predictions = np.arange(1, len(sorted_probas) + 1)
    # Compute number of true positives and false positives at each threshold
    cum_tp = np.cumsum(sorted_y)
    cum_fp = cum_count_positive_predictions - cum_tp
    # Compute sensitivities and specificities at each threshold
    all_sensitivities = cum_tp / P
    all_specificities = 1 - cum_fp / N
    # Compute tsss at each threshold
    all_tss = all_sensitivities + all_specificities - 1 
    # Find the threshold that maximizes TSS and is closest to 0.5
    best_threshold_idx= np.argmax(all_tss - 2e-3 * np.abs(sorted_probas - 0.5))
    best_threshold = sorted_probas[best_threshold_idx]

    return best_threshold

# Evaluation

def true_skill_statistic(y_true, y_pred, all=False):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() 
    sensitivity = tp / (tp + fn) 
    specificity = tn / (tn + fp) 
    tss = sensitivity + specificity - 1
    if all:
        return tss, sensitivity, specificity
    return tss

def pseudo_R2(y, y_pred):
    """Compute Pseudo R² for some model and task."""
    res = []
    log_likelihood = lambda y_true, y_pred: -np.sum(y_pred - y_true * np.log(y_pred + 1e-10) + gammaln(y_true + 1))
    
    # Log likelihood of the model with only the intercept
    y_mean = np.mean(y)
            
    intercept_log_likelihood = log_likelihood(y, np.full_like(y, y_mean))
    
    fitted_log_likelihood = log_likelihood(y, y_pred)
    return 1 - (fitted_log_likelihood / intercept_log_likelihood)

def mae(y, y_pred):
    return np.mean(np.abs(y - y_pred))

def loocv(equation: Equation, X, y, preprocess=normalize_by_iqr, prefitted_params=None):
    loo = LeaveOneOut()
    all_predictions = []
    all_targets = []
    all_params = []
    X = np.asarray(X)
    y = np.asarray(y)
    if prefitted_params != None and len(prefitted_params) == X.shape[0]:
        used_precomputed = True
        fit_equation = lambda i, data: equation.fit_with_params(prefitted_params[i])
    else:
        used_precomputed = False
        fit_equation = lambda i, data: equation.fit(data[0], data[1])
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train_s = preprocess(X_train)
        X_test_s = preprocess(X_test, X_train)
        # Optimize equation and fit parameters (Cs)
        fit_equation(i, (X_train_s, y_train))
        y_pred = equation.predict(X_test_s)
        all_predictions.append(y_pred.item())
        all_targets.append(y_test.item())
        all_params.append(equation.fitted_params)
    return np.array(all_targets), np.array(all_predictions), all_params, used_precomputed

def replace_numeric_parameters(equation):
    # Pattern to match numbers that are NOT simple (1 decimal place or less)
    pattern = r'(?<![a-zA-Z_.])-?(?:\d+\.\d{2,}|\d*\.\d+[eE][+-]?\d+|\d+[eE][+-]?\d+)(?![a-zA-Z_.])'
    return re.sub(pattern, 'C', equation)

def replace_optimized_parameters(equation_str, optimized_params, precision=6):
    """
    Replace placeholder 'C's in an equation string with optimized numerical parameters.
    
    Args:
        equation_str (str): Original equation with 'C' placeholders (e.g., "C*D + C")
        optimized_params (list): Learned parameters from optimization (e.g., [1.5, -0.3])
        precision (int): Number of significant digits to display
    
    Returns:
        str: Equation with numerical values instead of placeholders
    """
    param_idx = 0
    parts = []
    last_end = 0
    
    # Match standalone 'C' (not part of words like CT or C2)
    pattern = re.compile(r'\bC\b')
    
    for match in pattern.finditer(equation_str):
        start, end = match.start(), match.end()
        parts.append(equation_str[last_end:start])
        
        if param_idx < len(optimized_params):
            # Format parameter with specified precision
            param = optimized_params[param_idx]
            parts.append(f"{param:.{precision}g}")
            param_idx += 1
        else:
            # If more placeholders than parameters, leave as 'C'
            parts.append("C")
            
        last_end = end
    
    parts.append(equation_str[last_end:])
    
    if param_idx != len(optimized_params):
        raise ValueError(f"Found {param_idx} placeholders but got {len(optimized_params)} parameters")
    
    return "".join(parts)

# Plotting

def plot_predictions(df, idxs):
    df = df.sort_values(by='S').reset_index(drop=True)

    # Plot settings
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")

    # Plot the real target values
    sns.lineplot(x=range(len(df)), y=df['S'], label='Actual S', marker='o', color='black')

    # Plot the predictions
    for i in idxs:
        sns.lineplot(x=range(len(df)), y=df[f'Prediction_{i}'], label=f'Prediction_{i}', marker='o')

    plt.xlabel('Sample Index')
    plt.ylabel('Values')
    plt.title('Comparison of Predictions vs Actual S')
    plt.legend()
    plt.show()


def plot_x_vs_y(dfs, x: str, y: str, scatter=False):
    if not isinstance(dfs[0], list):
        dfs = [dfs]

    plt.figure(figsize=(10, 6))

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']  # Extend if needed
    markers =['o', '*', '+', 'x']

    for i, (name, df) in enumerate(dfs):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        if scatter:
            plt.scatter(df[x], df[y], marker=marker, color=color, label=name)
        else: 
            plt.plot(df[x], df[y], marker=marker, linestyle='-', color=color, label=name)

    plt.title(f'{x} vs. {y}')
    plt.xlabel(f'{x}')
    plt.ylabel(f'{y}')
    plt.grid(True)
    plt.legend()

def compute_pareto_frontier(df, x, y, minimize_x=True, minimize_y=True):
    # Sort by x based on minimization or maximization
    df_sorted = df.sort_values(by=x, ascending=minimize_x)
    pareto = []
    
    for _, row in df_sorted.iterrows():
        if not pareto:
            pareto.append(row)
        else:
            last = pareto[-1]
            if row[x] == last[x]:
                # If same x, update the last row if current row has a better y.
                if (minimize_y and row[y] < last[y]) or (not minimize_y and row[y] > last[y]):
                    pareto[-1] = row
            else:
                # New x value: Only add if the current row's y is better than the last Pareto's y.
                if (minimize_y and row[y] < last[y]) or (not minimize_y and row[y] > last[y]):
                    pareto.append(row)
    return df.__class__(pareto)  # preserves the original type (DataFrame)

import numpy as np
import pandas as pd

def compute_elbow_metric(df, complexity_col='complexity', tt_col='tt', maximize_tt=True):
    """
    Compute an elbow metric for each point in the Pareto frontier by comparing the slopes on either side.
    
    For each interior point, compute:
        slope_left  = (tt_i - tt_{i-1}) / (complexity_i - complexity_{i-1})
        slope_right = (tt_{i+1} - tt_i) / (complexity_{i+1} - complexity_i)
        elbow_metric = slope_right - slope_left
        
    For the last point, we assume a virtual slope_right of 0 (as if there's an extra point with the same tt
    but a slightly larger complexity).
    
    If tt is to be maximized then a more negative elbow metric is considered better 
    (since it means increasing complexity doesn't yield much additional tt).
    Conversely, if tt is to be minimized, then a more positive elbow metric is better.
    
    The first point’s elbow remains undefined (NaN).
    
    Parameters:
        df (pd.DataFrame): DataFrame containing Pareto frontier points.
        complexity_col (str): Column name for complexity.
        tt_col (str): Column name for the performance metric tt.
        maximize_tt (bool): Whether higher tt values are better.
        
    Returns:
        pd.DataFrame: A copy of `df` with additional columns 'slope_left', 'slope_right', and 'elbow_metric'.
    """
    df_sorted = df.sort_values(by=complexity_col, ascending=True).copy()
    n = len(df_sorted)
    
    # Initialize new columns
    df_sorted['slope_left'] = np.nan
    df_sorted['slope_right'] = np.nan
    df_sorted['elbow_metric'] = np.nan
    
    # Compute slopes for interior points
    for i in range(1, n - 1):
        x_prev = df_sorted.iloc[i - 1][complexity_col]
        x_curr = df_sorted.iloc[i][complexity_col]
        x_next = df_sorted.iloc[i + 1][complexity_col]
        y_prev = df_sorted.iloc[i - 1][tt_col]
        y_curr = df_sorted.iloc[i][tt_col]
        y_next = df_sorted.iloc[i + 1][tt_col]
        
        slope_left = (y_curr - y_prev) / (x_curr - x_prev) if (x_curr - x_prev) != 0 else np.nan
        slope_right = (y_next - y_curr) / (x_next - x_curr) if (x_next - x_curr) != 0 else np.nan
        elbow = slope_right - slope_left
        
        df_sorted.at[df_sorted.index[i], 'slope_left'] = slope_left
        df_sorted.at[df_sorted.index[i], 'slope_right'] = slope_right
        df_sorted.at[df_sorted.index[i], 'elbow_metric'] = elbow

    # For the last point, assume slope_right = 0
    if n > 1:
        x_prev = df_sorted.iloc[-2][complexity_col]
        x_curr = df_sorted.iloc[-1][complexity_col]
        y_prev = df_sorted.iloc[-2][tt_col]
        y_curr = df_sorted.iloc[-1][tt_col]
        slope_left = (y_curr - y_prev) / (x_curr - x_prev) if (x_curr - x_prev) != 0 else np.nan
        slope_right = 0  # Virtual slope for the last point
        elbow = slope_right - slope_left
        
        df_sorted.at[df_sorted.index[-1], 'slope_left'] = slope_left
        df_sorted.at[df_sorted.index[-1], 'slope_right'] = slope_right
        df_sorted.at[df_sorted.index[-1], 'elbow_metric'] = elbow

    return df_sorted


def plot_pareto_frontier(dfs, x: str, y: str, scatter=True, minimize_x=True, minimize_y=True):
    if not isinstance(dfs, list):
        dfs = [('Regressor', dfs)]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    markers = ['o', '+', 'x', '^', 's', 'D', 'p', 'h', '<', '>']
    last_x = max([df[x].max() for _, df in dfs]) if minimize_x else min([df[x].min() for _, df in dfs])

    for i, (name, df) in enumerate(dfs):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        if scatter:
            # Plot individual points
            plt.scatter(df[x], df[y], marker=marker, color=color, label=name)
            
            # Calculate and plot Pareto frontier
            pareto_df = compute_pareto_frontier(df, x, y, minimize_x, minimize_y)[[x, y]]
            last_y = pareto_df[y].min() if minimize_y else pareto_df[y].max()
            if (pareto_df[x].iloc[-1] != last_x) or (pareto_df[y].iloc[-1] != last_y):
                pareto_df = pd.concat([pareto_df, pd.DataFrame({x: [last_x], y: [last_y]})], ignore_index=True)
            plt.plot(pareto_df[x], pareto_df[y], color=color, linestyle='--', linewidth=1.5)
        else:
            plt.plot(df[x], df[y], marker=marker, linestyle='-', color=color, label=name)

    plt.title(f'{x} vs. {y}')
    plt.xlabel(f'{x}')
    plt.ylabel(f'{y}')
    plt.grid(True)
    plt.legend()

import pandas as pd

def df_to_latex(
    df,
    filename=None,
    index=False,
    escape=True,
    float_format="%.4f",
    bold_max=False,
    bold_min=False,
    caption="",
    label="tab:",
    position="h!"
):
    """
    Convert a DataFrame to LaTeX with optional styling.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        filename (str): If provided, saves LaTeX to a file. Default=None.
        index (bool): Whether to include the index. Default=False.
        escape (bool): Escape special LaTeX characters. Default=True.
        float_format (str): Format for floating-point numbers (e.g., "%.2f"). Default="%.4f".
        bold_max (bool): Bold the maximum value in each column. Default=False.
        bold_min (bool): Bold the minimum value in each column. Default=False.
        caption (str): Table caption. Default="".
        label (str): LaTeX label (for referencing). Default="tab:".
        position (str): LaTeX table placement (e.g., "h!"). Default="h!".
    
    Returns:
        str: LaTeX table code.
    """
    # Style the DataFrame (bold max/min if requested)
    styled_df = df.copy()
    if bold_max or bold_min:
        def highlight_extrema(x):
            x = x.astype(float)
            styles = pd.Series("", index=x.index)
            if bold_max:
                styles[x.idxmax()] = "\\textbf{%.4f}" % x.max()
            if bold_min:
                styles[x.idxmin()] = "\\textbf{%.4f}" % x.min()
            return styles
        styled_df = styled_df.style.apply(highlight_extrema)
    
    # Convert to LaTeX
    latex_code = styled_df.to_latex(
        index=index,
        escape=escape,
        float_format=float_format,
        caption=caption,
        label=label,
        position=position
    )
    
    # Save to file if requested
    if filename:
        with open(filename, "w") as f:
            f.write(latex_code)
    
    return latex_code