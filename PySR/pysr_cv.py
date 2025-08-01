from pysr import PySRRegressor
import numpy as np
import sympy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.special import gammaln

def identify_header(path, n=5, th=0.9):
    df1 = pd.read_csv(path, header='infer', nrows=n)
    df2 = pd.read_csv(path, header=None, nrows=n)
    sim = (df1.dtypes.values == df2.dtypes.values).mean()
    return 'infer' if sim < th else None

def load_dataset(path):
    header = identify_header(f"data/{path}.csv")
    csv = pd.read_csv(f"data/{path}.csv", header=header)
    x, t = csv.iloc[:, :-1], csv.iloc[:, -1]
    return x, t

X1, y1 = load_dataset("s_train")
X2, y2 = load_dataset("s_test")
X = pd.concat([X1, X2], ignore_index=True) 
y = pd.concat([y1, y2], ignore_index=True)
X = X/(2 * X.std())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = PySRRegressor(
    procs=10,
    populations=30,
    niterations=10, 
    annealing=True,
    batching=True,
    batch_size=20,
    ncycles_per_iteration=2000,
    multithreading=False,
    weight_simplify=0.1,
    weight_optimize=0.1,
    optimize_probability=0.5,
    maxsize=40,
    warmup_maxsize_by=0.1,
    warm_start=False,
    binary_operators=[
        "+", 
        "-", 
        "*", 
        "^", 
    ],
    unary_operators=[
        "log", 
        "exp",
        "tanh",
        "inv(x) = 1/(x + 1f-10)",
        "relu(x) = max(0, x)",
    ],
    extra_sympy_mappings={
        "saturation": lambda x, y: x * y / (x + y),
        "inv": lambda x: 1 / x,
        "relu": lambda x: sympy.Max(0, x),
    },
    complexity_of_operators={
        "^": 2,
    },
    complexity_of_variables=0,
    constraints={
        "*": (9, 9),
        "^": (-1, 1),
        "log": 9,
        "exp": 9,
        "tanh": 9,
        "relu": 5,
        "inv": 5,
    },
    nested_constraints = {
        "^": {"^": 0, "inv": 0, "log": 0, "exp": 0, "tanh": 0, "relu": 0},
        "relu": {"inv": 0, "tanh": 0, "relu": 0},
        "log": {"^": 0, "inv": 0, "log": 0, "exp": 0, "tanh": 0, "relu": 0},
        "exp": {"^": 0, "log": 0, "exp": 0, "tanh": 0, "relu": 0},
        "tanh": {"^": 0, "log": 0, "exp": 0, "tanh": 0, "relu": 0},
        "inv": {"^": 0, "inv": 0, "exp": 0, "tanh": 0, "relu": 0},
    },   
    elementwise_loss="loss(prediction, target) = (prediction - target)^2 / max(prediction, 1e-6)",  
    bumper=True,
    turbo=True
)

# model.fit(X_train, y_train, weights=1/(np.abs(y_train)+1))
model.fit(X_train, y_train)

# Evaluation
poisson_log_likelihood = lambda y_true, y_pred: np.mean(y_pred - y_true * np.log(y_pred + 1e-10) + gammaln(y_true + 1))

def pseudo_R2(model, X_test, y_test):
    """Compute Pseudo R² for Poisson regression model."""
    res = []
    
    # Log likelihood of the model with only the intercept
    y_mean = np.mean(y_train)
    intercept_log_likelihood = poisson_log_likelihood(y_test, np.full_like(y_test, y_mean))
    
    for equation in model.equations_['lambda_format']:
        y_pred = equation(X_test)
        
        # Log likelihood of the fitted model
        fitted_log_likelihood = poisson_log_likelihood(y_test, y_pred)
        
        # Compute Pseudo R²
        val = 1 - (fitted_log_likelihood / intercept_log_likelihood)
        res.append(val)
    
    return res

def pearson_statistic(model, X_test, y_test):
    """Compute Pearson Statistic for Poisson regression model."""
    res = []

    for equation in model.equations_['lambda_format']:
        y_pred = equation(X_test)
        val = np.sum(np.square(y_test - y_pred)/y_pred)
        res.append(val)

    return res

def pseudo_aic(model, X_test, y_test):
    """Compute AIC for Poisson regression model."""
    res = []

    for i, equation in enumerate(model.equations_['lambda_format']):
        # Number of parameters (complexity) from the model
        k = model.equations_.iloc[i]['complexity']

        # Predict y values using the equation
        y_pred = equation(X_test)

        # Compute log-likelihood for the fitted model
        log_likelihood = poisson_log_likelihood(y_test, y_pred)

        # Compute AIC
        aic_value = 2 * k - 2 * log_likelihood
        res.append(aic_value)

    return res

def evaluation(model, X_test, y_test):
    metrics = [pseudo_R2, pearson_statistic]
    metric_names = ['Pseudo R²', 'Pearson Statistic']

    results = {name: [] for name in metric_names}
    for metric, name in zip(metrics, metric_names):
        values = metric(model, X_test, y_test)
        results[name] = values

    df_results = pd.DataFrame(results)
    df_results['Equation'] = [f"Equation {i+1}" for i in range(len(df_results))]
    return df_results

# print(evaluation(model, X_test, y_test))

# Cross validation
def cross_validation(model, X, y, n_splits=5):
    """Perform cross-validation and evaluate the model."""
    kf = KFold(n_splits=n_splits)
    all_results = []

    for run, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)

        df_results = evaluation(model, X_test, y_test)
        df_results['Run'] = run
        all_results.append(df_results)
    
    df_all_results = pd.concat(all_results, ignore_index=True)

    return df_all_results

df_cross_val_results = cross_validation(model, X, y)
print(df_cross_val_results)