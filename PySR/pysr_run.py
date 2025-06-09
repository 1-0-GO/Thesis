from pysr import PySRRegressor
import sympy
import pandas as pd

df = pd.read_csv("poisson_train.csv")
X = df[['x']]
y = df['y']

model = PySRRegressor(
    procs=8,
    populations=24,
    niterations=10000000,  # Run forever
    timeout_in_seconds=60 * 60 * 0.1,    # Stop after 12 hours have passed.
    maxsize=25, # max complexity
    #equation_file="pysr_eqs.csv",
    binary_operators=[
        "+", 
        "-", 
        "*", 
        "^", 
        "cond", 
        "root(x, p) = (p>=1 && x>0) ? x^(1/p) : convert(typeof(x), NaN)",
        "harmonic(x, y) = 2*x*y/(x+y)", 
        "saturation(x, K) = K * x / (K + x)",
        "damping(x, β) = x / (1 + β * x)"
        "decay(x, λ) = λ>0 ? exp(-λ * x) : convert(typeof(x), NaN)",
        "threshold(x, τ) = max(0, x - τ)"
    ],
    unary_operators=[
        "log1p", 
        "square", 
        "inverse(x) = 1/(x+1e-6)", 
        "logistic(x) = 1 / (1 + exp(-x))"
    ],
    extra_sympy_mappings={
        "inverse(x)": lambda x: 1/(x+1e-6),
        "root(x, p)": lambda x,p: (p>1 and x>0) and x**(1/p) or sympy.nan,
        "harmonic(x, y)": lambda x,y: 2*x*y/(x+y), 
        "saturation(x, K)": lambda x, K: K * x / (K + x),
        "damping(x, β)": lambda x, β: x / (1 + β * x),
        "decay(x, λ)": lambda x, λ: λ>0 and sympy.exp(-λ * x) or sympy.nan,
        "threshold(x, τ)": lambda x, τ: max(0, x - τ)
    },
    complexity_of_operators={
        "+": 1,
        "-": 1.5,
        "*": 2,
        "^": 4, 
        "cond": 4,
        "log1p": 2,
        "square": 2,
        "inverse": 3,
        "logistic": 4,
        "root": 3,
        "harmonic": 4,
        "saturation": 3.5,
        "damping": 3,
        "decay": 3.5,
        "threshold": 3,
    },
    complexity_of_constants=2.5,
    constraints={
        "^": (3, 2.5),
        "cond": (6,-1),
        "square": 9,
        "inverse": 9,
        "root": (9, 2.5),
        "harmonic": (9, 9),
        "saturation": (9, 2.5),
        "damping": (9, 2.5),
        "decay": (9, 2.5),
        "threshold": (9, 2.5),
    },
    weight_randomize=0.001,
    warmup_maxsize_by=0.4,
    adaptive_parsimony_scaling=500,
    weight_optimize=0.001,
    elementwise_loss="loss(prediction, target) = prediction - target*log(prediction) + loggamma(target + 1)",  # Custom loss function (julia syntax)
    progress=True,
    bumper=True
)

model.fit(X, y)
print(model)