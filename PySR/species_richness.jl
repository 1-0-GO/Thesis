import SymbolicRegression: SRRegressor
import MLJ: machine, fit!, predict, report
import LoopVectorization
import SpecialFunctions: loggamma
import CSV
import DataFrames: DataFrame
import Statistics: mean
import Bumper

path_train = "../RSRM/data/SR/s_train.csv"
path_test = "../RSRM/data/SR/s_test.csv"

df_train = CSV.read(path_train, DataFrame)
df_test = CSV.read(path_test, DataFrame)

# Merge the datasets
df = vcat(df_train, df_test)

# Extract `X` and `y`
y = df[:, end]
X = df[:, 1:end-1]


# Define custom operators in Julia
harmonic(x, y) = 2 * x * y / (x+y)
damping(x, k) = x / (1 + k * x)
threshold(x, k) = max(0, x - k)
inverse(x) = 1/(x + 1e-6)
logistic(x) = 1 / (1 + exp(-x))
square(x) = x*x


# Define custom poisson loss likelihood 
loss(y,t) = max(1e-99, y) - t * log(max(1e-99, y)) + loggamma(t + 1) 


model = SRRegressor(
    niterations=100000,
    #timeout_in_seconds=60 * 60 * 12, # 12 hours
    maxsize=30, # max complexity
    binary_operators=[+, -, *, ^, harmonic, damping, threshold],
    unary_operators=[log1p, square, sqrt, inverse, logistic],
    complexity_of_operators=[(+) => 1, (-) => 1, (*) => 2, (^) => 6, log1p => 2,    #\\
    square => 3, sqrt => 3, exp => 2, inverse => 3, logistic => 5, harmonic => 4,   #\\
    damping => 4, threshold => 4],    
    complexity_of_constants=2,
    elementwise_loss=loss,
    constraints=[(^) => (4, 3), square => 7, inverse => 9, sqrt => 9, exp => 9,     #\\ 
    logistic => 9, harmonic => (9, 9), damping => (9, 2), threshold => (9, 2)],
    warmup_maxsize_by=0.2,
    adaptive_parsimony_scaling=2000,
    parsimony=0.2,
    bumper=true,
    turbo=true
)


mach = machine(model, X, y, scitype_check_level=0)

fit!(mach)

report(mach)

println(predict(mach, X))