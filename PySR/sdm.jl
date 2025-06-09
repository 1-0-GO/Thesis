import SymbolicRegression: SRRegressor
import MLJ: machine, fit!, predict, report
import LoopVectorization
import SpecialFunctions: loggamma
import CSV
import DataFrames: DataFrame
import Statistics: mean, std
import Bumper

function standardize!(X)
    for col in 1:size(X, 2)
        mean_val = mean(X[:, col])
        std_val = std(X[:, col])
        X[:, col] .= (X[:, col] .- mean_val) ./ std_val
    end
end

path_train = "../RSRM/data/SR/alestrus_train.csv"
path_test = "../RSRM/data/SR/alestrus_test.csv"

df_train = CSV.read(path_train, DataFrame)
df_test = CSV.read(path_test, DataFrame)

# Merge the datasets
df = vcat(df_train, df_test)

nrows = size(df, 1)
split_idx = Int(floor(0.8 * nrows))
X = df[1:split_idx, 1:end-1] |> Matrix 
y = df[1:split_idx, end] |> Vector     
# y .= ifelse.(y .== 0, -1, y)

# Standardize features
standardize!(X)

# Define custom operators in Julia
harmonic(x, y) = 2*x*y/(x+y)
damping(x, k) = x / (1 + k * x)
threshold(x, k) = max(0, x - k)
inverse(x) = 1/(x + 1e-6)
logistic(x) = 1 / (1 + exp(-x))
square(x) = x*x

# Custom loss
function sigmoid_loss(y_hat, y)
    phat = 1/(1 + exp(-y_hat))
    return -( y * log(phat)  + (1-y) * log(1 - phat) )
end


model = SRRegressor(
    niterations=100000,
    maxsize=30, # max complexity
    binary_operators=[+, -, *, ^, harmonic, damping, threshold],
    unary_operators=[log1p, square, sqrt, inverse, logistic],
    complexity_of_operators=[(+) => 1, (-) => 1, (*) => 2, (^) => 6, log1p => 2,    #\\
    square => 3, sqrt => 3, exp => 2, inverse => 3, logistic => 5, harmonic => 4,   #\\
    damping => 4, threshold => 4],    
    complexity_of_constants=2,
    elementwise_loss=sigmoid_loss,
    constraints=[(^) => (4, 3), square => 7, inverse => 7, sqrt => 7, exp => 7,     #\\ 
    logistic => 7, harmonic => (7, 7), damping => (7, 2), threshold => (7, 2)],
    nested_constraints = [
        (+) => Dict(),
        (-) => Dict(),
        (*) => Dict(),
        (^) => [square => 0, logistic => 0, inverse => 0],
        log1p => [harmonic => 0, saturation => 0, damping => 0, threshold => 0],
        square => [square => 0, inverse => 0, logistic => 0, saturation => 0],
        sqrt => [square => 0, harmonic => 0, saturation => 0],
        inverse => [inverse => 0, harmonic => 0, saturation => 0, threshold => 0],
        logistic => [logistic => 0, threshold => 0, harmonic => 0],
        harmonic => [harmonic => 0, saturation => 0, damping => 0, threshold => 0],
        saturation => [harmonic => 0, saturation => 0, damping => 0, threshold => 0],
        damping => [harmonic => 0, logistic => 0, inverse => 0, threshold => 0],
        threshold => [harmonic => 0, logistic => 0, inverse => 0, saturation => 0]
    ],
    warmup_maxsize_by=0.1,
    adaptive_parsimony_scaling=1000,
    bumper=true,
    turbo=true
)


mach = machine(model, X, y, scitype_check_level=0)

fit!(mach)

report(mach)

println(predict(mach, X))