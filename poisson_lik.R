n_obs <- 101
x <- seq(0,1,length.out=n_obs)
eq <- function(x, b) x^2+b
mu <- exp(eq(x,1))
y <- rpois(n_obs, mu)

neglog_lik <- function(b) sum(exp(eq(x,b)) - y*eq(x,b) + lgamma(y+1))
neglog_lik_vec <- Vectorize(neglog_lik)
b <- seq(0,2,length.out=101)
b[which.min(neglog_lik_vec(b))]

plot(b, neglog_lik_vec(b), type='l')

#### f(x) = x - sqrt(x) - 5 ####

eq <- function(x, b) x-sqrt(x)+b
#Generate or read
read = FALSE
write = TRUE  # if read is TRUE decides if we write the dataset to file
if(!read) {
  #Generate
  n_obs <- 20
  x <- rnorm(n_obs, mean=10, sd=2)
  mu <- exp(eq(x,-5))
  y <- rpois(n_obs, mu)
  df <- data.frame(x=x, y=y)
  if(write) {
    write.csv(df[1:10, ], "C:\\Users\\marti\\OneDrive\\Documentos\\Universidade\\2024_2025\\Symbolic Regression\\poisson_train.csv", row.names = FALSE)
    write.csv(df[11:20, ], "C:\\Users\\marti\\OneDrive\\Documentos\\Universidade\\2024_2025\\Symbolic Regression\\poisson_test.csv", row.names = FALSE)
  }
}else{
  #Read
  df_train <- read.csv("C:\\Users\\marti\\OneDrive\\Documentos\\Universidade\\2024_2025\\Symbolic Regression\\RSRM\\data\\Experiments\\poisson1_train.csv")
  df_test <- read.csv("C:\\Users\\marti\\OneDrive\\Documentos\\Universidade\\2024_2025\\Symbolic Regression\\RSRM\\data\\Experiments\\poisson1_test.csv")
  df <- rbind(df_train, df_test)
  n_obs <- nrow(df)
  x <- df$x
  y <- df$y
}

loss_eq <- function(eq) sum(exp(eq(x)) - y*eq(x) + lgamma(y+1))
neglog_lik(-5)
#RSRM found a function
eq1 <- function(x) log(123.41/x)
loss_eq(eq1)
# After only 0.589 seconds! Got lucky here
eq2 <- function(x) x - 108.457/x
loss_eq(eq2)
eq3 <- function(x) x*(0.0124476*x - log(log(log(x))))
loss_eq(eq3)
eq4 <- function(x) 2.63189
loss_eq(eq4)
# Fantastic approximation of the function
eq5 <- function(x) x - 8.564
loss_eq(eq5)
# What?! Better than the real one hahah
eq6 <- function(x) x - 2*log(x) - 3.5446 
loss_eq(eq6)

#### f(x) = x*log(x) - 2/x - 3 ####

n_obs <- 100
eq <- function(x) 0.5*x*log(x) + 3/x - 3
#Generate or read
read = TRUE
if(!read) {
  #Generate
  n_obs <- 20
  x <- rnorm(n_obs, mean=10, sd=2)
  mu <- exp(eq(x,-5))
  y <- rpois(n_obs, mu)
  df <- data.frame(x=x, y=y)
  write.csv(df[1:10, ], "C:\\Users\\marti\\OneDrive\\Documentos\\Universidade\\2024_2025\\Symbolic Regression\\poisson_train.csv", row.names = FALSE)
  write.csv(df[11:20, ], "C:\\Users\\marti\\OneDrive\\Documentos\\Universidade\\2024_2025\\Symbolic Regression\\poisson_test.csv", row.names = FALSE)
}else{
  #Read
  df_train <- read.csv("C:\\Users\\marti\\OneDrive\\Documentos\\Universidade\\2024_2025\\Symbolic Regression\\RSRM\\data\\Experiments\\poisson_train.csv")
  df_test <- read.csv("C:\\Users\\marti\\OneDrive\\Documentos\\Universidade\\2024_2025\\Symbolic Regression\\RSRM\\data\\Experiments\\poisson_test.csv")
  df <- rbind(df_train, df_test)
  n_obs <- nrow(df)
  x <- df$x
  y <- df$y
}

loss_eq(eq)
eq_1 <- function(x) 0.8229*x*log(x) - x
loss_eq(eq_1)

#### Data ####
data_SAR <- read.csv("Original Data/data_SAR.csv", sep=',')
loss_eq <- function(y_pred, y_target) sum(y_pred - y_target*log(y_pred) + lgamma(y_target+1))
foo <- function(T) (1162.52 - T)*log(T)/T
fool <- function(X1,X2)  (X2 + log(X1*(X1*X2 - X1)))*(-X2 + log(X1) + 15.458160451311374)
foo(data_SAR$T)
fool(data_SAR$A, data_SAR$T)
data_SAR$Scolopendrites
loss_eq(foo(data_SAR$T), data_SAR$Scolopendrites)
loss_eq(fool(data_SAR$A, data_SAR$T), data_SAR$Scolopendrites)

# Load necessary libraries
library(ggplot2)
library(dplyr)

# Define the functions
harmonic <- function(x, y) 2 * x * y / (x + y)
saturation <- function(x, k) k * x / (k + x)
damping <- function(x, k) x / (1 + k * x)
threshold <- function(x, k) pmax(0, x - k)
log1p <- function(x) log(1 + x)
logistic <- function(x) 1 / (1 + exp(-x))

# Generate data for plotting
x_vals <- seq(0, 10, length.out = 100)
y_vals <- seq(0, 10, length.out = 100)

data <- data.frame(
  x = rep(x_vals, 6),
  func = rep(c("harmonic", "saturation", "damping", "threshold", "log1p", "logistic"), each = length(x_vals)),
  y = c(harmonic(x_vals, 2), saturation(x_vals, 2), damping(x_vals, 2), threshold(x_vals, 2), log1p(x_vals), logistic(x_vals))
)

# Create the plot
p <- ggplot(data, aes(x = x, y = y)) +
  geom_line() +
  facet_wrap(~func, scales = "free_y", ncol = 3) +
  labs(title = "Plot of Various Functions", x = "x", y = "y") +
  theme_minimal()

# Display the plot
print(p)








