# Losses
from scipy.special import gammaln
import numpy as np
from numba import njit

def get_loss(name: str):
    losses_dict = {'MAE': MAE, 'MSE': MSE, 'RMSE': RMSE, 'R2': R2, 'RobustTanh': robust_loss,
                   'BinaryCrossEntropy': binary_cross_entropy, 'ExponentialLoss': exp_loss, 
                   'PoissonDeviance': poisson_deviance}
    if name not in losses_dict: 
        raise KeyError(f"'{name}' is not a valid loss name. Valid names are: {list(losses_dict.keys())}")
    loss_fn = losses_dict[name]
    cal = np.random.rand(1000)
    t = np.random.rand(1000)
    loss_fn(cal, t)
    return loss_fn


@njit
def MAE(cal, t):
    total = 0.0
    for i in range(len(cal)):
        total += abs(cal[i] - t[i])
    return total / len(cal)

@njit
def MSE(cal, t):
    total = 0.0
    for i in range(len(cal)):
        diff = cal[i] - t[i]
        total += diff * diff
    return total / len(cal)

@njit
def RMSE(cal, t):
    total = 0.0
    for i in range(len(cal)):
        diff = cal[i] - t[i]
        total += diff * diff
    return np.sqrt(total / len(cal))

@njit
def R2(cal, t):
    mean_t = 0.0
    for i in range(len(t)):
        mean_t += t[i]
    mean_t /= len(t)

    ss_res = 0.0
    ss_tot = 0.0
    for i in range(len(t)):
        diff_res = t[i] - cal[i]
        diff_tot = t[i] - mean_t
        ss_res += diff_res * diff_res
        ss_tot += diff_tot * diff_tot

    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

@njit
def robust_loss(cal, t):
    total = 0.0
    n = len(cal)
    for i in range(n):
        diff = cal[i] - t[i]
        total += diff * np.tanh(diff / 2.0)
    return total / n

@njit
def binary_cross_entropy(cal, t):
    total = 0.0
    n = len(cal)
    for i in range(n):
        x = cal[i]
        x = max(min(x, 50.0), -50.0)         # clip for stability
        prob = 1.0 / (1.0 + np.exp(-x))    # sigmoid
        prob = min(max(prob, 1e-15), 1 - 1e-15)  # clip to prevent log(0)
        total += -t[i] * np.log(prob) - (1 - t[i]) * np.log(1 - prob)
    return total / n

@njit
def exp_loss(cal, t):
    total = 0.0
    n = len(cal)
    for i in range(n):
        total += np.exp(-cal[i] * t[i])
    return total / n

def neglog_likelihood(cal, t):
    return float(np.mean(np.exp(cal) - t * cal + gammaln(t+1)))

def neglog_likelihood_nolink(cal, t):
    # calculate neglog_likelihood of Poisson without link
    mu = np.maximum(1e-99, cal) 
    ans = float(np.mean(mu - t * np.log(mu) + gammaln(t+1)))
    return ans

@njit
def poisson_deviance(cal, t):
    eps = 1e-8
    total = 0.0
    n = len(cal)
    for i in range(n):
        c = cal[i]
        ti = t[i]
        if c < eps:
            c = eps
        total += ti * np.log((ti + eps) / c) - (ti - c)
    return total / n

def poisson_deviance2(cal, t):
    eps = 1e-8
    # cal = np.asarray(cal).astype(float)
    cal = np.where(cal < eps, eps, cal)
    return np.mean(t * np.log((t + eps) / cal) - (t - cal))

def poisson_deviance3(cal, t):
    eps = 1e-10
    # cal = np.asarray(cal).astype(float)
    prediction_nonpositive_mask = cal < eps
    prediction_positive_mask = ~prediction_nonpositive_mask

    t_np = t[prediction_nonpositive_mask]
    pred_np = cal[prediction_nonpositive_mask]

    t_p = t[prediction_positive_mask]
    pred_p = cal[prediction_positive_mask]

    # Continuous Piecewise Loss function, to inform gradient even if outside of the domain
    loss_np = np.sum(t_np * np.log((t_np + eps) / eps) - (t_np + pred_np - 2 * eps))
    loss_p = np.sum(t_p * np.log((t_p + eps) / pred_p) - (t_p - pred_p))
    total_loss = (loss_np + loss_p)/t.shape[0]

    return total_loss