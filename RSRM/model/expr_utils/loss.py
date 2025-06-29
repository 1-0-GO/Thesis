# Losses
from scipy.special import gammaln
import numpy as np

def get_loss(name: str):
    losses_dict = {'binary': binary_cross_entropy, 'exp': exp_loss, 'RMSE': RMSE, 'robust': robust_loss,
                   'poisson_like_explink': neglog_likelihood, 'Poisson_like_nolink': neglog_likelihood_nolink,
                   'Poisson': poisson_deviance}
    if name not in losses_dict: 
        raise KeyError(f"'{name}' is not a valid loss name. Valid names are: {list(losses_dict.keys())}")
    return losses_dict[name]

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(cal, t):
    cal = sigmoid(cal)
    # Clip predictions to prevent log(0)
    cal = np.clip(cal, 1e-15, 1 - 1e-15)
    # Calculate binary cross-entropy loss
    return float(-np.sum(t * np.log(cal) + (1 - t) * np.log(1 - cal)))

def exp_loss(cal, t):
    return float(np.sum(np.exp(-cal * t)))

def RMSE(cal, t):
    return float(np.linalg.norm(cal - t, 1) ** 2)

def neglog_likelihood(cal, t):
    return float(np.sum(np.exp(cal) - t * cal + gammaln(t+1)))

def neglog_likelihood_nolink(cal, t):
    # calculate neglog_likelihood of Poisson without link
    mu = np.maximum(1e-99, cal) 
    ans = float(np.sum(mu - t * np.log(mu) + gammaln(t+1)))
    return ans

def robust_loss(cal, t):
    elem_loss = lambda y_pred, y_true: (y_pred - y_true)*np.tanh((y_pred-y_true)/2)
    return float(np.sum(elem_loss(cal, t)))

def poisson_deviance(cal, t):
    eps = 1e-10
    cal = np.asarray(cal).astype(float)
    prediction_nonpositive_mask = cal < eps
    prediction_positive_mask = ~prediction_nonpositive_mask

    t_np = t[prediction_nonpositive_mask]
    pred_np = cal[prediction_nonpositive_mask]

    t_p = t[prediction_positive_mask]
    pred_p = cal[prediction_positive_mask]

    # Continuous Piecewise Loss function, to inform gradient even if outside of the domain
    loss_np = np.sum(t_np * np.log((t_np + eps) / eps) - (t_np + pred_np - 2 * eps))
    loss_p = np.sum(t_p * np.log((t_p + eps) / pred_p) - (t_p - pred_p))
    total_loss = (loss_np + loss_p)

    return total_loss

def poisson_deviance2(cal, t):
    eps = 1e-10
    cal = np.asarray(cal).astype(float)
    cal = np.where(cal < eps, 0, cal)
    return np.sum(t * np.log((t + eps) / cal) - (t - cal))