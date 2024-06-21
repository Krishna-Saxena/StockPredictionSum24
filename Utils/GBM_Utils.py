import numpy as np


def sample_GBM(mu, sigma_sq, S_0, dt, add_BM=True):
    avg_drift = mu - sigma_sq / 2

    S_ts = np.concatenate(([S_0], np.zeros_like(dt)))
    for i in range(1, 1 + len(dt)):
        drift = avg_drift * dt[i - 1]
        if add_BM:
            drift += np.random.normal(loc=0, scale=sigma_sq * dt[i - 1] ** 0.5)

        S_ts[i] = S_ts[i - 1] * np.exp(drift)

    return S_ts


def get_MLE_params(log_prices, delta_ts):
    N = log_prices.shape[0]
    delta_X = log_prices[-1] - log_prices[0]
    delta_t = np.nansum(delta_ts)

    sigma_sq_hat = -1 / N * delta_X ** 2 / delta_t + \
                   np.nanmean((log_prices[1:] - log_prices[:-1]) ** 2 / delta_ts[1:])
    mu_hat = delta_X / delta_t + 0.5 * sigma_sq_hat

    return mu_hat, sigma_sq_hat
