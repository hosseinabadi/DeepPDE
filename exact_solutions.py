import numpy as np


def decompose_covariance_matrix(t, volatility1, volatility2, correlation):
    """ Decompose covariance matrix as in Lemma 3.1 of Bayer et. al (2018). """
    sigma_det = (1 - correlation ** 2) * volatility1 ** 2 * volatility2 ** 2
    sigma_sum = (volatility1 ** 2 + volatility2 ** 2
                 - 2 * correlation * volatility1 * volatility2)

    ev1 = volatility1 ** 2 - correlation * volatility1 * volatility2
    ev2 = -(volatility2 ** 2 - correlation * volatility1 * volatility2)
    ev_norm = np.sqrt(ev1 ** 2 + ev2 ** 2)

    eigenvalue = volatility1 ** 2 + volatility2 ** 2 - 2 * sigma_det / sigma_sum

    v_mat = np.array([ev1, ev2]) / ev_norm
    d = t * np.array([sigma_det / sigma_sum, eigenvalue])
    return d, v_mat


def one_dimensional_exact_solution(
        t, s, riskfree_rate, volatility, strike_price):
    """ Standard Black-Scholes formula """

    d1 = (1 / (volatility * np.sqrt(t))) * (
            np.log(s / strike_price)
            + (riskfree_rate + volatility ** 2 / 2.) * t
    )
    d2 = d1 - volatility * np.sqrt(t)
    return (norm.cdf(d1) * s
            - norm.cdf(d2) * strike_price * np.exp(-riskfree_rate * t))


def exact_solution(
        t, s1, s2, riskfree_rate, volatility1, volatility2, correlation):
    """ Compute the option price of a European basket call option. """
    if t == 0:
        return np.maximum(0.5 * (s1 + s2) - strike_price, 0)

    d, v = decompose_covariance_matrix(
        t, volatility1, volatility2, correlation)

    beta = [0.5 * s1 * np.exp(-0.5 * t * volatility1 ** 2),
            0.5 * s2 * np.exp(-0.5 * t * volatility2 ** 2)]
    integration_points, integration_weights = hermgauss(33)

    # Transform points and weights
    integration_points = np.sqrt(2 * d[1]) * integration_points.reshape(-1, 1)
    integration_weights = integration_weights.reshape(1, -1) / np.sqrt(np.pi)

    h_z = (beta[0] * np.exp(v[0] * integration_points)
           + beta[1] * np.exp(v[1] * integration_points))

    evaluation_at_integration_points = one_dimensional_exact_solution(
        t=1, s=h_z * np.exp(0.5 * d[0]),
        strike_price=np.exp(-riskfree_rate * t) * strike_price,
        volatility=np.sqrt(d[0]), riskfree_rate=0.
    )

    solution = np.matmul(integration_weights, evaluation_at_integration_points)

    return solution[0, 0]



