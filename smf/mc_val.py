r"""
Provides functionalities necessary to perform whole option valuation procedure, using Monte Carlo methods.
The following model of the market is considered:

$$ \log(\frac{S^i_{t+1}}{S^i_t}) = \alpha_i N^i_t + \beta_i, \ i=1, 2, $$

where \(\alpha_i, \beta_i \in \mathbb{R}\) and \(N_t^i\) are standard normal random variables such that:

- \(N^i_t, N^i_s\) are independent for \( t \neq s\) and all \(i, j\),
- \(N^1_t, N^2_t\) are correlated with constant correlation coefficient \(\rho\).

Time parameter \(t\) is assumed to be discrete with values in set \(\{0, 1, ..., T\}\). The existence of
numeraire process (asset \(S^0\)) is also assumed.

Module supports valuation of options that depends on more than two assets, as far as Radon-Nikodym derivative is not
involved in valuation process, that is - market scenarios are generated in martingale measure. Simulations performed in
martingale measure are highly sensitive to number of replications for big values of coefficient c (i.e. variance in
martingale measure). Calculations in measure P are sensitive either for big and small values of c.

Functions
---------
- valuate --> Return option value. Perform whole valuation process.

- calculate_option_value --> Return option value based on given prices' scenarios.

- generate_scenarios --> Return scenarios of assets' prices.

- calculate_asset_price --> Return asset prices for given arrangement of random increments.

- calculate_rn_derivative --> Return Radon-Nikodym derivatives for respective scenarios based on their random increments.

"""

import numpy as np

from smf.exceptions import GeneratingMethodError


def valuate(option, s0, r, t_expr, alpha, beta, corr_mx, days_a_year=None, n=10 ** 5, method='cmc',
            c=np.ones(2), with_p=False, two_assets=False, rnd_seed=None):
    """
    Wrapper function. Performs whole option valuating procedure.

    Parameters
    ----------
    option : object extending Options Abstract Base Class
        Option to be valuated.
    s0 : 1-D ndarray of floats
        Array of initial assets' prices.
    r : float
        Risk-free rate.
    t_expr : int
        Number of days to expiration.
    alpha : 1-D ndarray of floats
        Array of standard deviation of logarithmic returns of assets.
    beta : 1-D ndarray of floats
        Array of means of logarithmic returns of assets.
    corr_mx : 2-D ndarray of floats
        Correlation matrix.
    days_a_year : None or int
        Number of trading days in a year. If not specified, parameters alpha beta and r must be given 'per day'.
        If specified parameters alpha beta and r must be given 'per year' and will be scaled further inside function.
        The default is None.
    n : int, optional
        Number of replications. The default is 10 ** 5.
    method : str, optional
        Monte Carlo estimator. The default is 'cmc'-Crude Monte Carlo.
    c : 1-D ndarray of floats, optional
        Variance of normal variables in martingale measure. The default is [1, 1].
    with_p : bool, optional
        Measure used for generating scenarios. If with_p == False, trajectories are generated in martingale measure.
        The default is False.
    two_assets : bool, optional
        If payoff of valuated option depends on two assets.
    rnd_seed : int
        Seed for random number generator.

    Returns
    -------
    float
        Value of the option.

    """
    if days_a_year is not None:
        r /= days_a_year
        beta /= days_a_year
        alpha /= np.sqrt(days_a_year)

    numeraire = np.exp(r * np.arange(0, t_expr + 1), dtype=np.float_)

    if not with_p:
        paths = generate_scenarios(s0, r, t_expr, alpha, beta, corr_mx, method=method, c=c, n=n, rnd_seed=rnd_seed)
        rn = np.ones(1)
    else:
        paths, inc = generate_scenarios(s0, r, t_expr, alpha, beta, corr_mx,
                                        method=method, c=c, n=n, rnd_seed=rnd_seed,
                                        with_p=True, return_inc=True)
        dim = 2 if two_assets else 1
        rn = calculate_rn_derivative(c, alpha, beta, corr_mx[0, 1], r, inc, dim)
        
    return calculate_option_value(option, paths, numeraire, rn)


def calculate_option_value(option, paths, numeraire, rn=np.ones(1)):
    """
    Calculates value of given option based on given scenarios.

    Parameters
    ----------
    option : object extending Options Abstract Base Class
        Option to be valuated.
    paths : 3-D ndarray of floats
        Arrangement of scenarios of asset prices based on which the option is valuated.
    numeraire : 1-D ndarray of floats
        Realization of numeraire process.
    rn : 1-D ndarray of floats, optional
        Array of Radon-Nikodym derivatives for respective scenarios. If not specified, no measure change is assumed and
        and payoffs from respective simulations are multiplied by 1 while calculating option value.

    Returns
    -------
    float
        Value of the option.

    """
    h = option.calculate_payoff(paths) / numeraire[-1]
    return np.mean(rn[:, None] * h, dtype=np.float_)


def generate_scenarios(s0, r, t_expr, alpha, beta, corr_mx, n=10 ** 5, method='cmc', with_p=False, c=np.ones(2),
                       return_inc=False, rnd_seed=None):
    """
    Generates trajectories of assets' prices.

    Parameters
    ----------
    s0 : 1-D ndarray of floats
        Array of initial assets' prices.
    r : float
        Risk-free rate. Must be given 'per day', that is divided by the number of trading days in a year.
    t_expr : int
        Number of days to expiration
    alpha : 1-D ndarray of floats
        Array of standard deviation of logarithmic returns of assets.
    beta : 1-D ndarray of floats
        Array of means of logarithmic returns of assets.
    corr_mx : 2-D ndarray of floats
        Correlation matrix.
    n : int, optional
        Number of replications. The default is 10 ** 5.
    method : str, optional
        Monte Carlo estimator. The default is 'cmc'-Crude Monte Carlo.
    with_p : bool, optional
        Measure used for generating scenarios. If with_p == False, trajectories are generated in martingale measure.
        The default is False.
    c : 1-D ndarray of floats, optional
        Variance of normal variables in martingale measure. The default is [1, 1].
    return_inc : bool, optional
        If True, random increments based on which scenarios are generated, are returned. The default is False.
    rnd_seed : int
        Seed for random number generator.
    Returns
    -------
    3-D ndarray of floats
        Arrangement of asset prices' trajectories.
    inc : 3-D ndarray of floats, optional
        Arrangement of random variables based on which trajectories are generated.

    """
    if with_p:
        mean, cov = np.zeros(2), corr_mx 
    else:
        mean, cov = _calculate_mtg_coefficients(c, alpha, beta, corr_mx, r)

    inc = _generate_mv_norm(n, t_expr, mean, cov, method, rnd_seed)
    
    if return_inc:
        return calculate_asset_price(s0, alpha, beta, inc), inc
    else:
        return calculate_asset_price(s0, alpha, beta, inc)


def _calculate_mtg_coefficients(c, alpha, beta, corr_mx, r):
    """
    Calculates coefficients used in generating paths in martingale measure.

    Parameters
    ----------
    c : 1-D ndarray of floats
        Array of standard deviations of standard normal variables in martingale measure.
    alpha : 1-D ndarray of floats
        Array of standard deviation of logarithmic returns of assets.
    beta : 1-D ndarray of floats
        Array of means of logarithmic returns of assets.
    corr_mx : 2-D ndarray of floats
        Correlation matrix
    r : float
        Risk-free rate. Must be given 'per day', that is divided by the number of trading days in a year.

    Returns
    -------
    mean : 1-D ndarray of floats
        Vector of means of 2-dimensional standard normal distribution under martingale measure.
    cov : 2-D ndarray of floats
        Covariance matrix of 2-dimensional standard normal distribution under martingale measure.
    """
    phi = (beta - r + .5 * (alpha * c) ** 2) / alpha
    rho = corr_mx[0, 1]
    mean = np.array([-phi[0],
                     -phi[1] - .5 * alpha[1] * rho ** 2 * (c[0] ** 2 - c[1] ** 2)],
                    dtype=np.float_)
    cov = np.array([[c[0] ** 2, c[0] ** 2 * rho],
                    [c[0] ** 2 * rho, (rho * c[0]) ** 2 + (1 - rho ** 2) * c[1] ** 2]],
                   dtype=np.float_)
    return mean, cov


def _generate_mv_norm(n, k, mean, cov, method, seed):
    """
    Generates multivariate normal sample. Returns arrangement of shape (n, k, len(mean)).

    Parameters
    ----------
    n : int
    k : int
    mean : 1-D ndarray of floats
        Vector of means.
    cov : 2-D ndarray of floats
        Covariance matrix.
    method : str
        Monte Carlo estimator.
    seed : int
        Seed for random number generator.

    Raises
    ------
    GeneratingMethodError
        If method is neither 'cmc' nor 'av'.

    Returns
    -------
    3-D ndarray of floats
        Arrangement of random variables.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    if method.lower() == 'cmc':
        inc = rng.multivariate_normal(mean,
                                      cov,
                                      size=(n, k),
                                      method='cholesky').astype(np.float_)
    elif method.lower() == 'av':
        if n % 2 != 0:
            raise ValueError('If method=av, N must be even')
        inc = rng.multivariate_normal(np.zeros(2),
                                      cov,
                                      size=(int(n / 2), k),
                                      method='cholesky').astype(np.float_)
        inc = np.concatenate([inc, -inc], axis=0)
        inc += mean
    else:
        raise GeneratingMethodError(method)
    
    return inc


def calculate_asset_price(s0, alpha, beta, inc):
    """
    Calculates assets' prices based on given random increments.

    Parameters
    ----------
    s0 : 1-D ndarray of floats
        Array of initial prices of assets
    alpha : 1-D ndarray of floats
        Array of standard deviation of logarithmic returns of assets.
    beta : 1-D ndarray of floats
        Array of means of logarithmic returns of assets.
    inc : 3-D ndarray of floats
        Arrangement of random increments.

    Returns
    -------
    3-D ndarray of floats
        Arrangement of asset prices' scenarios.

    """
    x = alpha * inc
    x += beta
    s = s0 * np.exp(np.cumsum(x, axis=1),
                    dtype=np.float_)
    s0_arr = np.array([[s0], ] * inc.shape[0],
                      dtype=np.float_)
    return np.concatenate([s0_arr, s], axis=1)


def calculate_rn_derivative(c, alpha, beta, rho, r, inc, dim=1):
    """
    Calculates Radon-Nikodym derivatives for given random increments.
    Parameters
    ----------
    c : 1-D ndarray of floats
        Array of standard deviations of standard normal variables in martingale measure.
    alpha : 1-D ndarray of floats
        Array of standard deviation of logarithmic returns of assets.
    beta : 1-D ndarray of floats
        Array of means of logarithmic returns of assets.
    rho : float
        Correlation coefficient between assets.
    r : float
        Risk-free rate. Must be given 'per day', that is divided by the number of trading days in a year.
    inc : 3-D ndarray of floats
        Arrangement of random increments.
    dim : int
        Number of assets that change of measure involves.

    Returns
    -------
    1-D ndarray of floats
        Radon-Nikodym derivatives for respective increments.

    """
    if dim == 2:
        rn_0 = _rn_derivative_2d(0, 0, c, alpha, beta, r, rho)
        rn = _rn_derivative_2d(inc[:, :, 0], inc[:, :, 1], c, alpha, beta, r, rho)
    else:
        rn_0 = _rn_derivative_1d(0, c, alpha, beta, r)
        rn = _rn_derivative_1d(inc[:, :, 0], c, alpha, beta, r)

    return np.prod(rn, axis=1) * rn_0


def _rn_derivative_1d(x, c, alpha, beta, r):
    """
    Calculates Radon-Nikodym derivative in case one asset is involved.

    Parameters
    ----------
    x : float or ndarray of floats
        Value based on which derivative is calculated.
    c : 1-D ndarray of floats
        Array of standard deviations of standard normal variables in martingale measure.
    alpha : 1-D ndarray of floats
        Array of standard deviation of logarithmic returns of assets.
    beta : 1-D ndarray of floats
        Array of means of logarithmic returns of assets.
    r : float
        Risk-free rate.

    Returns
    -------
    float or 1-D ndarray of floats
        Radon-Nikodym derivative.
    """
    phi = (beta - r + .5 * (alpha * c) ** 2) / alpha
    z = x ** 2 / 2 - (x + phi[0]) ** 2 / (2 * c[0] ** 2)
    return np.exp(z) / c[0]


def _rn_derivative_2d(x, y, c, alpha, beta, r, rho):
    """
       Calculates Radon-Nikodym derivative in case two assets are involved.

       Parameters
       ----------
       x : float or ndarray of floats
           Value based on which derivative is calculated.
       y : float or ndarray of floats
           Value based on which derivative is calculated.
       c : 1-D ndarray of floats
           Array of standard deviations of standard normal variables in martingale measure.
       alpha : 1-D ndarray of floats
           Array of standard deviation of logarithmic returns of assets.
       beta : 1-D ndarray of floats
           Array of means of logarithmic returns of assets.
       r : float
           Risk-free rate.
       rho : float
           Correlation coefficient between assets.

       Returns
       -------
       float or 1-D ndarray of floats
           Radon-Nikodym derivative.
       """
    phi = (beta - r + .5 * (alpha * c) ** 2) / alpha
    z1 = (x + phi[0]) ** 2 * ((rho * c[0]) ** 2 + (1 - rho ** 2) * c[1] ** 2)
    z2 = 2 * c[0] ** 2 * rho * (y + phi[1] + rho * (c[0] ** 2 - c[1] ** 2) / 2) * (x + phi[0])
    z3 = c[0] ** 2 * (y + phi[1] + rho * (c[0] ** 2 - c[1] ** 2) / 2) ** 2
    z4 = (c[0] * c[1]) ** 2 * (x ** 2 - 2 * rho * x * y + y ** 2)
    z = - (z1 - z2 + z3 - z4) / (2 * (1 - rho ** 2) * (c[0] * c[1]) ** 2)
    return np.exp(z) / (c[0] * c[1])
