"""
Provides functions for calculating options' values using Black-Scholes analytical formulas.
"""
import numpy as np
import scipy.stats as stats

from smf.exceptions import OptionTypeError


def european_option_price(option_type, s, e, vol, t_expr, r, d=0):
    """
    Calculate european option price with B-S formula.

    Parameters
    ----------
    option_type : str
        Type of an option. Allowed values are 'call' and 'put'.
    s : float
        Spot price of an asset.
    e : float
        Exercise price.
    vol : float
        Volatility.
    t_expr : float
        Time to expiration.
    r : float, optional
        Risk-free interest rate.
    d : float, optional
        Dividend yield. Specified on interval [0, 1]. The default is 0.

    Raises
    ------
    OptionTypeError
        If option type is neither put nor call.

    Returns
    -------
    float
        European option price.

    References
    ----------
    https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model

    Examples
    --------

    >>> european_option_price('call', 10, 5, .15, .25, 0)
    5.0

    """
    d_1 = ((np.log(s / e) + (r - d + .5 * vol ** 2) * t_expr)
           / (vol * np.sqrt(t_expr)))
    d_2 = ((np.log(s / e) + (r - d - .5 * vol ** 2) * t_expr)
           / (vol * np.sqrt(t_expr)))
    if option_type.lower() == 'call':
        return (s * np.exp(-d * t_expr) * stats.norm.cdf(d_1)
                - e * np.exp(-r * t_expr) * stats.norm.cdf(d_2))
    elif option_type.lower() == 'put':
        return (-s * np.exp(-d * t_expr) * stats.norm.cdf(-d_1)
                + e * np.exp(-r * t_expr) * stats.norm.cdf(-d_2))
    raise OptionTypeError(option_type)
