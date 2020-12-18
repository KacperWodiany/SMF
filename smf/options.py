"""
Provides classes that define different options. The following derivatives are already defined:

- European Options,

- Up and Out Options,

- Up and In Options,

- Parisian Up an In Options,

- Lookback Options,

- Binary Lookback Options.

Particular options are described inside respective classes.
One can define his own custom options, by implementing Abstract Base Option Class and implementing abstract methods.
"""


from abc import ABC, abstractmethod

import numpy as np

from smf import utils
from smf.exceptions import OptionTypeError


class Option(ABC):
    """
    Abstract Base Class for implementing Options. Creating new option requires implementing both abstract methods.

    Methods
    -------
    calculate_payoff
        Calculates payoff of the option.

    _is_in
        Function that determines if option is 'in' (has value) or 'out' (became worthless due to some events).

    """

    @abstractmethod
    def calculate_payoff(self, *args):
        """
        Calculates option payoff.

        Parameters
        ----------
        args

        Returns
        -------
        None.

        """
        raise NotImplementedError()

    @abstractmethod
    def _is_in(self, *args):
        """
        Determines if option is 'in' or 'out'.

        Parameters
        ----------
        args

        Returns
        -------
        None.

        """
        raise NotImplementedError()


class EuropeanOption(Option):
    """
    European Option class. Payoff of the option is defined by max(S-E, 0) or max(E-S, 0), for call and put respectively,
    where S is price of an underlying asset at expiry and E is exercise price.

    """

    def __init__(self, strike, option_type):
        """

        Parameters
        ----------
        strike : float
            Exercise price.
        option_type : str
            Type of the option. Allowed values are 'call' and 'put'. This argument is case insensitive.

        Raises
        ------
        OptionTypeError
            When option_type matches neither 'put' nor 'call'.


        """
        self.strike = strike
        if option_type.lower() not in ['call', 'put']:
            raise OptionTypeError(option_type)
        self.option_type = option_type.lower()

    def calculate_payoff(self, path, idx=(0,)):
        """
        Calculates payoff of the option.

        Parameters
        ----------
        path : 3-D ndarrays of floats
            Arrangement of market scenarios. Single scenario is a 2-D array, with columns corresponding to prices of
            different assets in this scenario.
        idx : tuple of ints, optional
            Indexes of columns (assets) for which payoff is calculated. The default is (0,).

        Returns
        -------
        2-D ndarray of floats
            Payoffs of options.

        Examples
        --------
        >>> o = EuropeanOption(1, 'call')
        >>> a = np.array([[[1], [3]]])
        >>> a
        array([[[1],
                [3]]])
        >>> o.calculate_payoff(a)
        array([[2]])

        >>> b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> b
        array([[[1, 2],
                [3, 4]],
        <BLANKLINE>
               [[5, 6],
                [7, 8]]])
        >>> o.calculate_payoff(b, (0, 1))
        array([[2, 3],
               [6, 7]])

        >>> c = np.array([[[1], [3]], [[2], [4]]])
        >>> c
        array([[[1],
                [3]],
        <BLANKLINE>
               [[2],
                [4]]])
        >>> o.calculate_payoff(c)
        array([[2],
               [3]])



        """
        if self.option_type == 'call':
            payoff = np.maximum(path[:, -1, idx] - self.strike, 0)
        else:
            payoff = np.maximum(self.strike - path[:, -1, idx], 0)
        return payoff * self._is_in(path[:, :, idx])

    def _is_in(self, path):
        """
        Determines if option is 'in' or 'out'. If there are no additional features, european options are always 'in'.

        Parameters
        ----------
        path : 3-D ndarray of floats
            3-D array arrangement of market scenarios.

        Returns
        -------
        bool
            Always True.

        """
        return True


class UpAndOutOption(EuropeanOption):
    """
    Up and Out Option class. Payoff is paid under the condition that price of an underlying asset didn't reach
    barrier level before expiry.

    """

    def __init__(self, strike, barrier, option_type):
        """

        Parameters
        ----------
        strike : float
            Exercise price.
        barrier : float
            Level of asset price at which option becomes worthless ('out'), i.e. its value is equal to 0.
        option_type : str
            Type of an option. Allowed values are 'call' and 'put'. This argument is case insensitive.

        Returns
        -------
        None.

        """
        super().__init__(strike, option_type)
        self.barrier = barrier

    def _is_in(self, path):
        """
        Determines weather option is 'in' or 'out' for given path.

        Parameters
        ----------
        path : 3-D ndarray of floats
            3-D array arrangement of market scenarios.

        Returns
        -------
        2-D ndarray of bools
            Array of boolean values determining if option is 'in' or 'out' for respective asset in respective scenario.

        """

        return np.amax(path, axis=1) < self.barrier


class UpAndInOption(EuropeanOption):
    """
    Up and In Option class. Payoff is paid under the condition that price of an underlying asset reached
    barrier level before expiry.

    """

    def __init__(self, strike, barrier, option_type):
        """

        Parameters
        ----------
        strike : float
            Exercise price.
        barrier : float
            Level of price at which option promises to pay payoff.
        option_type : str
            Type of an option. Allowed values are 'call' and 'put'. This argument is case insensitive.

        Returns
        -------
        None.

        """
        super().__init__(strike, option_type)
        self.barrier = barrier

    def _is_in(self, path):
        """
        Determines weather option is 'in' or 'out' for given path.

        Parameters
        ----------
        path : 3-D ndarray of floats
            3-D array arrangement of market scenarios.

        Returns
        -------
        2-D ndarray of bools
            Array of boolean values determining if option is 'in' or 'out' for respective asset in respective scenario.

        """
        return np.amax(path, axis=1) >= self.barrier


class ParisianUpAndInOption(EuropeanOption):
    """
    Up and In Option with additional parisian condition class. Payoff is paid under the condition that price
    of an underlying asset was above some level for given amount of time (days) before expiry.

    """

    def __init__(self, strike, barrier, condition_len, option_type):
        """

        Parameters
        ----------
        strike : float
            Exercise price.
        barrier : float
            Level of price that has to be exceeded for given amount of time.
        condition_len : int
            Time (number of days) for which price of asset must be above
            barrier.
        option_type : str
            Type of an option. Allowed values are 'call' and 'put'. This argument is case insensitive.

        Returns
        -------
        None.

        """
        super().__init__(strike, option_type)
        self.barrier = barrier
        self.condition_len = condition_len

    def _is_in(self, path):
        """
        Determines weather option is 'in' or 'out' for given path.

        Parameters
        ----------
        path : 3-D ndarray of floats
            3-D array arrangement of market scenarios.

        Returns
        -------
        2-D ndarray of bools
            Array of boolean values determining if option is 'in' or 'out' for respective asset in respective scenario.

        """
        return np.apply_along_axis(self._check_parisian_condition,
                                   1,
                                   path)

    def _check_parisian_condition(self, path):
        """
        Check weather asset price was above barrier for given amount of time.

        Parameters
        ----------
        path : ndarray of floats
            Single path of prices.

        Returns
        -------
        bool
            True if path satisfied parisian condition, False otherwise

        """
        return utils.is_subvector(np.ones(self.condition_len),
                                  path > self.barrier)


class LookbackOption(Option):
    """
    Lookback Option class. Payoff for Lookback option is given by S - min(S) or max(s) - S for call and put
    respectively, where S is price of underlying asset at expiry and min/max is taken with respect to whole
    considered period.
    """

    def __init__(self, option_type):
        """

        Parameters
        ----------
        option_type : str
            Type of an option. Allowed values are 'call' and 'put'. This argument is case insensitive.

        Raises
        ------
        OptionTypeError
            If option_type matches neither 'put' nor 'call'.

        Returns
        -------
        None.

        """
        if option_type.lower() not in ['call', 'put']:
            raise OptionTypeError(option_type)
        self.option_type = option_type.lower()

    def calculate_payoff(self, path, idx=(0,)):
        """
        Calculates payoff of the option.

        Parameters
        ----------
        path : 3-D ndarrays of floats
            Arrangement of market scenarios. Single scenario is a 2-D array, with columns corresponding to
            prices of different assets in this scenario.
        idx : tuple of ints, optional
            Indexes of columns (assets) for which payoff is calculated. The default is (0,),

        Returns
        -------
        2-D ndarray of floats
            Payoffs, of options on specified assets, for each scenario.

        """
        if self.option_type == 'call':
            payoff = path[:, -1, idx] - np.amin(path[:, :, idx], axis=1)
        else:
            payoff = np.amax(path[:, :, idx], axis=1) - path[:, -1, idx]
        return payoff * self._is_in(path[:, :, idx])

    def _is_in(self, path):
        """
        Determines if option is 'in' or 'out'. If there are no additional features, lookback options are always 'in'.

        Parameters
        ----------
        path : 3-D ndarray of floats
            3-D array arrangement of market scenarios.

        Returns
        -------
        bool
            Always True.

        """
        return True


class BinaryLookbackOption(LookbackOption):
    """
    Lookback Option with additional binary condition class. Payoff is paid only if some condition is satisfied
    (regardless sufficient asset price). In this case additional binary condition, is that the price of some
    different asset must exceed its price from the day the option was settled.

    """

    def __init__(self, option_type):
        """

        Parameters
        ----------
        option_type : str
            Type of an option. Allowed values are 'call' and 'put'. This argument is case insensitive.

        Returns
        -------
        None.

        """
        super().__init__(option_type)

    def calculate_payoff(self, path, idx=(0,), cond_idx=1):
        """
        Calculates payoff of the option.

        Parameters
        ----------
        path : 3-D ndarrays of floats
            Arrangement of market scenarios. Single scenario is a 2-D array, with columns corresponding to prices of
            different assets in this scenario.
        idx : tuple of ints, optional
            Indexes of columns (assets) for which payoff is calculated. The default is (0,).
        cond_idx : int, optional
            Index of column with respect to which condition is evaluated. The default is 1.

        Returns
        -------
        2-D ndarray of floats
            Payoffs, of options on specified assets, for each scenario.

        Examples
        --------
        >>> o = BinaryLookbackOption('call')
        >>> a = np.array([[[1, 1], [3, 2], [8, 3], [5, 6]]])
        >>> a
        array([[[1, 1],
                [3, 2],
                [8, 3],
                [5, 6]]])
        >>> o.calculate_payoff(a)
        array([[4]])

        >>> b = np.array([[[1, 1], [3, 2], [8, 3], [5, 5]], [[1, 10], [3, 2], [7, 8], [6, 7]]])
        >>> b
        array([[[ 1,  1],
                [ 3,  2],
                [ 8,  3],
                [ 5,  5]],
        <BLANKLINE>
               [[ 1, 10],
                [ 3,  2],
                [ 7,  8],
                [ 6,  7]]])
        """
        return (super().calculate_payoff(path, idx) *
                self._is_in(path[:, :, cond_idx])[:, np.newaxis])

    def _is_in(self, path):
        """
        Determines weather option is 'in' or 'out' for given path.

        Parameters
        ----------
        path : 3-D ndarrays of floats
            Arrangement of market scenarios. Single scenario is a 2-D array, with columns corresponding to
            prices of different assets in this scenario.

        Returns
        -------
        1-D ndarray of booleans
            Array of boolean values determining if option is 'in' or
            'out' in respective scenario.

        """
        return path[:, -1] > path[:, 0]
