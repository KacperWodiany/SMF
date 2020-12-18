"""
Provides custom exceptions' classes.
"""


class OptionTypeError(ValueError):
    """
    Exception raised when option type is neither put nor call.

    """

    def __init__(self, option_type):
        """

        Parameters
        ----------
        option_type : string
            Option type name that caused an error.

        Returns
        -------
        None.

        """
        self.option_type = option_type
        super().__init__(self.option_type + ' is not a valid option type')


class GeneratingMethodError(ValueError):
    """
    Exception raised when specified generating method is unavailable.

    """

    def __init__(self, method):
        """

        Parameters
        ----------
        method : string
           Monte Carlo estimator name that caused an error.

        Returns
        -------
        None.

        """
        self.method = method
        super().__init__(self.method + ' is not a valid Monte Carlo estimator')
