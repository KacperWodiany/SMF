import numpy as np

from smf import options


def test_european_call_payoff_single_scenario_single_asset():
    o = options.EuropeanOption(1, 'call')
    scen = np.array([[[1],
                      [3]]])
    assert np.array_equiv(o.calculate_payoff(scen), [[2]])


def test_european_put_payoff_single_scenario_single_asset():
    o = options.EuropeanOption(1, 'put')
    scen = np.array([[[1],
                      [0]]])
    assert np.array_equiv(o.calculate_payoff(scen), [[1]])


def test_european_call_payoff_single_scenario_many_assets():
    o = options.EuropeanOption(1, 'call')
    scen = np.array([[[1, 2],
                      [3, 4]]])
    assert np.array_equiv(o.calculate_payoff(scen, (0, 1)), [[2, 3]])


def test_european_call_payoff_many_scenarios_single_asset():
    o = options.EuropeanOption(1, 'call')
    scen = np.array([[[1],
                      [3]],

                     [[2],
                      [4]]])
    assert np.array_equiv(o.calculate_payoff(scen), [[2],
                                                     [3]])


def test_european_call_payoff_many_scenarios_many_assets():
    o = options.EuropeanOption(1, 'call')
    scen = np.array([[[1, 2],
                      [3, 4]],

                     [[5, 6],
                      [7, 8]]])
    assert np.array_equiv(o.calculate_payoff(scen, (0, 1)), [[2, 3],
                                                             [6, 7]])


def test_up_and_out_call_payoff_single_scenario_single_path():
    o = options.UpAndOutOption(4, 8, 'call')
    scen = np.array([[[1],
                      [3],
                      [8],
                      [5]]])
    assert np.array_equiv(o.calculate_payoff(scen), [[0]])


def test_up_and_out_call_payoff_single_scenario_many_assets():
    o = options.UpAndOutOption(4, 8, 'call')
    scen = np.array([[[1, 1],
                      [3, 2],
                      [8, 3],
                      [5, 5]]])
    assert np.array_equiv(o.calculate_payoff(scen, (0, 1)), [[0, 1]])


def test_up_and_out_call_payoff_many_scenarios_single_asset():
    o = options.UpAndOutOption(4, 8, 'call')
    scen = np.array([[[1],
                      [3],
                      [8],
                      [5]],

                     [[1],
                      [3],
                      [4],
                      [5]]])
    assert np.array_equiv(o.calculate_payoff(scen), [[0],
                                                     [1]])


def test_up_and_out_call_payoff_many_scenarios_many_assets():
    o = options.UpAndOutOption(4, 8, 'call')
    scen = np.array([[[1, 1],
                      [3, 2],
                      [8, 3],
                      [5, 5]],

                     [[1, 1],
                      [3, 2],
                      [7, 7],
                      [5, 7]]])
    assert np.array_equiv(o.calculate_payoff(scen, (0, 1)), [[0, 1],
                                                             [1, 3]])


def test_up_and_in_call_payoff_single_scenario_single_path():
    o = options.UpAndInOption(4, 8, 'call')
    scen = np.array([[[1],
                      [3],
                      [8],
                      [5]]])
    assert np.array_equiv(o.calculate_payoff(scen), [[1]])


def test_up_and_in_call_payoff_single_scenario_many_assets():
    o = options.UpAndInOption(4, 8, 'call')
    scen = np.array([[[1, 1],
                      [3, 2],
                      [8, 3],
                      [5, 5]]])
    assert np.array_equiv(o.calculate_payoff(scen, (0, 1)), [[1, 0]])


def test_up_and_in_call_payoff_many_scenarios_single_asset():
    o = options.UpAndInOption(4, 8, 'call')
    scen = np.array([[[1],
                      [3],
                      [8],
                      [5]],

                     [[1],
                      [3],
                      [4],
                      [5]]])
    assert np.array_equiv(o.calculate_payoff(scen), [[1],
                                                     [0]])


def test_up_and_in_call_payoff_many_scenarios_many_assets():
    o = options.UpAndInOption(4, 8, 'call')
    scen = np.array([[[1, 1],
                      [3, 2],
                      [8, 3],
                      [5, 5]],

                     [[1, 1],
                      [3, 2],
                      [7, 8],
                      [5, 7]]])
    assert np.array_equiv(o.calculate_payoff(scen, (0, 1)), [[1, 0],
                                                             [0, 3]])


def test_parisian_up_and_in_call_payoff_single_scenario_single_path():
    o = options.ParisianUpAndInOption(4, 6, 2, 'call')
    scen = np.array([[[1],
                      [3],
                      [8],
                      [5]]])
    assert np.array_equiv(o.calculate_payoff(scen), [[0]])


def test_parisian_up_and_in_call_payoff_single_scenario_many_assets():
    o = options.ParisianUpAndInOption(4, 2, 3, 'call')
    scen = np.array([[[1, 1],
                      [3, 2],
                      [8, 3],
                      [5, 5]]])
    assert np.array_equiv(o.calculate_payoff(scen, (0, 1)), [[1, 0]])


def test_parisian_up_and_in_call_payoff_many_scenarios_single_asset():
    o = options.ParisianUpAndInOption(4, 4, 2, 'call')
    scen = np.array([[[1],
                      [3],
                      [8],
                      [5]],

                     [[1],
                      [3],
                      [4],
                      [5]]])
    assert np.array_equiv(o.calculate_payoff(scen), [[1],
                                                     [0]])


def test_parisian_up_and_in_call_payoff_many_scenarios_many_assets():
    o = options.ParisianUpAndInOption(4, 6, 2, 'call')
    scen = np.array([[[1, 1],
                      [3, 2],
                      [8, 3],
                      [5, 5]],

                     [[1, 1],
                      [3, 2],
                      [7, 8],
                      [5, 7]]])
    assert np.array_equiv(o.calculate_payoff(scen, (0, 1)), [[0, 0],
                                                             [0, 3]])


def test_lookback_call_payoff_single_scenario_single_path():
    o = options.LookbackOption('call')
    scen = np.array([[[1],
                      [3],
                      [8],
                      [5]]])
    assert np.array_equiv(o.calculate_payoff(scen), [[4]])


def test_lookback_call_payoff_single_scenario_many_assets():
    o = options.LookbackOption('call')
    scen = np.array([[[1, 1],
                      [3, 2],
                      [8, 3],
                      [5, 6]]])
    assert np.array_equiv(o.calculate_payoff(scen, (0, 1)), [[4, 5]])


def test_lookback_call_payoff_many_scenarios_single_asset():
    o = options.LookbackOption('call')
    scen = np.array([[[1],
                      [3],
                      [8],
                      [5]],

                     [[1],
                      [3],
                      [4],
                      [5]]])
    assert np.array_equiv(o.calculate_payoff(scen), [[4],
                                                     [4]])


def test_lookback_call_payoff_many_scenarios_many_assets():
    o = options.LookbackOption('call')
    scen = np.array([[[1, 1],
                      [3, 2],
                      [8, 3],
                      [5, 5]],

                     [[1, 1],
                      [3, 2],
                      [7, 8],
                      [5, 7]]])
    assert np.array_equiv(o.calculate_payoff(scen, (0, 1)), [[4, 4],
                                                             [4, 6]])


def test_binary_lookback_call_payoff_single_scenario():
    o = options.BinaryLookbackOption('call')
    scen = np.array([[[1, 1],
                      [3, 2],
                      [8, 3],
                      [5, 6]]])
    assert np.array_equiv(o.calculate_payoff(scen), [[4]])


def test_binary_lookback_call_payoff_many_scenarios():
    o = options.BinaryLookbackOption('call')
    scen = np.array([[[1, 1],
                      [3, 2],
                      [8, 3],
                      [5, 5]],

                     [[1, 10],
                      [3, 2],
                      [7, 8],
                      [6, 7]]])
    assert np.array_equiv(o.calculate_payoff(scen), [[4],
                                                     [0]])
