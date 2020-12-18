import numpy as np

import smf.mc_val as mc


def test_is_discounted_price_process_martingale():
    s0 = np.array([2203.7,   84.2])
    t_expr = 184
    days_a_year = 247
    r = 0.013 / days_a_year
    alpha = np.array([0.00947784, 0.01790058])
    beta = np.array([-0.00018925, -0.0012299])
    corr_mx = np.array([[1., 0.50664951], [0.50664951, 1.]])
    paths = mc.generate_scenarios(s0,
                                  r,
                                  t_expr,
                                  alpha,
                                  beta,
                                  corr_mx)
    num = np.exp(r * np.arange(0, t_expr+1),
                 dtype=np.float_)
    assert np.allclose(np.mean(paths[:, -1, :], axis=0) / num[-1], s0, rtol=.01)

def test_is_univariate_radon_nikodym_expectation_equal_1():
    s0 = np.array([2203.7, 84.2])
    t_expr = 184
    days_a_year = 247
    r = 0.013 / days_a_year
    alpha = np.array([0.00947784, 0.01790058])
    beta = np.array([-0.00018925, -0.0012299])
    corr_mx = np.array([[1., 0.50664951], [0.50664951, 1.]])
    paths, inc = mc.generate_scenarios(s0,
                                       r,
                                       t_expr,
                                       alpha,
                                       beta,
                                       corr_mx,
                                       with_p=True,
                                       return_inc=True)
    rn = mc.calculate_rn_derivative(np.ones(2), alpha, beta, corr_mx[0, 1], r, inc, dim=1)
    assert np.allclose(np.mean(rn), 1, rtol=.01)


def test_is_multivariate_radon_nikodym_expectation_equal_1():
    s0 = np.array([2203.7, 84.2])
    t_expr = 184
    days_a_year = 247
    r = 0.013 / days_a_year
    alpha = np.array([0.00947784, 0.01790058])
    beta = np.array([-0.00018925, -0.0012299])
    corr_mx = np.array([[1., 0.50664951], [0.50664951, 1.]])
    paths, inc = mc.generate_scenarios(s0,
                                       r,
                                       t_expr,
                                       alpha,
                                       beta,
                                       corr_mx,
                                       with_p=True,
                                       return_inc=True)
    rn = mc.calculate_rn_derivative(np.ones(2), alpha, beta, corr_mx[0, 1], r, inc, dim=2)
    assert np.allclose(np.mean(rn), 1, rtol=.01)
