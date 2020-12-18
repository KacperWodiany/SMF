import numpy as np

from smf import mc_val, options

# Setting parameters values
alpha = np.array([9.477836454179126499e-03, 1.790058145714905347e-02],
                 dtype=np.float_)

beta = np.array([-1.892486260469783519e-04, -1.229901256661283197e-03],
                dtype=np.float_)

cov_mx = np.array([[8.98293839e-05, 8.59575399e-05],
                   [8.59575399e-05, 3.20430817e-04]],
                  dtype=np.float_)

corr_mx = np.array([[1, 0.50664951],
                    [0.50664951, 1]],
                   dtype=np.float_)

s0 = np.array([2203.7, 84.2],
              dtype=np.float_)

c = np.ones(2, dtype=np.float_)

days = 247
T = 184

rf = np.float_(0.013242203985850343 / days)


# Defining option
A = options.EuropeanOption(2200, 'call')

# Valuating using smf.mc_val.valuate

val = mc_val.valuate(A, s0, rf, T, alpha, beta, corr_mx)

# Valuating step by step

paths = mc_val.generate_scenarios(s0, rf, T, alpha, beta, corr_mx)

numeraire = np.exp(rf * np.arange(0, T+1), dtype=np.float_)

val_step_by_step = mc_val.calculate_option_value(A, paths, numeraire)

# Valuating step by step using measure P

paths_p, inc_p = mc_val.generate_scenarios(s0, rf, T, alpha, beta, corr_mx, with_p=True, return_inc=True)

rn = mc_val.calculate_rn_derivative(c, alpha, beta, corr_mx[0, 1], rf, inc_p)

numeraire = np.exp(rf * np.arange(0, T+1), dtype=np.float_)

val_step_by_step_rn = mc_val.calculate_option_value(A, paths_p, numeraire, rn)

print(val)
print(val_step_by_step)
print(val_step_by_step_rn)

