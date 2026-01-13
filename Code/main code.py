# ========================================================================
# 
#    CORRECTLY REFORMULATED WITH CCP20 CONSTRAINTS
#    Fixed: S_bar grid search, proper objective, and constraint handling
# 
# ========================================================================

import pandas as pd
import numpy as np
from scipy.special import comb
from scipy.optimize import linprog
from scipy import stats

from rpy2 import robjects
from rpy2.robjects import FloatVector, r
robjects.r('library(sgt)')
r_qsgt = robjects.r['qsgt']

print("CORRECTLY REFORMULATED CODE WITH CCP20 CONSTRAINTS")

def calculate_cer_pvalues(passive_returns, active_returns, gamma_values=[1, 5, 10]):
    """Calculate CER improvements and p-values"""
    from scipy.stats import ttest_rel
    
    # Calculate monthly statistics
    passive_mean_monthly = np.mean(passive_returns)
    active_mean_monthly = np.mean(active_returns)
    passive_std_monthly = np.std(passive_returns, ddof=1)
    active_std_monthly = np.std(active_returns, ddof=1)
    
    # Annualize
    passive_mean_annual = passive_mean_monthly * 12
    active_mean_annual = active_mean_monthly * 12
    passive_std_annual = passive_std_monthly * np.sqrt(12)
    active_std_annual = active_std_monthly * np.sqrt(12)
    
    results = {}
    
    for gamma in gamma_values:
        # Calculate CERs
        cer_passive = passive_mean_annual - (gamma / 2) * (passive_std_annual ** 2)
        cer_active = active_mean_annual - (gamma / 2) * (active_std_annual ** 2)
        cer_improvement = cer_active - cer_passive
        
        # Monthly CER series for t-test
        passive_var_annual = passive_std_annual ** 2
        active_var_annual = active_std_annual ** 2
        
        cer_passive_series = passive_returns * 12 - (gamma / 2) * passive_var_annual
        cer_active_series = active_returns * 12 - (gamma / 2) * active_var_annual
        
        # Paired t-test
        _, p_value = ttest_rel(cer_active_series, cer_passive_series)
        
        results[gamma] = {
            'improvement': cer_improvement,
            'p_value': p_value
        }
    
    return results

def cvar_projected(returns, probs, level=0.01):
    """Compute CVaR from projected distribution with probability weights."""
    sorted_indices = np.argsort(returns)
    sorted_returns = returns[sorted_indices]
    sorted_probs = probs[sorted_indices]

    cum_prob = np.cumsum(sorted_probs)
    in_tail = cum_prob <= level

    if in_tail.sum() > 0:
        tail_returns = sorted_returns[in_tail]
        tail_probs = sorted_probs[in_tail]
        cvar = np.sum(tail_returns * tail_probs) / tail_probs.sum()
    else:
        cvar = sorted_returns[0]

    return cvar

# ============================================================================
# LOAD DATA (UNCHANGED)
# ============================================================================

options_data = pd.read_csv("data_combined_new.csv")
vix_data = pd.read_csv("VIX.csv")
mz_coefficients = pd.read_csv("mincer_zarnowitz_coefficients_21d.csv") 
mz_coefficients['date'] = pd.to_datetime(mz_coefficients['date'])

sgt_params = pd.read_csv("sgt_parameters_old.csv")
sgt_params['date'] = pd.to_datetime(sgt_params['date'])

# Convert dates
options_data['date'] = pd.to_datetime(options_data['date'])
vix_data['date'] = pd.to_datetime(vix_data['DATE'], format='%m/%d/%Y')

# Merge VIX
options_data = options_data.merge(vix_data[['date', 'CLOSE']], on='date', how='left')
options_data = options_data.rename(columns={'CLOSE': 'VIX'})

# Get unique month IDs
unique_ids = sorted(options_data['id'].unique())

# ============================================================================
# SELECT TRADING DAY (UNCHANGED)
# ============================================================================

print("Selecting trading days with priority logic...")

selected_data_list = []

for month_id in sorted(options_data['id'].unique()):
    month_subset = options_data[options_data['id'] == month_id]

    has_29 = (month_subset['Maturity'] == 29).any()
    has_28 = (month_subset['Maturity'] == 28).any()
    has_30 = (month_subset['Maturity'] == 30).any()

    if has_29:
        selected = month_subset[month_subset['Maturity'] == 29].copy()
        print(f"Month {month_id}: Using 28-day options")
    elif has_28:
        selected = month_subset[month_subset['Maturity'] == 28].copy()
        print(f"Month {month_id}: No 28-day available, using 29-day options")
    elif has_30:
        selected = month_subset[month_subset['Maturity'] == 30].copy()
        print(f"Month {month_id}: No 29 or 28-day available, using 30-day options")
    else:
        print(f"Month {month_id}: WARNING - No valid options available, skipping!")
        continue

    selected_data_list.append(selected)

# Combine selected data
options_data = pd.concat(selected_data_list, ignore_index=True)

print(f"\nTotal selected options: {len(options_data)}")
print(f"Unique months: {options_data['id'].nunique()}")

# ============================================================================
# INITIALIZE STORAGE FOR RESULTS (UNCHANGED)
# ============================================================================

option_payoffs_list = []
lpm_results_list = []
performance_list = []
composition_list = []
positions_list = []

# ============================================================================
# LOOP THROUGH ALL MONTHS
# ============================================================================

print("Starting analysis for all months...\n")

for month_id in unique_ids:

    print(f"{'='*80}")
    print(f"Month ID: {month_id}")
    print(f"{'='*80}")

    try:
        # ====================================================================
        # FILTER DATA FOR THIS MONTH (UNCHANGED)
        # ====================================================================

        month_data = options_data[options_data['id'] == month_id].copy()

        if len(month_data) == 0:
            print(f"No data for month {month_id}, skipping...\n")
            continue

        date = month_data['date'].iloc[0]
        print(f"Date: {date}")
        print(f"Options: {len(month_data)} (Calls: {(month_data['type'] == 'call').sum()}, Puts: {(month_data['type'] == 'put').sum()})")

        # ====================================================================
        # GET PARAMETERS (UNCHANGED)
        # ====================================================================

        S0 = month_data['sp'].iloc[0]
        S0_hat = month_data['sp_hat'].iloc[0]
        rf = month_data['rf'].iloc[0]
        q = month_data['q'].iloc[0]
        VIX = month_data['VIX'].iloc[0]
        T = month_data['Maturity'].iloc[0] / 365

        if pd.isna(VIX):
            print(f"Missing VIX, skipping...\n")
            continue

        print(f"S0: {S0:.2f}, rf: {rf:.4f}, q: {q:.4f}, VIX: {VIX:.2f}")

        # Get M-Z coefficients
        mz_row = mz_coefficients[mz_coefficients['id'] == month_id]

        if len(mz_row) == 0:
            mz_row = mz_coefficients[mz_coefficients['date'] == date]

            if len(mz_row) == 0:
                print(f"WARNING: No M-Z coefficients for month {month_id}")
                c0 = mz_coefficients['c0'].median()
                c1 = mz_coefficients['c1'].median()
                n_obs = np.nan
            else:
                c0 = mz_row['c0'].iloc[0]
                c1 = mz_row['c1'].iloc[0]
                n_obs = mz_row['n_obs'].iloc[0]
        else:
            c0 = mz_row['c0'].iloc[0]
            c1 = mz_row['c1'].iloc[0]
            n_obs = mz_row['n_obs'].iloc[0]

        print(f"M-Z Coefficients: c0={c0:.4f}, c1={c1:.4f}")

        # Get SGT parameters
        sgt_row = sgt_params[sgt_params['id'] == month_id]

        if len(sgt_row) == 0:
            sgt_row = sgt_params[sgt_params['date'] == date]

            if len(sgt_row) == 0:
                theta1 = sgt_params['theta1'].median()
                theta2 = sgt_params['theta2'].median()
                theta3 = sgt_params['theta3'].median()
            else:
                theta1 = sgt_row['theta1'].iloc[0]
                theta2 = sgt_row['theta2'].iloc[0]
                theta3 = sgt_row['theta3'].iloc[0]
        else:
            theta1 = sgt_row['theta1'].iloc[0]
            theta2 = sgt_row['theta2'].iloc[0]
            theta3 = sgt_row['theta3'].iloc[0]

        print(f"SGT Parameters: θ1={theta1:.4f}, θ2={theta2:.4f}, θ3={theta3:.4f}")


        # ====================================================================
        # GENERATE SGT-BASED SCENARIOS (UNCHANGED)
        # ====================================================================

        n_scenarios = 150

        VIX_decimal = VIX / 100
        sigma_annual = VIX_decimal

        # DYNAMIC bias correction for VIX
        sigma_annual_corrected = c0 + c1 * VIX_decimal
        sigma_period = sigma_annual_corrected * np.sqrt(T)

        # CAPM drift
        gamma = 3.25
        lambda_period = gamma * (sigma_period ** 2)
        mu = (rf - q) * T + lambda_period

        # BINOMIAL PROBABILITIES
        from scipy.special import gammaln

        n_steps = n_scenarios - 1
        k_vals = np.arange(n_scenarios)
        log_binom = gammaln(n_steps + 1) - gammaln(k_vals + 1) - gammaln(n_steps - k_vals + 1)
        log_probs = log_binom - n_steps * np.log(2)
        probs = np.exp(log_probs)
        probs = probs / probs.sum()
        
        # # ============================================
        # # TRUNCATION (as per paper Section 3.2.3)
        # # ============================================
        # # Remove extreme tail events with cumulative probability < 1e-13
        # cumulative_prob_temp = np.cumsum(probs)
        # lower_threshold = 1e-13
        # upper_threshold = 1 - 1e-13
        
        # # Find indices to keep
        # keep_indices = (cumulative_prob_temp > lower_threshold) & (cumulative_prob_temp < upper_threshold)
        
        # # Only apply truncation if there are scenarios to remove
        # if not np.all(keep_indices):
        #     # Store the original number
        #     n_scenarios_original = n_scenarios
            
        #     # Truncate probabilities
        #     probs = probs[keep_indices]
            
        #     # Renormalize probabilities to sum to 1
        #     probs = probs / probs.sum()
            
        #     # Update k for the quantile generation
        #     k_vals = k_vals[keep_indices]
            
        #     # Update number of scenarios
        #     n_scenarios = len(probs)
            

        # Cumulative probabilities and midpoints
        cumulative_prob = np.cumsum(probs)
        u_vals = cumulative_prob - probs / 2

        # Generate SGT quantiles
        r_probs = FloatVector(u_vals.tolist())

        robjects.r('''
            get_sgt_quantiles <- function(probs, mu_val, sigma_val, lambda_val, p_val, q_val) {
                sgt::qsgt(probs, mu=mu_val, sigma=sigma_val, lambda=lambda_val, p=p_val, q=q_val)
            }
        ''')
        r_get_quantiles = robjects.r['get_sgt_quantiles']
        theta1 = float(theta1)
        theta2 = float(theta2)
        theta3 = float(theta3)

        z_sgt = np.array(r_get_quantiles(r_probs, 0.0, 1.0, theta3, theta1, theta2))

        # Generate return scenarios
        R_scenarios = mu + sigma_period * z_sgt
        scenarios = S0 * (1.0 + R_scenarios)
        scenarios = np.maximum(scenarios, 0.01)  # Avoid division by zero
        

        # ====================================================================
        # BUILD PAYOFF MATRIX (UNCHANGED)
        # ====================================================================

        n_options = len(month_data)
        payoff_matrix = np.zeros((n_scenarios, n_options))

        for i in range(n_options):
            strike = month_data.iloc[i]['x']
            option_type = month_data.iloc[i]['type']

            if option_type == 'call':
                payoff_matrix[:, i] = np.maximum(scenarios - strike, 0)
            else:
                payoff_matrix[:, i] = np.maximum(strike - scenarios, 0)

        # ====================================================================
        # SETUP LP WITH CCP20 CONSTRAINTS - GRID SEARCH OVER S_BAR
        # ====================================================================

        asks = month_data['ask'].values
        bids = month_data['bid'].values

        # CCP20 parameters
        k = 0.0025  # Transaction cost rate for index (CCP20 uses 0.25% one-way)
        n_vars = 2 * n_options

        # *** CCP20 APPROACH: Grid search over S_bar values ***
        # Paper uses range [S_t, 1.15*S_t] with fine partition
        S_bar_grid = np.linspace(S0, 1.15 * S0, 60)  # CCP20 partitions this segment

        best_result = None
        best_expected_return  = -np.inf
        best_S_bar = None
        feasible_count = 0

        print(f"\nSearching over {len(S_bar_grid)} S_bar values...")

        for S_bar in S_bar_grid:

            # *** OBJECTIVE: MAXIMIZE EXPECTED PAYOFF (CCP20 Eq. 5) ***
            # Note: linprog minimizes, so we negate
            expected_payoffs = probs @ payoff_matrix
            c = np.concatenate([-expected_payoffs, expected_payoffs])

            A_ub = []
            b_ub = []

            # ====================================================================
            # CCP20 CONSTRAINT (4a): A(S_{t+1}) >= 0 for S_{t+1} <= S_bar
            # ====================================================================

            for j in range(n_scenarios):
                if scenarios[j] <= S_bar:
                    # Convert payoff to index units with transaction costs
                    # Long: buy option, sell at expiry → pay (1+k) to convert to index
                    # Short: sell option, buy at expiry → receive (1-k) when convert

                    payoff_per_dollar_long = payoff_matrix[j, :] / (scenarios[j] * (1 + k))
                    payoff_per_dollar_short = payoff_matrix[j, :] / (scenarios[j] * (1 - k))

                    # A(S) = alpha * payoff_long - beta * payoff_short >= 0
                    # Rearrange: -alpha * payoff_long + beta * payoff_short <= 0
                    constraint = np.concatenate([-payoff_per_dollar_long, payoff_per_dollar_short])
                    A_ub.append(constraint)
                    b_ub.append(0)

            # ====================================================================
            # CCP20 CONSTRAINT (4b): A(S_{t+1}) <= 0 for S_{t+1} >= S_bar
            # ====================================================================
    
            # *** ADD THIS LINE ***
            lower_bound_CCP = 0.6 * S0  # CCP's lower threshold
            
            for j in range(n_scenarios):
                # *** CORRECTED CONDITION ***
                if scenarios[j] >= lower_bound_CCP and scenarios[j] <= S_bar:
                    
                    # Convert payoff to index units with transaction costs
                    payoff_per_dollar_long = payoff_matrix[j, :] / (scenarios[j] * (1 + k))
                    payoff_per_dollar_short = payoff_matrix[j, :] / (scenarios[j] * (1 - k))
            
                    # A(S) = alpha * payoff_long - beta * payoff_short >= 0
                    # Rearrange: -alpha * payoff_long + beta * payoff_short <= 0
                    constraint = np.concatenate([-payoff_per_dollar_long, payoff_per_dollar_short])
                    A_ub.append(constraint)
                    b_ub.append(0)

            # ====================================================================
            # CCP20 CONSTRAINT (4d): Non-increasing payoff for S > S_bar
            # Check that slope A'(S) <= 0 between consecutive strikes above S_bar
            # ====================================================================

            strikes = month_data['x'].values
            option_types = month_data['type'].values

            # Get sorted unique strike prices above S_bar
            strikes_above = strikes[strikes > S_bar]
            strikes_above_unique = np.unique(strikes_above)

            if len(strikes_above_unique) > 1:
                for idx in range(len(strikes_above_unique) - 1):
                    K_low = strikes_above_unique[idx]
                    K_high = strikes_above_unique[idx + 1]

                    # Find scenarios at or near these strikes
                    # Use scenarios closest to these strikes
                    j_low = np.argmin(np.abs(scenarios - K_low))
                    j_high = np.argmin(np.abs(scenarios - K_high))

                    if scenarios[j_low] >= S_bar and scenarios[j_high] >= S_bar and j_high > j_low:
                        # Slope = (A(S_high) - A(S_low)) / (S_high - S_low) <= 0
                        # => A(S_high) <= A(S_low)
                        # => A(S_high) - A(S_low) <= 0

                        payoff_high_long = payoff_matrix[j_high, :] / (scenarios[j_high] * (1 + k))
                        payoff_high_short = payoff_matrix[j_high, :] / (scenarios[j_high] * (1 - k))

                        payoff_low_long = payoff_matrix[j_low, :] / (scenarios[j_low] * (1 + k))
                        payoff_low_short = payoff_matrix[j_low, :] / (scenarios[j_low] * (1 - k))

                        # (alpha*payoff_high_long - beta*payoff_high_short) - (alpha*payoff_low_long - beta*payoff_low_short) <= 0
                        diff_long = payoff_high_long - payoff_low_long
                        diff_short = payoff_high_short - payoff_low_short

                        constraint = np.concatenate([diff_long, -diff_short])
                        A_ub.append(constraint)
                        b_ub.append(0)

            # ====================================================================
            # POSITION LIMIT
            # ====================================================================

            position_constraint = np.ones(n_vars)
            A_ub.append(position_constraint)
            b_ub.append(1)

            # ====================================================================
            # CCP20 CONSTRAINT (2): Zero net cost EQUALITY constraint
            # ====================================================================

            net_cost_constraint = np.concatenate([asks, -bids])
            A_eq = [net_cost_constraint]
            b_eq = [0]

            # Convert to arrays and check for invalid values
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)


            # *** FIX: Check for inf/nan values ***
            if np.any(~np.isfinite(A_ub)) or np.any(~np.isfinite(b_ub)):
                continue  # Skip this S_bar if constraints are invalid

            bounds = [(0, None) for _ in range(n_vars)]

            # ====================================================================
            # SOLVE LP FOR THIS S_BAR
            # ====================================================================

            result = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs',
                options={'presolve': True, 'disp': False}
            )

            if result.success:
                feasible_count += 1

                # Extract positions
                alpha_temp = result.x[:n_options]
                beta_temp = result.x[n_options:]
                # Calculate expected return = net premium + E[payoff]
                net_premium_temp = np.sum(bids * beta_temp) - np.sum(asks * alpha_temp)
                # Calculate Sharpe ratio for selection (CCP20 base case criterion)
                option_payoffs_temp = (alpha_temp - beta_temp) @ payoff_matrix.T
                expected_payoff_temp = probs @ option_payoffs_temp
                
                expected_return_temp = net_premium_temp * (1 + rf)**T + expected_payoff_temp

                if expected_return_temp  > best_expected_return:
                    best_expected_return = expected_return_temp
                    best_result = result
                    best_S_bar = S_bar


        print(f"Found {feasible_count} feasible S_bar values out of {len(S_bar_grid)}")

        if best_result is None:
            print(f"LP FAILED: No feasible S_bar found\n")
            continue

        print(f"Best S_bar: {best_S_bar:.2f} (Expected Return: {best_expected_return:.6f})")

        # ====================================================================
        # EXTRACT OPTIMAL SOLUTION
        # ====================================================================

        alpha_vals = best_result.x[:n_options]
        beta_vals = best_result.x[n_options:]

        net_premium = np.sum(bids * beta_vals) - np.sum(asks * alpha_vals)
        net_premium_with_interest = net_premium * (1 + rf) ** T
        premium_percentage = net_premium / S0

        print(f"Net Premium: {net_premium:.6f} (should be ~0)")
        print(f"Alpha sum: {alpha_vals.sum():.4f}, Beta sum: {beta_vals.sum():.4f}")

        # ====================================================================
        # TRACK PORTFOLIO COMPOSITION (NEW!)
        # ====================================================================
        
        selection_threshold = 1e-6
        
        month_composition = {
            'buy_call_weight': 0,
            'buy_put_weight': 0,
            'write_call_weight': 0,
            'write_put_weight': 0,
            # Moneyness lists
            'buy_call_moneyness': [],
            'buy_call_positions': [],        # NEW!
            'write_call_moneyness': [],
            'write_call_positions': [],      # NEW!
            'buy_put_moneyness': [],
            'buy_put_positions': [],         # NEW!
            'write_put_moneyness': [],
            'write_put_positions': [],       # NEW!
            # IV lists
            'buy_call_iv': [],
            'buy_call_iv_positions': [],     # NEW!
            'write_call_iv': [],
            'write_call_iv_positions': [],   # NEW!
            'buy_put_iv': [],
            'buy_put_iv_positions': [],      # NEW!
            'write_put_iv': [],
            'write_put_iv_positions': [],    # NEW!
        }
        
        for i in range(n_options):
            option_type = month_data.iloc[i]['type']
            moneyness = month_data.iloc[i]['moneyness']
            iv = month_data.iloc[i]['IV']
            
            if alpha_vals[i] > selection_threshold:
                if option_type == 'call':
                    month_composition['buy_call_weight'] += alpha_vals[i]
                    month_composition['buy_call_moneyness'].append(moneyness * 100)
                    month_composition['buy_call_positions'].append(alpha_vals[i])
                    month_composition['buy_call_iv'].append(iv)
                    month_composition['buy_call_iv_positions'].append(alpha_vals[i])
                else:
                    month_composition['buy_put_weight'] += alpha_vals[i]
                    month_composition['buy_put_moneyness'].append(moneyness * 100)
                    month_composition['buy_put_positions'].append(alpha_vals[i])
                    month_composition['buy_put_iv'].append(iv)
                    month_composition['buy_put_iv_positions'].append(alpha_vals[i])
            
            if beta_vals[i] > selection_threshold:
                if option_type == 'call':
                    month_composition['write_call_weight'] += beta_vals[i]
                    month_composition['write_call_moneyness'].append(moneyness * 100)
                    month_composition['write_call_positions'].append(beta_vals[i])
                    month_composition['write_call_iv'].append(iv)
                    month_composition['write_call_iv_positions'].append(beta_vals[i])
                else:
                    month_composition['write_put_weight'] += beta_vals[i]
                    month_composition['write_put_moneyness'].append(moneyness * 100)
                    month_composition['write_put_positions'].append(beta_vals[i])
                    month_composition['write_put_iv'].append(iv)
                    month_composition['write_put_iv_positions'].append(beta_vals[i])
        
        def weighted_avg(values, weights):
            if len(values) == 0 or len(weights) == 0:
                return np.nan
            return np.average(values, weights=weights)
        
        # Store composition data
        composition_row = {
            'id': month_id,
            'date': date,
            'buy_call_weight': month_composition['buy_call_weight'],
            'buy_put_weight': month_composition['buy_put_weight'],
            'write_call_weight': month_composition['write_call_weight'],
            'write_put_weight': month_composition['write_put_weight'],
            'write_call_moneyness': month_composition['write_call_moneyness'],
            'buy_call_moneyness': month_composition['buy_call_moneyness'],
            'buy_call_moneyness_mean': weighted_avg(
                month_composition['buy_call_moneyness'],
                month_composition['buy_call_positions']
            ),
            'write_call_moneyness_mean': weighted_avg(
                month_composition['write_call_moneyness'],
                month_composition['write_call_positions']
            ),
            'buy_put_moneyness_mean': weighted_avg(
                month_composition['buy_put_moneyness'],
                month_composition['buy_put_positions']
            ),
            'write_put_moneyness_mean': weighted_avg(
                month_composition['write_put_moneyness'],
                month_composition['write_put_positions']
            ),
            'buy_call_iv_mean': weighted_avg(
                month_composition['buy_call_iv'],
                month_composition['buy_call_iv_positions']
            ),
            'write_call_iv_mean': weighted_avg(
                month_composition['write_call_iv'],
                month_composition['write_call_iv_positions']
            ),
            'buy_put_iv_mean': weighted_avg(
                month_composition['buy_put_iv'],
                month_composition['buy_put_iv_positions']
            ),
            'write_put_iv_mean': weighted_avg(
                month_composition['write_put_iv'],
                month_composition['write_put_iv_positions']
            ),
        }
        
        composition_list.append(composition_row)
        
        # ====================================================================
        # CALCULATE RETURNS
        # ====================================================================
        
        # Calculate option payoffs by scenario
        option_payoffs_by_scenario = (alpha_vals - beta_vals) @ payoff_matrix.T

        # Calculate returns
        index_returns = (scenarios - S0) / S0
        active_payoff = (scenarios - S0 + option_payoffs_by_scenario) / S0
        enhanced_payoff = (scenarios - S0 + net_premium_with_interest + option_payoffs_by_scenario) / S0

        expected_index_return = np.sum(probs * index_returns)
        expected_active_payoff = np.sum(probs * active_payoff)
        expected_enhanced_return = np.sum(probs * enhanced_payoff)
        expected_outperformance = expected_active_payoff - expected_index_return

        print(f"Expected Index Return: {expected_index_return*100:.4f}%")
        print(f"Expected Active Payoff: {expected_active_payoff*100:.4f}%")
        
        print(f"Net Premium: {net_premium:.4f}")
        print(f"Expected Index Return: {expected_index_return*100:.4f}%")
        print(f"Expected Active Payoff: {expected_active_payoff*100:.4f}%")
            
        print(f"Month {month_id}:")
        print(f"  rf: {rf}")
        print(f"  q: {q}")
        print(f"  VIX: {VIX}")
        print(f"  mu: {mu}")
        print(f"  mu*T: {mu*T}")
        print(f"  Expected return: {expected_index_return}")
        
        # print(f"  sigma: {sigma}")
        # print(f"  z_sgt mean: {z_sgt.mean()}")
        # print(f"  z_sgt std: {z_sgt.std()}")
        # print(f"  R_scenarios mean: {R_scenarios.mean()}")
        # print(f"  R_scenarios std: {R_scenarios.std()}")
        # print(f"  scenarios mean: {scenarios.mean()}")
        
        sorted_passive = np.sort(index_returns)
        sorted_active = np.sort(enhanced_payoff)
        cutoff = max(1, int(n_scenarios * 0.01))
        proj_passive_cvar = cvar_projected(index_returns, probs, level=0.01)
        proj_active_cvar = cvar_projected(enhanced_payoff, probs, level=0.01)
        
        # ====================================================================
        # CALCULATE LPM
        # ====================================================================
        
        max_return = index_returns.max()
        min_return = index_returns.min()
        n_thresholds = 150
        step_size = (max_return - min_return) / 140
        
        thresholds = np.zeros(n_thresholds)
        thresholds[0] = min_return
        for i in range(1, n_thresholds):
            thresholds[i] = thresholds[i-1] + step_size
        
        lpm_index = np.zeros(n_thresholds)
        lpm_portfolio = np.zeros(n_thresholds)
        
        for i in range(n_thresholds):
            threshold = thresholds[i]
            
            indicator_index = (index_returns <= threshold).astype(float)
            lpm_index[i] = np.sum(probs * (threshold - index_returns) * indicator_index)
            
            indicator_portfolio = (active_payoff <= threshold).astype(float)
            lpm_portfolio[i] = np.sum(probs * (threshold - active_payoff) * indicator_portfolio)
        
        lpm_difference = lpm_index - lpm_portfolio
        dominance_violations = (lpm_portfolio > lpm_index).sum()
        
        print(f"SSD Dominance: {'YES' if dominance_violations == 0 else f'NO ({dominance_violations} violations)'}")
        

        # ====================================================================
        # OUT-OF-SAMPLE EVALUATION (ENHANCED WITH PER-OPTION-TYPE TRACKING)
        # ====================================================================
        
        S_expiry = month_data['Spot_Expiration'].iloc[0]
        
            
        # Initialize tracking for each option type
        actual_option_payoff = 0
        buy_call_payoff = 0
        buy_put_payoff = 0
        write_call_payoff = 0
        write_put_payoff = 0
        
        buy_call_premium_paid = 0
        buy_put_premium_paid = 0
        write_call_premium_received = 0
        write_put_premium_received = 0
        
        # Calculate payoffs by option type
        for i in range(n_options):
            strike = month_data.iloc[i]['x']
            option_type = month_data.iloc[i]['type']
            ask_price = month_data.iloc[i]['ask']
            bid_price = month_data.iloc[i]['bid']
            
            # Calculate intrinsic value at expiry
            if option_type == 'call':
                payoff_i = max(S_expiry - strike, 0)
            else:
                payoff_i = max(strike - S_expiry, 0)
            
            # Track long positions (bought options)
            if alpha_vals[i] > 1e-6:
                if option_type == 'call':
                    buy_call_payoff += alpha_vals[i] * payoff_i
                    buy_call_premium_paid += alpha_vals[i] * ask_price
                else:
                    buy_put_payoff += alpha_vals[i] * payoff_i
                    buy_put_premium_paid += alpha_vals[i] * ask_price
            
            # Track short positions (written options)
            if beta_vals[i] > 1e-6:
                if option_type == 'call':
                    write_call_payoff -= beta_vals[i] * payoff_i  # Negative because we pay out
                    write_call_premium_received += beta_vals[i] * bid_price
                else:
                    write_put_payoff -= beta_vals[i] * payoff_i  # Negative because we pay out
                    write_put_premium_received += beta_vals[i] * bid_price
            
            # Total payoff
            actual_option_payoff += alpha_vals[i] * payoff_i - beta_vals[i] * payoff_i
        
        # Calculate returns for each option type (including premiums with interest)
        # Return = (Payoff - Premium_Paid) / S0  OR  (Payoff + Premium_Received) / S0
        
        if buy_call_premium_paid > 0:
            buy_call_return = (buy_call_payoff - buy_call_premium_paid * (1 + rf) ** T) / S0
        else:
            buy_call_return = 0
        
        if buy_put_premium_paid > 0:
            buy_put_return = (buy_put_payoff - buy_put_premium_paid * (1 + rf) ** T) / S0
        else:
            buy_put_return = 0
        
        if write_call_premium_received > 0:
            write_call_return = (write_call_payoff + write_call_premium_received * (1 + rf) ** T) / S0
        else:
            write_call_return = 0
        
        if write_put_premium_received > 0:
            write_put_return = (write_put_payoff + write_put_premium_received * (1 + rf) ** T) / S0
        else:
            write_put_return = 0
        
        # Overall portfolio returns
        realized_index_return = (S_expiry - S0) / S0
        realized_enhanced_return = (S_expiry - S0 + net_premium_with_interest + actual_option_payoff) / S0
        realized_outperformance = realized_enhanced_return - realized_index_return
        
        print(f"\nOUT-OF-SAMPLE:")
        print(f"  S_expiry: {S_expiry:.2f}")
        print(f"  BuyCall Return: {buy_call_return*100:.2f}%")
        print(f"  BuyPut Return: {buy_put_return*100:.2f}%")
        print(f"  WriteCall Return: {write_call_return*100:.2f}%")
        print(f"  WritePut Return: {write_put_return*100:.2f}%")
        print(f"  Total Enhanced Return: {realized_enhanced_return*100:.4f}%")
        

        
        oos_available = True
        
        # ====================================================================
        # STORE RESULTS (UPDATED TO INCLUDE PER-OPTION-TYPE RETURNS)
        # ====================================================================
        proj_passive_mean = expected_index_return  # Monthly
        proj_passive_std = np.sqrt(np.sum(probs * (index_returns - expected_index_return)**2))
        mu3_passive = np.sum(probs * (index_returns - proj_passive_mean)**3)
        proj_passive_skew = mu3_passive / (proj_passive_std**3)
        
        proj_active_mean = expected_enhanced_return  # Monthly
        proj_active_std = np.sqrt(np.sum(probs * (enhanced_payoff - expected_enhanced_return)**2))
        mu3_active = np.sum(probs * (enhanced_payoff - proj_active_mean)**3)
        proj_active_skew = mu3_active / (proj_active_std**3)
        
        
        perf_row = {
            'id': month_id,
            'date': date,
            'S0': S0,
            'VIX': VIX,
            'S_expiry': S_expiry,
            'premium_percentage': premium_percentage,
            'proj_passive_mean': proj_passive_mean,
            'proj_passive_std': proj_passive_std,
            'proj_passive_skew': proj_passive_skew,
            'proj_passive_cvar': proj_passive_cvar,
            'proj_active_mean': proj_active_mean,
            'proj_active_std': proj_active_std,
            'proj_active_skew': proj_active_skew,
            'proj_active_cvar': proj_active_cvar,
            'expected_index_return': expected_index_return,
            'expected_active_payoff': expected_active_payoff,
            'expected_enhanced_return': expected_enhanced_return,
            'expected_outperformance': expected_outperformance,
            'realized_index_return': realized_index_return,
            'realized_enhanced_return': realized_enhanced_return,
            'realized_outperformance': realized_outperformance,
            'net_premium': net_premium,
            'actual_option_payoff': actual_option_payoff,
            'buy_call_return': buy_call_return,
            'buy_put_return': buy_put_return,
            'write_call_return': write_call_return,
            'write_put_return': write_put_return,
            'ssd_dominance': (dominance_violations == 0),
            'oos_available': oos_available
        }   
        performance_list.append(perf_row)
        
        positions_list.append({
            'id': month_id,
            'date': date,
            'S0': S0,
            'strikes': month_data['x'].values.tolist(),
            'types': month_data['type'].values.tolist(),
            'alpha': alpha_vals.tolist(),
            'beta': beta_vals.tolist()
        })
     
        print()

    except Exception as e:
        print(f"ERROR: {str(e)}\n")
        continue

# ============================================================================
# CREATE DATAFRAMES
# ============================================================================
performance_df = pd.DataFrame(performance_list)
print(f"\n{'='*80}")
print("CREATING RESULT TABLES")
print(f"{'='*80}\n")

option_payoffs_df = pd.DataFrame(option_payoffs_list)
lpm_results_df = pd.DataFrame(lpm_results_list)
performance_df = pd.DataFrame(performance_list)
composition_df = pd.DataFrame(composition_list)


# Helper functions
def calculate_cer_from_mean_std(mean_return, std_return, gamma):
    if gamma == 1:
        return mean_return - 0.5 * (std_return ** 2)
    else:
        variance = std_return ** 2
        return mean_return - (gamma / 2) * variance * (1 + mean_return)

def calculate_cvar(returns, confidence_level=0.99):
    if len(returns) == 0:
        return np.nan
    sorted_returns = np.sort(returns)
    cutoff_index = max(1, int(len(sorted_returns) * (1 - confidence_level)))
    return sorted_returns[:cutoff_index].mean()

def annualize_return(monthly_return):
    return monthly_return * 12  # Simple annualization

def annualize_vol(monthly_vol):
    return monthly_vol * np.sqrt(12)

# ===========================================================================
# USE ALL DATA (NO VIX FILTERING)
# ===========================================================================

# Get VIX for display purposes only
vix_data_clean = pd.read_csv("VIX.csv")
vix_data_clean['date'] = pd.to_datetime(vix_data_clean['DATE'], format='%m/%d/%Y')
vix_merge = performance_df[['id', 'date']].merge(
    vix_data_clean[['date', 'CLOSE']].rename(columns={'CLOSE': 'VIX'}),
    on='date',
    how='left'
)
performance_df['VIX'] = vix_merge['VIX'].values

# NO FILTERING - Use all data
all_data_df = performance_df.copy()
all_composition = composition_df.copy()
oos_data = all_data_df[~all_data_df['realized_index_return'].isna()].copy()

print(f"Total dates: {len(all_data_df)}")
print(f"Dates with OOS: {len(oos_data)}\n")



# ===========================================================================
# CALCULATE STATISTICS
# ===========================================================================

n_dates = len(all_data_df)
n_nonzeros = (
    (composition_df['buy_call_weight'] > 1e-6) |
    (composition_df['buy_put_weight'] > 1e-6) |
    (composition_df['write_call_weight'] > 1e-6) |
    (composition_df['write_put_weight'] > 1e-6)
).sum()

# Breadth
breadth_by_month = options_data.groupby('id').size()
breadth_median = int(breadth_by_month.median())
breadth_iqr = int(breadth_by_month.quantile(0.75) - breadth_by_month.quantile(0.25))

# VIX stats (for display)
vix_mean = all_data_df['VIX'].mean()

# Composition stats - ALL 4 option types
comp_stats = {
    'buy_call': (all_composition['buy_call_weight'].median(), 
                 all_composition['buy_call_weight'].quantile(0.75) - all_composition['buy_call_weight'].quantile(0.25)),
    'buy_put': (all_composition['buy_put_weight'].median(),
                all_composition['buy_put_weight'].quantile(0.75) - all_composition['buy_put_weight'].quantile(0.25)),
    'write_call': (all_composition['write_call_weight'].median(),
                   all_composition['write_call_weight'].quantile(0.75) - all_composition['write_call_weight'].quantile(0.25)),
    'write_put': (all_composition['write_put_weight'].median(),
                  all_composition['write_put_weight'].quantile(0.75) - all_composition['write_put_weight'].quantile(0.25)),
}

# Moneyness stats
moneyness_stats = {
    'buy_call': (all_composition['buy_call_moneyness_mean'].median(),
                 all_composition['buy_call_moneyness_mean'].quantile(0.75) - all_composition['buy_call_moneyness_mean'].quantile(0.25)),
    'buy_put': (all_composition['buy_put_moneyness_mean'].median(),
                all_composition['buy_put_moneyness_mean'].quantile(0.75) - all_composition['buy_put_moneyness_mean'].quantile(0.25)),
    'write_call': (all_composition['write_call_moneyness_mean'].median(),
                   all_composition['write_call_moneyness_mean'].quantile(0.75) - all_composition['write_call_moneyness_mean'].quantile(0.25)),
    'write_put': (all_composition['write_put_moneyness_mean'].median(),
                  all_composition['write_put_moneyness_mean'].quantile(0.75) - all_composition['write_put_moneyness_mean'].quantile(0.25)),
}

# Implied Vol stats
iv_stats = {
    'buy_call': (all_composition['buy_call_iv_mean'].median(),
                 all_composition['buy_call_iv_mean'].quantile(0.75) - all_composition['buy_call_iv_mean'].quantile(0.25)),
    'buy_put': (all_composition['buy_put_iv_mean'].median(),
                all_composition['buy_put_iv_mean'].quantile(0.75) - all_composition['buy_put_iv_mean'].quantile(0.25)),
    'write_call': (all_composition['write_call_iv_mean'].median(),
                   all_composition['write_call_iv_mean'].quantile(0.75) - all_composition['write_call_iv_mean'].quantile(0.25)),
    'write_put': (all_composition['write_put_iv_mean'].median(),
                  all_composition['write_put_iv_mean'].quantile(0.75) - all_composition['write_put_iv_mean'].quantile(0.25)),
}

# ===========================================================================
# PROJECTED DISTRIBUTION (MEDIAN and IQR across months)
# ===========================================================================

passive_mean_proj = all_data_df['proj_passive_mean'].median()

passive_mean_proj_iqr = all_data_df['proj_passive_mean'].quantile(0.75) - all_data_df['proj_passive_mean'].quantile(0.25)

passive_std_proj = all_data_df['proj_passive_std'].median()
passive_std_proj_iqr = all_data_df['proj_passive_std'].quantile(0.75) - all_data_df['proj_passive_std'].quantile(0.25)

passive_skew_proj = all_data_df['proj_passive_skew'].median()
passive_skew_proj_iqr = all_data_df['proj_passive_skew'].quantile(0.75) - all_data_df['proj_passive_skew'].quantile(0.25)

passive_cvar_proj = all_data_df['proj_passive_cvar'].median()
passive_cvar_proj_iqr = all_data_df['proj_passive_cvar'].quantile(0.75) - all_data_df['proj_passive_cvar'].quantile(0.25)


premium_inc = all_data_df['premium_percentage'].median()
premium_inc_iqr = all_data_df['premium_percentage'].quantile(0.75) - all_data_df['premium_percentage'].quantile(0.25)

active_mean_proj = all_data_df['proj_active_mean'].median()
active_mean_proj_iqr = all_data_df['proj_active_mean'].quantile(0.75) - all_data_df['proj_active_mean'].quantile(0.25)

active_std_proj = all_data_df['proj_active_std'].median()
active_std_proj_iqr = all_data_df['proj_active_std'].quantile(0.75) - all_data_df['proj_active_std'].quantile(0.25)

active_skew_proj = all_data_df['proj_active_skew'].median()
active_skew_proj_iqr = all_data_df['proj_active_skew'].quantile(0.75) - all_data_df['proj_active_skew'].quantile(0.25)

active_cvar_proj = all_data_df['proj_active_cvar'].median()
active_cvar_proj_iqr = all_data_df['proj_active_cvar'].quantile(0.75) - all_data_df['proj_active_cvar'].quantile(0.25)
print(active_cvar_proj)


# ===========================================================================
# ADDITIONAL TABLES — BY VIX TERCILES
# Mean, Std, Skew (using atoms) with Median & IQR
# ===========================================================================

print("\n" + "="*80)
print("VIX TERCILE SUB-TABLES (MEAN, STD, SKEW — MEDIAN & IQR)")
print("="*80 + "\n")

# 1) Assign terciles
performance_df['VIX_tercile'] = pd.qcut(
    performance_df['VIX'],
    q=3,
    labels=['Low VIX', 'Mid VIX', 'High VIX']
)

# Helper
def med_iqr(series):
    med = series.median()
    iqr = series.quantile(0.75) - series.quantile(0.25)
    return med, iqr

# 2) Build results for each tercile
tercile_tables = {}

for label in ['Low VIX', 'Mid VIX', 'High VIX']:
    group = performance_df[performance_df['VIX_tercile'] == label]

    # PASSIVE statistics
    pm, pm_iqr = med_iqr(group['proj_passive_mean'])
    ps, ps_iqr = med_iqr(group['proj_passive_std'])
    pk, pk_iqr = med_iqr(group['proj_passive_skew'])

    # ACTIVE statistics
    am, am_iqr = med_iqr(group['proj_active_mean'])
    asd, asd_iqr = med_iqr(group['proj_active_std'])
    ak, ak_iqr = med_iqr(group['proj_active_skew'])

    tercile_tables[label] = pd.DataFrame({
        'Metric': [
            'Passive Mean', 'Passive Std', 'Passive Skew',
            'Active Mean', 'Active Std', 'Active Skew'
        ],
        'Median': [
            pm, ps, pk,
            am, asd, ak
        ],
        'IQR': [
            pm_iqr, ps_iqr, pk_iqr,
            am_iqr, asd_iqr, ak_iqr
        ]
    })

# 3) Print each table
for label in ['Low VIX', 'Mid VIX', 'High VIX']:
    print("\n" + "-"*60)
    print(f"{label.upper()} REGIME")
    print("-"*60)
    print(tercile_tables[label].to_string(index=False))
    print()

# ===========================================================================
# REALIZED PERFORMANCE (Out-of-sample)
# CORRECT: Mean FIRST, then annualize
# CORRECT: Active MINUS Passive (not the other way around)
# ===========================================================================

if len(oos_data) > 0:
    buy_call_mean_monthly = oos_data['buy_call_return'].mean()
    buy_put_mean_monthly = oos_data['buy_put_return'].mean()
    write_call_mean_monthly = oos_data['write_call_return'].mean()
    write_put_mean_monthly = oos_data['write_put_return'].mean()
    
    buy_call_mean_annual = annualize_return(buy_call_mean_monthly)
    buy_put_mean_annual = annualize_return(buy_put_mean_monthly)
    write_call_mean_annual = annualize_return(write_call_mean_monthly)
    write_put_mean_annual = annualize_return(write_put_mean_monthly)
    
    buy_call_std = oos_data['buy_call_return'].std()
    buy_put_std = oos_data['buy_put_return'].std()
    write_call_std = oos_data['write_call_return'].std()
    write_put_std = oos_data['write_put_return'].std()


    # Step 1: Calculate MONTHLY means
    passive_mean_monthly = oos_data['realized_index_return'].mean()
    active_mean_monthly = oos_data['realized_enhanced_return'].mean()
    
    # Step 2: Annualize the means
    passive_mean_real = annualize_return(passive_mean_monthly)
    active_mean_real = annualize_return(active_mean_monthly)
    
    # Step 3: Calculate volatilities (monthly std, then annualize)
    passive_std_real = annualize_vol(oos_data['realized_index_return'].std())
    active_std_real = annualize_vol(oos_data['realized_enhanced_return'].std())
    
    # ===========================================================================
    # PERFORMANCE IMPROVEMENTS (CORRECT: Active - Passive)
    # ===========================================================================
    
    mean_improvement = active_mean_real - passive_mean_real
    # Calculate CER p-values
    passive_rets = oos_data['realized_index_return'].values
    active_rets = oos_data['realized_enhanced_return'].values
    cer_results = calculate_cer_pvalues(passive_rets, active_rets, gamma_values=[1, 4, 10])
    p_cer1 = cer_results[1]['p_value']
    p_cer4 = cer_results[4]['p_value']
    p_cer10 = cer_results[10]['p_value']
    print(p_cer1)
    print(p_cer4)
    print(p_cer10)
    # CERs calculated on annualized values
    cer1_passive = calculate_cer_from_mean_std(passive_mean_real, passive_std_real, 1)
    cer1_active = calculate_cer_from_mean_std(active_mean_real, active_std_real, 1)
    cer1_improvement = cer1_active - cer1_passive
    
    cer4_passive = calculate_cer_from_mean_std(passive_mean_real, passive_std_real, 4)
    cer4_active = calculate_cer_from_mean_std(active_mean_real, active_std_real, 4)
    cer4_improvement = cer4_active - cer4_passive
    
    cer10_passive = calculate_cer_from_mean_std(passive_mean_real, passive_std_real, 10)
    cer10_active = calculate_cer_from_mean_std(active_mean_real, active_std_real, 10)
    cer10_improvement = cer10_active - cer10_passive
    
    # P-values from paired t-test
    from scipy.stats import ttest_rel
    _, p_mean = ttest_rel(oos_data['realized_enhanced_return'], oos_data['realized_index_return'])
    
else:
    passive_mean_real = np.nan
    active_mean_real = np.nan
    mean_improvement = np.nan
    cer1_improvement = np.nan
    cer4_improvement = np.nan
    cer10_improvement = np.nan
    p_mean = np.nan
    buy_call_mean_annual = np.nan
    buy_put_mean_annual = np.nan
    write_call_mean_annual = np.nan
    write_put_mean_annual = np.nan
    buy_call_std = np.nan
    buy_put_std = np.nan
    write_call_std = np.nan
    write_put_std = np.nan

# ===========================================================================
# BUILD TABLE (NO ELRT)
# ===========================================================================

table_data = []

def add_row(metric, value, pval_or_iqr=""):
    table_data.append({
        'Metric': metric,
        'Value': value,
        'P-value/IQR': pval_or_iqr
    })

# Header
add_row('VIX', 'All Regimes', '')
add_row('Crisis', 'Included', '')
add_row('', '', '')

# Count
add_row('Count', '', '')
add_row('  Dates', str(n_dates), '')
add_row('  Non-zeros', str(n_nonzeros), '')
add_row('  Breadth', str(breadth_median), f'[{breadth_iqr}]')
add_row('', '', '')

# Composition
add_row('Composition', '', '')
add_row('  BuyCall', f'{comp_stats["buy_call"][0]:.2f}', f'[{comp_stats["buy_call"][1]:.2f}]')
add_row('  BuyPut', f'{comp_stats["buy_put"][0]:.2f}', f'[{comp_stats["buy_put"][1]:.2f}]')
add_row('  WriteCall', f'{comp_stats["write_call"][0]:.2f}', f'[{comp_stats["write_call"][1]:.2f}]')
add_row('  WritePut', f'{comp_stats["write_put"][0]:.2f}', f'[{comp_stats["write_put"][1]:.2f}]')
add_row('', '', '')

# Moneyness
add_row('Moneyness', '', '')
add_row('  BuyCall', f'{moneyness_stats["buy_call"][0]:.2f}', f'[{moneyness_stats["buy_call"][1]:.2f}]')
add_row('  BuyPut', f'{moneyness_stats["buy_put"][0]:.2f}', f'[{moneyness_stats["buy_put"][1]:.2f}]')
add_row('  WriteCall', f'{moneyness_stats["write_call"][0]:.2f}', f'[{moneyness_stats["write_call"][1]:.2f}]')
add_row('  WritePut', f'{moneyness_stats["write_put"][0]:.2f}', f'[{moneyness_stats["write_put"][1]:.2f}]')
add_row('', '', '')

# Implied Vol
add_row('Implied Vol.', '', '')
add_row('  BuyCall', f'{iv_stats["buy_call"][0]:.2f}', f'[{iv_stats["buy_call"][1]:.2f}]')
add_row('  BuyPut', f'{iv_stats["buy_put"][0]:.2f}', f'[{iv_stats["buy_put"][1]:.2f}]')
add_row('  WriteCall', f'{iv_stats["write_call"][0]:.2f}', f'[{iv_stats["write_call"][1]:.2f}]')
add_row('  WritePut', f'{iv_stats["write_put"][0]:.2f}', f'[{iv_stats["write_put"][1]:.2f}]')
add_row('', '', '')

import math

# Projected Distribution (ANNUALIZED %)
# Projected Distribution with proper annualization and IQR
add_row('Projected Distribution', '', '')
add_row('  Passive Mean', f'{annualize_return(passive_mean_proj)*100:.2f}', 
        f'[{annualize_return(passive_mean_proj_iqr)*100:.2f}]')
add_row('  Passive Std', f'{annualize_vol(passive_std_proj)*100:.2f}', 
        f'[{annualize_vol(passive_std_proj_iqr)*100:.2f}]')
add_row('  Passive Skew', f'{passive_skew_proj:.2f}', 
        f'[{passive_skew_proj_iqr:.2f}]')
add_row('  Passive CVaR', f'{passive_cvar_proj*100*math.sqrt(12):.2f}', 
        f'[{passive_cvar_proj_iqr*100*math.sqrt(12):.2f}]')
#add_row('  Passive CVaR', f'{passive_cvar_proj*math.sqrt(12)*100:.2f}', '')
add_row('  Premium Inc.', f'{annualize_return(premium_inc)*100:.2f}', 
        f'[{annualize_return(premium_inc_iqr)*100:.2f}]')

add_row('  Active Mean', f'{annualize_return(active_mean_proj)*100:.2f}', 
        f'[{annualize_return(active_mean_proj_iqr)*100:.2f}]')
add_row('  Active Std', f'{annualize_vol(active_std_proj)*100:.2f}', 
        f'[{annualize_vol(active_std_proj_iqr)*100:.2f}]')
add_row('  Active Skew', f'{active_skew_proj:.2f}', 
        f'[{active_skew_proj_iqr:.2f}]')
add_row('  Active CVar', f'{active_cvar_proj*100*math.sqrt(12):.2f}', 
        f'[{active_cvar_proj_iqr*100*math.sqrt(12):.2f}]')
#add_row('  Active CVaR', f'{active_cvar_proj*100*math.sqrt(12):.2f}', '')
add_row('', '', '')


# Realized Performance
add_row('Realized Performance', '', '')
add_row('  Passive Mean', f'{passive_mean_real*100:.2f}%' if not np.isnan(passive_mean_real) else 'N/A', '')
add_row('  Active Mean', f'{active_mean_real*100:.2f}%' if not np.isnan(active_mean_real) else 'N/A', '')
add_row('', '', '')

# Performance Improvements (ANNUALIZED %)
add_row('Performance Improvements', '', '')
add_row('  Mean', f'{mean_improvement*100:.2f}' if not np.isnan(mean_improvement) else 'N/A',
        f'({p_mean:.2f})' if not np.isnan(p_mean) else '')
add_row('  CER1', f'{cer1_improvement*100:.2f}' if not np.isnan(cer1_improvement) else 'N/A', 
        f'({p_cer1:.2f})')
add_row('  CER4', f'{cer4_improvement*100:.2f}' if not np.isnan(cer4_improvement) else 'N/A', 
        f'({p_cer4:.2f})')
add_row('  CER10', f'{cer10_improvement*100:.2f}' if not np.isnan(cer10_improvement) else 'N/A', 
        f'({p_cer10:.2f})')
add_row('', '', '')


# ===========================================================================
# UPDATE THE TABLE SECTION
# ===========================================================================

# Realized Performance
add_row('Realized Performance', '', '')
add_row('  Passive Mean', f'{passive_mean_real*100:.2f}%' if not np.isnan(passive_mean_real) else 'N/A', '')
add_row('  Active Mean', f'{active_mean_real*100:.2f}%' if not np.isnan(active_mean_real) else 'N/A', '')
add_row('  BuyCall Mean', 
        f'{buy_call_mean_annual*100:.2f}' if not np.isnan(buy_call_mean_annual) else 'N/A',
        f'({annualize_vol(buy_call_std)*100:.2f})' if not np.isnan(buy_call_std) else '')
add_row('  BuyPut Mean', 
        f'{buy_put_mean_annual*100:.2f}' if not np.isnan(buy_put_mean_annual) else 'N/A',
        f'({annualize_vol(buy_put_std)*100:.2f})' if not np.isnan(buy_put_std) else '')
add_row('  WriteCall Mean', 
        f'{write_call_mean_annual*100:.2f}' if not np.isnan(write_call_mean_annual) else 'N/A',
        f'({annualize_vol(write_call_std)*100:.2f})' if not np.isnan(write_call_std) else '')
add_row('  WritePut Mean', 
        f'{write_put_mean_annual*100:.2f}' if not np.isnan(write_put_mean_annual) else 'N/A',
        f'({annualize_vol(write_put_std)*100:.2f})' if not np.isnan(write_put_std) else '')
add_row('', '', '')

# NO ELRT CALCULATION
add_row('SSD test', '', '')
add_row('  Proportion with SSD', f'{n_nonzeros/n_dates:.2f}' if n_dates > 0 else 'N/A', '')

# ===========================================================================
# CREATE AND DISPLAY TABLE
# ===========================================================================

results_table = pd.DataFrame(table_data)

# Display
print("\n" + "="*80)
print("FINAL RESULTS TABLE - ALL VIX REGIMES dynamic thetas and c0 c1")
print("="*80 + "\n")
print(results_table.to_string(index=False))



# ===========================================================================
# LOW VIX REGIME ANALYSIS
# ===========================================================================

print("\n" + "="*80)
print("LOW VIX REGIME ANALYSIS")
print("="*80 + "\n")

# Filter to Low VIX ONLY
low_vix_df = performance_df[performance_df['VIX_tercile'] == 'Low VIX'].copy()
low_vix_composition = composition_df[composition_df['id'].isin(low_vix_df['id'])].copy()
low_vix_oos = low_vix_df[~low_vix_df['realized_index_return'].isna()].copy()

print(f"Low VIX dates: {len(low_vix_df)}")
print(f"Low VIX with OOS: {len(low_vix_oos)}\n")

# ===========================================================================
# CALCULATE STATISTICS
# ===========================================================================

n_dates_low = len(low_vix_df)
n_nonzeros_low = (low_vix_df.groupby('id')['expected_outperformance'].sum() != 0).sum()

# Breadth (from low vix data only)
breadth_by_month_low = low_vix_composition.groupby('id').size()
breadth_median_low = int(breadth_by_month_low.median())
breadth_iqr_low = int(breadth_by_month_low.quantile(0.75) - breadth_by_month_low.quantile(0.25))

# VIX stats
vix_mean_low = low_vix_df['VIX'].mean()

# Composition stats - ALL 4 option types
comp_stats_low = {
    'buy_call': (low_vix_composition['buy_call_weight'].median(), 
                 low_vix_composition['buy_call_weight'].quantile(0.75) - low_vix_composition['buy_call_weight'].quantile(0.25)),
    'buy_put': (low_vix_composition['buy_put_weight'].median(),
                low_vix_composition['buy_put_weight'].quantile(0.75) - low_vix_composition['buy_put_weight'].quantile(0.25)),
    'write_call': (low_vix_composition['write_call_weight'].median(),
                   low_vix_composition['write_call_weight'].quantile(0.75) - low_vix_composition['write_call_weight'].quantile(0.25)),
    'write_put': (low_vix_composition['write_put_weight'].median(),
                  low_vix_composition['write_put_weight'].quantile(0.75) - low_vix_composition['write_put_weight'].quantile(0.25)),
}

# Moneyness stats
moneyness_stats_low = {
    'buy_call': (low_vix_composition['buy_call_moneyness_mean'].median(),
                 low_vix_composition['buy_call_moneyness_mean'].quantile(0.75) - low_vix_composition['buy_call_moneyness_mean'].quantile(0.25)),
    'buy_put': (low_vix_composition['buy_put_moneyness_mean'].median(),
                low_vix_composition['buy_put_moneyness_mean'].quantile(0.75) - low_vix_composition['buy_put_moneyness_mean'].quantile(0.25)),
    'write_call': (low_vix_composition['write_call_moneyness_mean'].median(),
                   low_vix_composition['write_call_moneyness_mean'].quantile(0.75) - low_vix_composition['write_call_moneyness_mean'].quantile(0.25)),
    'write_put': (low_vix_composition['write_put_moneyness_mean'].median(),
                  low_vix_composition['write_put_moneyness_mean'].quantile(0.75) - low_vix_composition['write_put_moneyness_mean'].quantile(0.25)),
}

# Implied Vol stats
iv_stats_low = {
    'buy_call': (low_vix_composition['buy_call_iv_mean'].median(),
                 low_vix_composition['buy_call_iv_mean'].quantile(0.75) - low_vix_composition['buy_call_iv_mean'].quantile(0.25)),
    'buy_put': (low_vix_composition['buy_put_iv_mean'].median(),
                low_vix_composition['buy_put_iv_mean'].quantile(0.75) - low_vix_composition['buy_put_iv_mean'].quantile(0.25)),
    'write_call': (low_vix_composition['write_call_iv_mean'].median(),
                   low_vix_composition['write_call_iv_mean'].quantile(0.75) - low_vix_composition['write_call_iv_mean'].quantile(0.25)),
    'write_put': (low_vix_composition['write_put_iv_mean'].median(),
                  low_vix_composition['write_put_iv_mean'].quantile(0.75) - low_vix_composition['write_put_iv_mean'].quantile(0.25)),
}

# ===========================================================================
# PROJECTED DISTRIBUTION (MEDIAN and IQR across months)
# ===========================================================================

passive_mean_proj_low = low_vix_df['proj_passive_mean'].median()
passive_mean_proj_iqr_low = low_vix_df['proj_passive_mean'].quantile(0.75) - low_vix_df['proj_passive_mean'].quantile(0.25)

passive_std_proj_low = low_vix_df['proj_passive_std'].median()
passive_std_proj_iqr_low = low_vix_df['proj_passive_std'].quantile(0.75) - low_vix_df['proj_passive_std'].quantile(0.25)

passive_skew_proj_low = low_vix_df['proj_passive_skew'].median()
passive_skew_proj_iqr_low = low_vix_df['proj_passive_skew'].quantile(0.75) - low_vix_df['proj_passive_skew'].quantile(0.25)

passive_cvar_proj_low = low_vix_df['proj_passive_cvar'].median()
passive_cvar_proj_iqr_low = low_vix_df['proj_passive_cvar'].quantile(0.75) - low_vix_df['proj_passive_cvar'].quantile(0.25)


premium_inc_low = low_vix_df['premium_percentage'].median()
premium_inc_iqr_low = low_vix_df['premium_percentage'].quantile(0.75) - low_vix_df['premium_percentage'].quantile(0.25)


active_mean_proj_low = low_vix_df['proj_active_mean'].median()
active_mean_proj_iqr_low = low_vix_df['proj_active_mean'].quantile(0.75) - low_vix_df['proj_active_mean'].quantile(0.25)

active_std_proj_low = low_vix_df['proj_active_std'].median()
active_std_proj_iqr_low = low_vix_df['proj_active_std'].quantile(0.75) - low_vix_df['proj_active_std'].quantile(0.25)

active_skew_proj_low = low_vix_df['proj_active_skew'].median()
active_skew_proj_iqr_low = low_vix_df['proj_active_skew'].quantile(0.75) - low_vix_df['proj_active_skew'].quantile(0.25)

active_cvar_proj_low = low_vix_df['proj_active_cvar'].median()
active_cvar_proj_iqr_low = low_vix_df['proj_active_cvar'].quantile(0.75) - low_vix_df['proj_active_cvar'].quantile(0.25)

# ===========================================================================
# REALIZED PERFORMANCE (Out-of-sample) - LOW VIX ONLY
# ===========================================================================

if len(low_vix_oos) > 0:
    buy_call_mean_monthly_low = low_vix_oos['buy_call_return'].mean()
    buy_put_mean_monthly_low = low_vix_oos['buy_put_return'].mean()
    write_call_mean_monthly_low = low_vix_oos['write_call_return'].mean()
    write_put_mean_monthly_low = low_vix_oos['write_put_return'].mean()
    
    buy_call_mean_annual_low = annualize_return(buy_call_mean_monthly_low)
    buy_put_mean_annual_low = annualize_return(buy_put_mean_monthly_low)
    write_call_mean_annual_low = annualize_return(write_call_mean_monthly_low)
    write_put_mean_annual_low = annualize_return(write_put_mean_monthly_low)
    
    buy_call_std_low = low_vix_oos['buy_call_return'].std()
    buy_put_std_low = low_vix_oos['buy_put_return'].std()
    write_call_std_low = low_vix_oos['write_call_return'].std()
    write_put_std_low = low_vix_oos['write_put_return'].std()

    # Step 1: Calculate MONTHLY means
    passive_mean_monthly_low = low_vix_oos['realized_index_return'].mean()
    active_mean_monthly_low = low_vix_oos['realized_enhanced_return'].mean()
    
    # Step 2: Annualize the means
    passive_mean_real_low = annualize_return(passive_mean_monthly_low)
    active_mean_real_low = annualize_return(active_mean_monthly_low)
    
    # Step 3: Calculate volatilities (monthly std, then annualize)
    passive_std_real_low = annualize_vol(low_vix_oos['realized_index_return'].std())
    active_std_real_low = annualize_vol(low_vix_oos['realized_enhanced_return'].std())
    
    # ===========================================================================
    # PERFORMANCE IMPROVEMENTS (Active - Passive)
    # ===========================================================================
    
    mean_improvement_low = active_mean_real_low - passive_mean_real_low
    passive_rets_low = low_vix_oos['realized_index_return'].values
    active_rets_low = low_vix_oos['realized_enhanced_return'].values
    cer_results_low = calculate_cer_pvalues(passive_rets_low, active_rets_low, gamma_values=[1, 4, 10])
    p_cer1_low = cer_results_low[1]['p_value']
    p_cer4_low = cer_results_low[4]['p_value']
    p_cer10_low = cer_results_low[10]['p_value']
    print(p_cer1_low)
    print(p_cer4_low)
    print(p_cer10_low)
    
    # CERs calculated on annualized values
    cer1_passive_low = calculate_cer_from_mean_std(passive_mean_real_low, passive_std_real_low, 1)
    cer1_active_low = calculate_cer_from_mean_std(active_mean_real_low, active_std_real_low, 1)
    cer1_improvement_low = cer1_active_low - cer1_passive_low
    
    cer4_passive_low = calculate_cer_from_mean_std(passive_mean_real_low, passive_std_real_low, 4)
    cer4_active_low = calculate_cer_from_mean_std(active_mean_real_low, active_std_real_low, 4)
    cer4_improvement_low = cer4_active_low - cer4_passive_low
    
    cer10_passive_low = calculate_cer_from_mean_std(passive_mean_real_low, passive_std_real_low, 10)
    cer10_active_low = calculate_cer_from_mean_std(active_mean_real_low, active_std_real_low, 10)
    cer10_improvement_low = cer10_active_low - cer10_passive_low
    
    # P-values from paired t-test
    from scipy.stats import ttest_rel
    _, p_mean_low = ttest_rel(low_vix_oos['realized_enhanced_return'], low_vix_oos['realized_index_return'])
    
else:
    passive_mean_real_low = np.nan
    active_mean_real_low = np.nan
    mean_improvement_low = np.nan
    cer1_improvement_low = np.nan
    cer4_improvement_low = np.nan
    cer10_improvement_low = np.nan
    p_mean_low = np.nan
    buy_call_mean_annual_low = np.nan
    buy_put_mean_annual_low = np.nan
    write_call_mean_annual_low = np.nan
    write_put_mean_annual_low = np.nan
    buy_call_std_low = np.nan
    buy_put_std_low = np.nan
    write_call_std_low = np.nan
    write_put_std_low = np.nan

# ===========================================================================
# BUILD TABLE FOR LOW VIX
# ===========================================================================

import math

table_data_low = []

def add_row(metric, value, pval_or_iqr=""):
    table_data_low.append({
        'Metric': metric,
        'Value': value,
        'P-value/IQR': pval_or_iqr
    })

# Header
add_row('VIX', 'Low VIX Tercile', '')
add_row('VIX Mean', f'{vix_mean_low:.2f}', '')
add_row('', '', '')

# Count
add_row('Count', '', '')
add_row('  Dates', str(n_dates_low), '')
add_row('  Non-zeros', str(n_nonzeros_low), '')
add_row('  Breadth', str(breadth_median_low), f'[{breadth_iqr_low}]')
add_row('', '', '')

# Composition
add_row('Composition', '', '')
add_row('  BuyCall', f'{comp_stats_low["buy_call"][0]:.2f}', f'[{comp_stats_low["buy_call"][1]:.2f}]')
add_row('  BuyPut', f'{comp_stats_low["buy_put"][0]:.2f}', f'[{comp_stats_low["buy_put"][1]:.2f}]')
add_row('  WriteCall', f'{comp_stats_low["write_call"][0]:.2f}', f'[{comp_stats_low["write_call"][1]:.2f}]')
add_row('  WritePut', f'{comp_stats_low["write_put"][0]:.2f}', f'[{comp_stats_low["write_put"][1]:.2f}]')
add_row('', '', '')

# Moneyness
add_row('Moneyness', '', '')
add_row('  BuyCall', f'{moneyness_stats_low["buy_call"][0]:.2f}', f'[{moneyness_stats_low["buy_call"][1]:.2f}]')
add_row('  BuyPut', f'{moneyness_stats_low["buy_put"][0]:.2f}', f'[{moneyness_stats_low["buy_put"][1]:.2f}]')
add_row('  WriteCall', f'{moneyness_stats_low["write_call"][0]:.2f}', f'[{moneyness_stats_low["write_call"][1]:.2f}]')
add_row('  WritePut', f'{moneyness_stats_low["write_put"][0]:.2f}', f'[{moneyness_stats_low["write_put"][1]:.2f}]')
add_row('', '', '')

# Implied Vol
add_row('Implied Vol.', '', '')
add_row('  BuyCall', f'{iv_stats_low["buy_call"][0]:.2f}', f'[{iv_stats_low["buy_call"][1]:.2f}]')
add_row('  BuyPut', f'{iv_stats_low["buy_put"][0]:.2f}', f'[{iv_stats_low["buy_put"][1]:.2f}]')
add_row('  WriteCall', f'{iv_stats_low["write_call"][0]:.2f}', f'[{iv_stats_low["write_call"][1]:.2f}]')
add_row('  WritePut', f'{iv_stats_low["write_put"][0]:.2f}', f'[{iv_stats_low["write_put"][1]:.2f}]')
add_row('', '', '')

# Projected Distribution (ANNUALIZED %)
add_row('Projected Distribution', '', '')
add_row('  Passive Mean', f'{annualize_return(passive_mean_proj_low)*100:.2f}', 
        f'[{annualize_return(passive_mean_proj_iqr_low)*100:.2f}]')
add_row('  Passive Std', f'{annualize_vol(passive_std_proj_low)*100:.2f}', 
        f'[{annualize_vol(passive_std_proj_iqr_low)*100:.2f}]')
add_row('  Passive Skew', f'{passive_skew_proj_low:.2f}', 
        f'[{passive_skew_proj_iqr_low:.2f}]')
add_row('  Passive CVaR', f'{passive_cvar_proj_low*math.sqrt(12)*100:.2f}', '')
#add_row('  Premium Inc.', f'{annualize_return(premium_inc_low)*100:.2f}', '')
add_row('  Premium Inc.', f'{annualize_return(premium_inc_low)*100:.2f}', 
        f'[{annualize_return(premium_inc_iqr_low)*100:.2f}]')
add_row('  Active Mean', f'{annualize_return(active_mean_proj_low)*100:.2f}', 
        f'[{annualize_return(active_mean_proj_iqr_low)*100:.2f}]')
add_row('  Active Std', f'{annualize_vol(active_std_proj_low)*100:.2f}', 
        f'[{annualize_vol(active_std_proj_iqr_low)*100:.2f}]')
add_row('  Active Skew', f'{active_skew_proj_low:.2f}', 
        f'[{active_skew_proj_iqr_low:.2f}]')
add_row('  Active CVaR', f'{active_cvar_proj_low*100*math.sqrt(12):.2f}', '')
add_row('', '', '')

# Realized Performance
add_row('Realized Performance', '', '')
add_row('  Passive Mean', f'{passive_mean_real_low*100:.2f}%' if not np.isnan(passive_mean_real_low) else 'N/A', '')
add_row('  Active Mean', f'{active_mean_real_low*100:.2f}%' if not np.isnan(active_mean_real_low) else 'N/A', '')
add_row('  BuyCall Mean', 
        f'{buy_call_mean_annual_low*100:.2f}' if not np.isnan(buy_call_mean_annual_low) else 'N/A',
        f'({annualize_vol(buy_call_std_low)*100:.2f})' if not np.isnan(buy_call_std_low) else '')
add_row('  BuyPut Mean', 
        f'{buy_put_mean_annual_low*100:.2f}' if not np.isnan(buy_put_mean_annual_low) else 'N/A',
        f'({annualize_vol(buy_put_std_low)*100:.2f})' if not np.isnan(buy_put_std_low) else '')
add_row('  WriteCall Mean', 
        f'{write_call_mean_annual_low*100:.2f}' if not np.isnan(write_call_mean_annual_low) else 'N/A',
        f'({annualize_vol(write_call_std_low)*100:.2f})' if not np.isnan(write_call_std_low) else '')
add_row('  WritePut Mean', 
        f'{write_put_mean_annual_low*100:.2f}' if not np.isnan(write_put_mean_annual_low) else 'N/A',
        f'({annualize_vol(write_put_std_low)*100:.2f})' if not np.isnan(write_put_std_low) else '')
add_row('', '', '')

# Performance Improvements (ANNUALIZED %)
add_row('Performance Improvements', '', '')
add_row('  Mean', f'{mean_improvement_low*100:.2f}' if not np.isnan(mean_improvement_low) else 'N/A',
        f'({p_mean_low:.3f})' if not np.isnan(p_mean_low) else '')
add_row('  CER1', f'{cer1_improvement_low*100:.2f}' if not np.isnan(cer1_improvement_low) else 'N/A', 
        f'({p_cer1_low:.3f})')
add_row('  CER4', f'{cer4_improvement_low*100:.2f}' if not np.isnan(cer4_improvement_low) else 'N/A', 
        f'({p_cer4_low:.3f})')
add_row('  CER10', f'{cer10_improvement_low*100:.2f}' if not np.isnan(cer10_improvement_low) else 'N/A', 
        f'({p_cer10_low:.3f})')
add_row('', '', '')

# SSD test
add_row('SSD test', '', '')
add_row('  Proportion with SSD', f'{n_nonzeros_low/n_dates_low:.2f}' if n_dates_low > 0 else 'N/A', '')

# ===========================================================================
# CREATE AND DISPLAY TABLE
# ===========================================================================

results_table_low = pd.DataFrame(table_data_low)

# Display
print("\n" + "="*80)
print("FINAL RESULTS TABLE - LOW VIX REGIME")
print("="*80 + "\n")
#print(results_table_low.to_string(index=False))




# ===========================================================================
# MEDIUM VIX REGIME ANALYSIS
# ===========================================================================

print("\n" + "="*80)
print("MEDIUM VIX REGIME ANALYSIS")
print("="*80 + "\n")

# Filter to Mid VIX ONLY
mid_vix_df = performance_df[performance_df['VIX_tercile'] == 'Mid VIX'].copy()
mid_vix_composition = composition_df[composition_df['id'].isin(mid_vix_df['id'])].copy()
mid_vix_oos = mid_vix_df[~mid_vix_df['realized_index_return'].isna()].copy()

print(f"Mid VIX dates: {len(mid_vix_df)}")
print(f"Mid VIX with OOS: {len(mid_vix_oos)}\n")

# ===========================================================================
# CALCULATE STATISTICS
# ===========================================================================

n_dates_mid = len(mid_vix_df)
n_nonzeros_mid = (mid_vix_df.groupby('id')['expected_outperformance'].sum() != 0).sum()


# Breadth (from mid vix data only)
breadth_by_month_mid = mid_vix_composition.groupby('id').size()
breadth_median_mid = int(breadth_by_month_mid.median())
breadth_iqr_mid = int(breadth_by_month_mid.quantile(0.75) - breadth_by_month_mid.quantile(0.25))

# VIX stats
vix_mean_mid = mid_vix_df['VIX'].mean()

# Composition stats - ALL 4 option types
comp_stats_mid = {
    'buy_call': (mid_vix_composition['buy_call_weight'].median(), 
                 mid_vix_composition['buy_call_weight'].quantile(0.75) - mid_vix_composition['buy_call_weight'].quantile(0.25)),
    'buy_put': (mid_vix_composition['buy_put_weight'].median(),
                mid_vix_composition['buy_put_weight'].quantile(0.75) - mid_vix_composition['buy_put_weight'].quantile(0.25)),
    'write_call': (mid_vix_composition['write_call_weight'].median(),
                   mid_vix_composition['write_call_weight'].quantile(0.75) - mid_vix_composition['write_call_weight'].quantile(0.25)),
    'write_put': (mid_vix_composition['write_put_weight'].median(),
                  mid_vix_composition['write_put_weight'].quantile(0.75) - mid_vix_composition['write_put_weight'].quantile(0.25)),
}

# Moneyness stats
moneyness_stats_mid = {
    'buy_call': (mid_vix_composition['buy_call_moneyness_mean'].median(),
                 mid_vix_composition['buy_call_moneyness_mean'].quantile(0.75) - mid_vix_composition['buy_call_moneyness_mean'].quantile(0.25)),
    'buy_put': (mid_vix_composition['buy_put_moneyness_mean'].median(),
                mid_vix_composition['buy_put_moneyness_mean'].quantile(0.75) - mid_vix_composition['buy_put_moneyness_mean'].quantile(0.25)),
    'write_call': (mid_vix_composition['write_call_moneyness_mean'].median(),
                   mid_vix_composition['write_call_moneyness_mean'].quantile(0.75) - mid_vix_composition['write_call_moneyness_mean'].quantile(0.25)),
    'write_put': (mid_vix_composition['write_put_moneyness_mean'].median(),
                  mid_vix_composition['write_put_moneyness_mean'].quantile(0.75) - mid_vix_composition['write_put_moneyness_mean'].quantile(0.25)),
}

# Implied Vol stats
iv_stats_mid = {
    'buy_call': (mid_vix_composition['buy_call_iv_mean'].median(),
                 mid_vix_composition['buy_call_iv_mean'].quantile(0.75) - mid_vix_composition['buy_call_iv_mean'].quantile(0.25)),
    'buy_put': (mid_vix_composition['buy_put_iv_mean'].median(),
                mid_vix_composition['buy_put_iv_mean'].quantile(0.75) - mid_vix_composition['buy_put_iv_mean'].quantile(0.25)),
    'write_call': (mid_vix_composition['write_call_iv_mean'].median(),
                   mid_vix_composition['write_call_iv_mean'].quantile(0.75) - mid_vix_composition['write_call_iv_mean'].quantile(0.25)),
    'write_put': (mid_vix_composition['write_put_iv_mean'].median(),
                  mid_vix_composition['write_put_iv_mean'].quantile(0.75) - mid_vix_composition['write_put_iv_mean'].quantile(0.25)),
}

# ===========================================================================
# PROJECTED DISTRIBUTION (MEDIAN and IQR across months)
# ===========================================================================

passive_mean_proj_mid = mid_vix_df['proj_passive_mean'].median()
passive_mean_proj_iqr_mid = mid_vix_df['proj_passive_mean'].quantile(0.75) - mid_vix_df['proj_passive_mean'].quantile(0.25)

passive_std_proj_mid = mid_vix_df['proj_passive_std'].median()
passive_std_proj_iqr_mid = mid_vix_df['proj_passive_std'].quantile(0.75) - mid_vix_df['proj_passive_std'].quantile(0.25)

passive_skew_proj_mid = mid_vix_df['proj_passive_skew'].median()
passive_skew_proj_iqr_mid = mid_vix_df['proj_passive_skew'].quantile(0.75) - mid_vix_df['proj_passive_skew'].quantile(0.25)

passive_cvar_proj_mid = mid_vix_df['proj_passive_cvar'].median()
passive_cvar_proj_iqr_mid = mid_vix_df['proj_passive_cvar'].quantile(0.75) - mid_vix_df['proj_passive_cvar'].quantile(0.25)


premium_inc_mid = mid_vix_df['premium_percentage'].median()
premium_inc_iqr_mid = mid_vix_df['premium_percentage'].quantile(0.75) - mid_vix_df['premium_percentage'].quantile(0.25)


active_mean_proj_mid = mid_vix_df['proj_active_mean'].median()
active_mean_proj_iqr_mid = mid_vix_df['proj_active_mean'].quantile(0.75) - mid_vix_df['proj_active_mean'].quantile(0.25)

active_std_proj_mid = mid_vix_df['proj_active_std'].median()
active_std_proj_iqr_mid = mid_vix_df['proj_active_std'].quantile(0.75) - mid_vix_df['proj_active_std'].quantile(0.25)

active_skew_proj_mid = mid_vix_df['proj_active_skew'].median()
active_skew_proj_iqr_mid = mid_vix_df['proj_active_skew'].quantile(0.75) - mid_vix_df['proj_active_skew'].quantile(0.25)

active_cvar_proj_mid = mid_vix_df['proj_active_cvar'].median()
active_cvar_proj_iqr_mid = mid_vix_df['proj_active_cvar'].quantile(0.75) - mid_vix_df['proj_active_cvar'].quantile(0.25)


# ===========================================================================
# REALIZED PERFORMANCE (Out-of-sample) - MID VIX ONLY
# ===========================================================================

if len(mid_vix_oos) > 0:
    buy_call_mean_monthly_mid = mid_vix_oos['buy_call_return'].mean()
    buy_put_mean_monthly_mid = mid_vix_oos['buy_put_return'].mean()
    write_call_mean_monthly_mid = mid_vix_oos['write_call_return'].mean()
    write_put_mean_monthly_mid = mid_vix_oos['write_put_return'].mean()
    
    buy_call_mean_annual_mid = annualize_return(buy_call_mean_monthly_mid)
    buy_put_mean_annual_mid = annualize_return(buy_put_mean_monthly_mid)
    write_call_mean_annual_mid = annualize_return(write_call_mean_monthly_mid)
    write_put_mean_annual_mid = annualize_return(write_put_mean_monthly_mid)
    
    buy_call_std_mid = mid_vix_oos['buy_call_return'].std()
    buy_put_std_mid = mid_vix_oos['buy_put_return'].std()
    write_call_std_mid = mid_vix_oos['write_call_return'].std()
    write_put_std_mid = mid_vix_oos['write_put_return'].std()

    # Step 1: Calculate MONTHLY means
    passive_mean_monthly_mid = mid_vix_oos['realized_index_return'].mean()
    active_mean_monthly_mid = mid_vix_oos['realized_enhanced_return'].mean()
    
    # Step 2: Annualize the means
    passive_mean_real_mid = annualize_return(passive_mean_monthly_mid)
    active_mean_real_mid = annualize_return(active_mean_monthly_mid)
    
    # Step 3: Calculate volatilities (monthly std, then annualize)
    passive_std_real_mid = annualize_vol(mid_vix_oos['realized_index_return'].std())
    active_std_real_mid = annualize_vol(mid_vix_oos['realized_enhanced_return'].std())
    
    # ===========================================================================
    # PERFORMANCE IMPROVEMENTS (Active - Passive)
    # ===========================================================================
    
    mean_improvement_mid = active_mean_real_mid - passive_mean_real_mid
    passive_rets_mid = mid_vix_oos['realized_index_return'].values
    active_rets_mid = mid_vix_oos['realized_enhanced_return'].values
    cer_results_mid = calculate_cer_pvalues(passive_rets_mid, active_rets_mid, gamma_values=[1, 4, 10])
    p_cer1_mid = cer_results_mid[1]['p_value']
    p_cer4_mid = cer_results_mid[4]['p_value']
    p_cer10_mid = cer_results_mid[10]['p_value']
    print(p_cer1_mid)
    print(p_cer4_mid)
    print(p_cer10_mid)
    
    # CERs calculated on annualized values
    cer1_passive_mid = calculate_cer_from_mean_std(passive_mean_real_mid, passive_std_real_mid, 1)
    cer1_active_mid = calculate_cer_from_mean_std(active_mean_real_mid, active_std_real_mid, 1)
    cer1_improvement_mid = cer1_active_mid - cer1_passive_mid
    
    cer4_passive_mid = calculate_cer_from_mean_std(passive_mean_real_mid, passive_std_real_mid, 4)
    cer4_active_mid = calculate_cer_from_mean_std(active_mean_real_mid, active_std_real_mid, 4)
    cer4_improvement_mid = cer4_active_mid - cer4_passive_mid
    
    cer10_passive_mid = calculate_cer_from_mean_std(passive_mean_real_mid, passive_std_real_mid, 10)
    cer10_active_mid = calculate_cer_from_mean_std(active_mean_real_mid, active_std_real_mid, 10)
    cer10_improvement_mid = cer10_active_mid - cer10_passive_mid
    
    # P-values from paired t-test
    from scipy.stats import ttest_rel
    _, p_mean_mid = ttest_rel(mid_vix_oos['realized_enhanced_return'], mid_vix_oos['realized_index_return'])
    
else:
    passive_mean_real_mid = np.nan
    active_mean_real_mid = np.nan
    mean_improvement_mid = np.nan
    cer1_improvement_mid = np.nan
    cer4_improvement_mid = np.nan
    cer10_improvement_mid = np.nan
    p_mean_mid = np.nan
    buy_call_mean_annual_mid = np.nan
    buy_put_mean_annual_mid = np.nan
    write_call_mean_annual_mid = np.nan
    write_put_mean_annual_mid = np.nan
    buy_call_std_mid = np.nan
    buy_put_std_mid = np.nan
    write_call_std_mid = np.nan
    write_put_std_mid = np.nan

# ===========================================================================
# BUILD TABLE FOR MID VIX
# ===========================================================================

import math

table_data_mid = []

def add_row(metric, value, pval_or_iqr=""):
    table_data_mid.append({
        'Metric': metric,
        'Value': value,
        'P-value/IQR': pval_or_iqr
    })

# Header
add_row('VIX', 'Mid VIX Tercile', '')
add_row('VIX Mean', f'{vix_mean_mid:.2f}', '')
add_row('', '', '')

# Count
add_row('Count', '', '')
add_row('  Dates', str(n_dates_mid), '')
add_row('  Non-zeros', str(n_nonzeros_mid), '')
add_row('  Breadth', str(breadth_median_mid), f'[{breadth_iqr_mid}]')
add_row('', '', '')

# Composition
add_row('Composition', '', '')
add_row('  BuyCall', f'{comp_stats_mid["buy_call"][0]:.2f}', f'[{comp_stats_mid["buy_call"][1]:.2f}]')
add_row('  BuyPut', f'{comp_stats_mid["buy_put"][0]:.2f}', f'[{comp_stats_mid["buy_put"][1]:.2f}]')
add_row('  WriteCall', f'{comp_stats_mid["write_call"][0]:.2f}', f'[{comp_stats_mid["write_call"][1]:.2f}]')
add_row('  WritePut', f'{comp_stats_mid["write_put"][0]:.2f}', f'[{comp_stats_mid["write_put"][1]:.2f}]')
add_row('', '', '')

# Moneyness
add_row('Moneyness', '', '')
add_row('  BuyCall', f'{moneyness_stats_mid["buy_call"][0]:.2f}', f'[{moneyness_stats_mid["buy_call"][1]:.2f}]')
add_row('  BuyPut', f'{moneyness_stats_mid["buy_put"][0]:.2f}', f'[{moneyness_stats_mid["buy_put"][1]:.2f}]')
add_row('  WriteCall', f'{moneyness_stats_mid["write_call"][0]:.2f}', f'[{moneyness_stats_mid["write_call"][1]:.2f}]')
add_row('  WritePut', f'{moneyness_stats_mid["write_put"][0]:.2f}', f'[{moneyness_stats_mid["write_put"][1]:.2f}]')
add_row('', '', '')

# Implied Vol
add_row('Implied Vol.', '', '')
add_row('  BuyCall', f'{iv_stats_mid["buy_call"][0]:.2f}', f'[{iv_stats_mid["buy_call"][1]:.2f}]')
add_row('  BuyPut', f'{iv_stats_mid["buy_put"][0]:.2f}', f'[{iv_stats_mid["buy_put"][1]:.2f}]')
add_row('  WriteCall', f'{iv_stats_mid["write_call"][0]:.2f}', f'[{iv_stats_mid["write_call"][1]:.2f}]')
add_row('  WritePut', f'{iv_stats_mid["write_put"][0]:.2f}', f'[{iv_stats_mid["write_put"][1]:.2f}]')
add_row('', '', '')

# Projected Distribution (ANNUALIZED %)
add_row('Projected Distribution', '', '')
add_row('  Passive Mean', f'{annualize_return(passive_mean_proj_mid)*100:.2f}', 
        f'[{annualize_return(passive_mean_proj_iqr_mid)*100:.2f}]')
add_row('  Passive Std', f'{annualize_vol(passive_std_proj_mid)*100:.2f}', 
        f'[{annualize_vol(passive_std_proj_iqr_mid)*100:.2f}]')
add_row('  Passive Skew', f'{passive_skew_proj_mid:.2f}', 
        f'[{passive_skew_proj_iqr_mid:.2f}]')
add_row('  Passive CVaR', f'{passive_cvar_proj_mid*math.sqrt(12)*100:.2f}', '')
#add_row('  Premium Inc.', f'{annualize_return(premium_inc_mid)*100:.2f}', '')

add_row('  Premium Inc.', f'{annualize_return(premium_inc_mid)*100:.2f}', 
        f'[{annualize_return(premium_inc_iqr_mid)*100:.2f}]')

add_row('  Active Mean', f'{annualize_return(active_mean_proj_mid)*100:.2f}', 
        f'[{annualize_return(active_mean_proj_iqr_mid)*100:.2f}]')
add_row('  Active Std', f'{annualize_vol(active_std_proj_mid)*100:.2f}', 
        f'[{annualize_vol(active_std_proj_iqr_mid)*100:.2f}]')
add_row('  Active Skew', f'{active_skew_proj_mid:.2f}', 
        f'[{active_skew_proj_iqr_mid:.2f}]')
add_row('  Active CVaR', f'{active_cvar_proj_mid*100*math.sqrt(12):.2f}', '')
add_row('', '', '')

# Realized Performance
add_row('Realized Performance', '', '')
add_row('  Passive Mean', f'{passive_mean_real_mid*100:.2f}%' if not np.isnan(passive_mean_real_mid) else 'N/A', '')
add_row('  Active Mean', f'{active_mean_real_mid*100:.2f}%' if not np.isnan(active_mean_real_mid) else 'N/A', '')
add_row('  BuyCall Mean', 
        f'{buy_call_mean_annual_mid*100:.2f}' if not np.isnan(buy_call_mean_annual_mid) else 'N/A',
        f'({annualize_vol(buy_call_std_mid)*100:.2f})' if not np.isnan(buy_call_std_mid) else '')
add_row('  BuyPut Mean', 
        f'{buy_put_mean_annual_mid*100:.2f}' if not np.isnan(buy_put_mean_annual_mid) else 'N/A',
        f'({annualize_vol(buy_put_std_mid)*100:.2f})' if not np.isnan(buy_put_std_mid) else '')
add_row('  WriteCall Mean', 
        f'{write_call_mean_annual_mid*100:.2f}' if not np.isnan(write_call_mean_annual_mid) else 'N/A',
        f'({annualize_vol(write_call_std_mid)*100:.2f})' if not np.isnan(write_call_std_mid) else '')
add_row('  WritePut Mean', 
        f'{write_put_mean_annual_mid*100:.2f}' if not np.isnan(write_put_mean_annual_mid) else 'N/A',
        f'({annualize_vol(write_put_std_mid)*100:.2f})' if not np.isnan(write_put_std_mid) else '')
add_row('', '', '')

# Performance Improvements (ANNUALIZED %)
add_row('Performance Improvements', '', '')
add_row('  Mean', f'{mean_improvement_mid*100:.2f}' if not np.isnan(mean_improvement_mid) else 'N/A',
        f'({p_mean_mid:.2f})' if not np.isnan(p_mean_mid) else '')
add_row('  CER1', f'{cer1_improvement_mid*100:.2f}' if not np.isnan(cer1_improvement_mid) else 'N/A', 
       f'({p_cer1_mid:.2f})')
add_row('  CER4', f'{cer4_improvement_mid*100:.2f}' if not np.isnan(cer4_improvement_mid) else 'N/A', 
        f'({p_cer4_mid:.2f})')
add_row('  CER10', f'{cer10_improvement_mid*100:.2f}' if not np.isnan(cer10_improvement_mid) else 'N/A', 
        f'({p_cer10_mid:.2f})')
add_row('', '', '')

# SSD test
add_row('SSD test', '', '')
add_row('  Proportion with SSD', f'{n_nonzeros_mid/n_dates_mid:.2f}' if n_dates_mid > 0 else 'N/A', '')

# ===========================================================================
# CREATE AND DISPLAY TABLE
# ===========================================================================

results_table_mid = pd.DataFrame(table_data_mid)

# Display
print("\n" + "="*80)
print("FINAL RESULTS TABLE - MID VIX REGIME")
print("="*80 + "\n")
#print(results_table_mid.to_string(index=False))





# ===========================================================================
# HIGH VIX REGIME ANALYSIS
# ===========================================================================

print("\n" + "="*80)
print("HIGH VIX REGIME ANALYSIS")
print("="*80 + "\n")

# Filter to High VIX ONLY
high_vix_df = performance_df[performance_df['VIX_tercile'] == 'High VIX'].copy()
high_vix_composition = composition_df[composition_df['id'].isin(high_vix_df['id'])].copy()
high_vix_oos = high_vix_df[~high_vix_df['realized_index_return'].isna()].copy()

print(f"High VIX dates: {len(high_vix_df)}")
print(f"High VIX with OOS: {len(high_vix_oos)}\n")

# ===========================================================================
# CALCULATE STATISTICS
# ===========================================================================

n_dates_high = len(high_vix_df)
n_nonzeros_high = (high_vix_df.groupby('id')['expected_outperformance'].sum() != 0).sum()


# Breadth (from high vix data only)
breadth_by_month_high = high_vix_composition.groupby('id').size()
breadth_median_high = int(breadth_by_month_high.median())
breadth_iqr_high = int(breadth_by_month_high.quantile(0.75) - breadth_by_month_high.quantile(0.25))

# VIX stats
vix_mean_high = high_vix_df['VIX'].mean()

# Composition stats - ALL 4 option types
comp_stats_high = {
    'buy_call': (high_vix_composition['buy_call_weight'].median(), 
                 high_vix_composition['buy_call_weight'].quantile(0.75) - high_vix_composition['buy_call_weight'].quantile(0.25)),
    'buy_put': (high_vix_composition['buy_put_weight'].median(),
                high_vix_composition['buy_put_weight'].quantile(0.75) - high_vix_composition['buy_put_weight'].quantile(0.25)),
    'write_call': (high_vix_composition['write_call_weight'].median(),
                   high_vix_composition['write_call_weight'].quantile(0.75) - high_vix_composition['write_call_weight'].quantile(0.25)),
    'write_put': (high_vix_composition['write_put_weight'].median(),
                  high_vix_composition['write_put_weight'].quantile(0.75) - high_vix_composition['write_put_weight'].quantile(0.25)),
}

# Moneyness stats
moneyness_stats_high = {
    'buy_call': (high_vix_composition['buy_call_moneyness_mean'].median(),
                 high_vix_composition['buy_call_moneyness_mean'].quantile(0.75) - high_vix_composition['buy_call_moneyness_mean'].quantile(0.25)),
    'buy_put': (high_vix_composition['buy_put_moneyness_mean'].median(),
                high_vix_composition['buy_put_moneyness_mean'].quantile(0.75) - high_vix_composition['buy_put_moneyness_mean'].quantile(0.25)),
    'write_call': (high_vix_composition['write_call_moneyness_mean'].median(),
                   high_vix_composition['write_call_moneyness_mean'].quantile(0.75) - high_vix_composition['write_call_moneyness_mean'].quantile(0.25)),
    'write_put': (high_vix_composition['write_put_moneyness_mean'].median(),
                  high_vix_composition['write_put_moneyness_mean'].quantile(0.75) - high_vix_composition['write_put_moneyness_mean'].quantile(0.25)),
}

# Implied Vol stats
iv_stats_high = {
    'buy_call': (high_vix_composition['buy_call_iv_mean'].median(),
                 high_vix_composition['buy_call_iv_mean'].quantile(0.75) - high_vix_composition['buy_call_iv_mean'].quantile(0.25)),
    'buy_put': (high_vix_composition['buy_put_iv_mean'].median(),
                high_vix_composition['buy_put_iv_mean'].quantile(0.75) - high_vix_composition['buy_put_iv_mean'].quantile(0.25)),
    'write_call': (high_vix_composition['write_call_iv_mean'].median(),
                   high_vix_composition['write_call_iv_mean'].quantile(0.75) - high_vix_composition['write_call_iv_mean'].quantile(0.25)),
    'write_put': (high_vix_composition['write_put_iv_mean'].median(),
                  high_vix_composition['write_put_iv_mean'].quantile(0.75) - high_vix_composition['write_put_iv_mean'].quantile(0.25)),
}

# ===========================================================================
# PROJECTED DISTRIBUTION (MEDIAN and IQR across months)
# ===========================================================================

passive_mean_proj_high = high_vix_df['proj_passive_mean'].median()
passive_mean_proj_iqr_high = high_vix_df['proj_passive_mean'].quantile(0.75) - high_vix_df['proj_passive_mean'].quantile(0.25)

passive_std_proj_high = high_vix_df['proj_passive_std'].median()
passive_std_proj_iqr_high = high_vix_df['proj_passive_std'].quantile(0.75) - high_vix_df['proj_passive_std'].quantile(0.25)

passive_skew_proj_high = high_vix_df['proj_passive_skew'].median()
passive_skew_proj_iqr_high = high_vix_df['proj_passive_skew'].quantile(0.75) - high_vix_df['proj_passive_skew'].quantile(0.25)

passive_cvar_proj_high = high_vix_df['proj_passive_cvar'].median()
passive_cvar_proj_iqr_high = high_vix_df['proj_passive_cvar'].quantile(0.75) - high_vix_df['proj_passive_cvar'].quantile(0.25)


premium_inc_high = high_vix_df['premium_percentage'].median()
premium_inc_iqr_high = high_vix_df['premium_percentage'].quantile(0.75) - high_vix_df['premium_percentage'].quantile(0.25)


active_mean_proj_high = high_vix_df['proj_active_mean'].median()
active_mean_proj_iqr_high = high_vix_df['proj_active_mean'].quantile(0.75) - high_vix_df['proj_active_mean'].quantile(0.25)

active_std_proj_high = high_vix_df['proj_active_std'].median()
active_std_proj_iqr_high = high_vix_df['proj_active_std'].quantile(0.75) - high_vix_df['proj_active_std'].quantile(0.25)

active_skew_proj_high = high_vix_df['proj_active_skew'].median()
active_skew_proj_iqr_high = high_vix_df['proj_active_skew'].quantile(0.75) - high_vix_df['proj_active_skew'].quantile(0.25)

active_cvar_proj_high = high_vix_df['proj_active_cvar'].median()
active_cvar_proj_iqr_high = high_vix_df['proj_active_cvar'].quantile(0.75) - high_vix_df['proj_active_cvar'].quantile(0.25)


# ===========================================================================
# REALIZED PERFORMANCE (Out-of-sample) - HIGH VIX ONLY
# ===========================================================================

if len(high_vix_oos) > 0:
    buy_call_mean_monthly_high = high_vix_oos['buy_call_return'].mean()
    buy_put_mean_monthly_high = high_vix_oos['buy_put_return'].mean()
    write_call_mean_monthly_high = high_vix_oos['write_call_return'].mean()
    write_put_mean_monthly_high = high_vix_oos['write_put_return'].mean()
    
    buy_call_mean_annual_high = annualize_return(buy_call_mean_monthly_high)
    buy_put_mean_annual_high = annualize_return(buy_put_mean_monthly_high)
    write_call_mean_annual_high = annualize_return(write_call_mean_monthly_high)
    write_put_mean_annual_high = annualize_return(write_put_mean_monthly_high)
    
    buy_call_std_high = high_vix_oos['buy_call_return'].std()
    buy_put_std_high = high_vix_oos['buy_put_return'].std()
    write_call_std_high = high_vix_oos['write_call_return'].std()
    write_put_std_high = high_vix_oos['write_put_return'].std()

    # Step 1: Calculate MONTHLY means
    passive_mean_monthly_high = high_vix_oos['realized_index_return'].mean()
    active_mean_monthly_high = high_vix_oos['realized_enhanced_return'].mean()
    
    # Step 2: Annualize the means
    passive_mean_real_high = annualize_return(passive_mean_monthly_high)
    active_mean_real_high = annualize_return(active_mean_monthly_high)
    
    # Step 3: Calculate volatilities (monthly std, then annualize)
    passive_std_real_high = annualize_vol(high_vix_oos['realized_index_return'].std())
    active_std_real_high = annualize_vol(high_vix_oos['realized_enhanced_return'].std())
    
    # ===========================================================================
    # PERFORMANCE IMPROVEMENTS (Active - Passive)
    # ===========================================================================
    
    mean_improvement_high = active_mean_real_high - passive_mean_real_high
    passive_rets_high = high_vix_oos['realized_index_return'].values
    active_rets_high = high_vix_oos['realized_enhanced_return'].values
    cer_results_high = calculate_cer_pvalues(passive_rets_high, active_rets_high, gamma_values=[1, 4, 10])
    p_cer1_high = cer_results_high[1]['p_value']
    p_cer4_high = cer_results_high[4]['p_value']
    p_cer10_high = cer_results_high[10]['p_value']
    print(p_cer1_high)
    print(p_cer4_high)
    print(p_cer10_high)
    
    # CERs calculated on annualized values
    cer1_passive_high = calculate_cer_from_mean_std(passive_mean_real_high, passive_std_real_high, 1)
    cer1_active_high = calculate_cer_from_mean_std(active_mean_real_high, active_std_real_high, 1)
    cer1_improvement_high = cer1_active_high - cer1_passive_high
    
    cer4_passive_high = calculate_cer_from_mean_std(passive_mean_real_high, passive_std_real_high, 4)
    cer4_active_high = calculate_cer_from_mean_std(active_mean_real_high, active_std_real_high, 4)
    cer4_improvement_high = cer4_active_high - cer4_passive_high
    
    cer10_passive_high = calculate_cer_from_mean_std(passive_mean_real_high, passive_std_real_high, 10)
    cer10_active_high = calculate_cer_from_mean_std(active_mean_real_high, active_std_real_high, 10)
    cer10_improvement_high = cer10_active_high - cer10_passive_high
    
    # P-values from paired t-test
    from scipy.stats import ttest_rel
    _, p_mean_high = ttest_rel(high_vix_oos['realized_enhanced_return'], high_vix_oos['realized_index_return'])
    
else:
    passive_mean_real_high = np.nan
    active_mean_real_high = np.nan
    mean_improvement_high = np.nan
    cer1_improvement_high = np.nan
    cer4_improvement_high = np.nan
    cer10_improvement_high = np.nan
    p_mean_high = np.nan
    buy_call_mean_annual_high = np.nan
    buy_put_mean_annual_high = np.nan
    write_call_mean_annual_high = np.nan
    write_put_mean_annual_high = np.nan
    buy_call_std_high = np.nan
    buy_put_std_high = np.nan
    write_call_std_high = np.nan
    write_put_std_high = np.nan




# ===========================================================================
# VIX REGIMES SIDE-BY-SIDE COMPARISON TABLE (ALL + 3 TERCILES)
# ===========================================================================

import pandas as pd
import numpy as np
import math

print("\n" + "="*140)
print("MRP DRO GAMMA_RANGE = [2.3, 4.0] N = 150")
print("="*140 + "\n")

# ===========================================================================
# Helper function to format values
# ===========================================================================

def format_value(val, is_percentage=False, decimals=2):
    """Format values consistently"""
    if pd.isna(val) or np.isnan(val):
        return 'N/A'
    if is_percentage:
        return f'{val*100:.{decimals}f}'
    return f'{val:.{decimals}f}'

def format_with_iqr(median, iqr, is_percentage=False, decimals=2):
    """Format median with IQR in brackets"""
    med_str = format_value(median, is_percentage, decimals)
    iqr_str = format_value(iqr, is_percentage, decimals)
    if med_str == 'N/A' or iqr_str == 'N/A':
        return 'N/A'
    return f'{med_str} [{iqr_str}]'

# ===========================================================================
# RECALCULATE NON-ZEROS USING NET_PREMIUM FOR ALL REGIMES
# ===========================================================================

# For All VIX
n_nonzeros_all = (all_data_df.groupby('id')['expected_outperformance'].sum() != 0).sum()


# ===========================================================================
# COLLECT DATA FOR ALL FOUR REGIMES
# ===========================================================================

regime_data = {
    'All VIX': {
        'vix_mean': vix_mean,
        'n_dates': n_dates,
        'n_nonzeros': n_nonzeros_all,
        'breadth_median': breadth_median,
        'breadth_iqr': breadth_iqr,
        
        # Composition
        'comp_buy_call': comp_stats['buy_call'],
        'comp_buy_put': comp_stats['buy_put'],
        'comp_write_call': comp_stats['write_call'],
        'comp_write_put': comp_stats['write_put'],
        
        # Moneyness
        'mon_buy_call': moneyness_stats['buy_call'],
        'mon_buy_put': moneyness_stats['buy_put'],
        'mon_write_call': moneyness_stats['write_call'],
        'mon_write_put': moneyness_stats['write_put'],
        
        # Implied Vol
        'iv_buy_call': iv_stats['buy_call'],
        'iv_buy_put': iv_stats['buy_put'],
        'iv_write_call': iv_stats['write_call'],
        'iv_write_put': iv_stats['write_put'],
        
        # Projected Distribution
        'proj_passive_mean': (passive_mean_proj, passive_mean_proj_iqr),
        'proj_passive_std': (passive_std_proj, passive_std_proj_iqr),
        'proj_passive_skew': (passive_skew_proj, passive_skew_proj_iqr),
        'proj_passive_cvar': (passive_cvar_proj, passive_cvar_proj_iqr), 
        'premium_inc': (premium_inc, premium_inc_iqr), 
        'proj_active_mean': (active_mean_proj, active_mean_proj_iqr),
        'proj_active_std': (active_std_proj, active_std_proj_iqr),
        'proj_active_skew': (active_skew_proj, active_skew_proj_iqr),
        'proj_active_cvar': (active_cvar_proj, active_cvar_proj_iqr), 
        
        # Realized Performance
        'real_passive_mean': passive_mean_real,
        'real_active_mean': active_mean_real,
        'real_buy_call': (buy_call_mean_annual, buy_call_std),
        'real_buy_put': (buy_put_mean_annual, buy_put_std),
        'real_write_call': (write_call_mean_annual, write_call_std),
        'real_write_put': (write_put_mean_annual, write_put_std),
        
        # Improvements
        'mean_improvement': mean_improvement,
        'p_mean': p_mean,
        'cer1_improvement': cer1_improvement,
        'cer4_improvement': cer4_improvement,
        'cer10_improvement': cer10_improvement,
        
        # SSD
        'ssd_proportion': n_nonzeros_all/n_dates if n_dates > 0 else np.nan
    },
    
    'Low VIX': {
        'vix_mean': vix_mean_low,
        'n_dates': n_dates_low,
        'n_nonzeros': n_nonzeros_low,
        'breadth_median': breadth_median_low,
        'breadth_iqr': breadth_iqr_low,
        
        # Composition
        'comp_buy_call': comp_stats_low['buy_call'],
        'comp_buy_put': comp_stats_low['buy_put'],
        'comp_write_call': comp_stats_low['write_call'],
        'comp_write_put': comp_stats_low['write_put'],
        
        # Moneyness
        'mon_buy_call': moneyness_stats_low['buy_call'],
        'mon_buy_put': moneyness_stats_low['buy_put'],
        'mon_write_call': moneyness_stats_low['write_call'],
        'mon_write_put': moneyness_stats_low['write_put'],
        
        # Implied Vol
        'iv_buy_call': iv_stats_low['buy_call'],
        'iv_buy_put': iv_stats_low['buy_put'],
        'iv_write_call': iv_stats_low['write_call'],
        'iv_write_put': iv_stats_low['write_put'],
        
        # Projected Distribution
        'proj_passive_mean': (passive_mean_proj_low, passive_mean_proj_iqr_low),
        'proj_passive_std': (passive_std_proj_low, passive_std_proj_iqr_low),
        'proj_passive_skew': (passive_skew_proj_low, passive_skew_proj_iqr_low),
        'proj_passive_cvar': (passive_cvar_proj_low, passive_cvar_proj_iqr_low), 
        'premium_inc': (premium_inc_low, premium_inc_iqr_low), 
        'proj_active_mean': (active_mean_proj_low, active_mean_proj_iqr_low),
        'proj_active_std': (active_std_proj_low, active_std_proj_iqr_low),
        'proj_active_skew': (active_skew_proj_low, active_skew_proj_iqr_low),
        'proj_active_cvar': (active_cvar_proj_low  , active_cvar_proj_iqr_low ), 
        
        # Realized Performance
        'real_passive_mean': passive_mean_real_low,
        'real_active_mean': active_mean_real_low,
        'real_buy_call': (buy_call_mean_annual_low, buy_call_std_low),
        'real_buy_put': (buy_put_mean_annual_low, buy_put_std_low),
        'real_write_call': (write_call_mean_annual_low, write_call_std_low),
        'real_write_put': (write_put_mean_annual_low, write_put_std_low),
        
        # Improvements
        'mean_improvement': mean_improvement_low,
        'p_mean': p_mean_low,
        'cer1_improvement': cer1_improvement_low,
        'cer4_improvement': cer4_improvement_low,
        'cer10_improvement': cer10_improvement_low,
        
        # SSD
        'ssd_proportion': n_nonzeros_low/n_dates_low if n_dates_low > 0 else np.nan
    },
    
    'Mid VIX': {
        'vix_mean': vix_mean_mid,
        'n_dates': n_dates_mid,
        'n_nonzeros': n_nonzeros_mid,
        'breadth_median': breadth_median_mid,
        'breadth_iqr': breadth_iqr_mid,
        
        # Composition
        'comp_buy_call': comp_stats_mid['buy_call'],
        'comp_buy_put': comp_stats_mid['buy_put'],
        'comp_write_call': comp_stats_mid['write_call'],
        'comp_write_put': comp_stats_mid['write_put'],
        
        # Moneyness
        'mon_buy_call': moneyness_stats_mid['buy_call'],
        'mon_buy_put': moneyness_stats_mid['buy_put'],
        'mon_write_call': moneyness_stats_mid['write_call'],
        'mon_write_put': moneyness_stats_mid['write_put'],
        
        # Implied Vol
        'iv_buy_call': iv_stats_mid['buy_call'],
        'iv_buy_put': iv_stats_mid['buy_put'],
        'iv_write_call': iv_stats_mid['write_call'],
        'iv_write_put': iv_stats_mid['write_put'],
        
        # Projected Distribution
        'proj_passive_mean': (passive_mean_proj_mid, passive_mean_proj_iqr_mid),
        'proj_passive_std': (passive_std_proj_mid, passive_std_proj_iqr_mid),
        'proj_passive_skew': (passive_skew_proj_mid, passive_skew_proj_iqr_mid),
        'proj_passive_cvar': (passive_cvar_proj_mid, passive_cvar_proj_iqr_mid), 
        'premium_inc': (premium_inc_mid, premium_inc_iqr_mid), 
        'proj_active_mean': (active_mean_proj_mid, active_mean_proj_iqr_mid),
        'proj_active_std': (active_std_proj_mid, active_std_proj_iqr_mid),
        'proj_active_skew': (active_skew_proj_mid, active_skew_proj_iqr_mid),
        'proj_active_cvar': (active_cvar_proj_mid, active_cvar_proj_iqr_mid), 
        
        # Realized Performance
        'real_passive_mean': passive_mean_real_mid,
        'real_active_mean': active_mean_real_mid,
        'real_buy_call': (buy_call_mean_annual_mid, buy_call_std_mid),
        'real_buy_put': (buy_put_mean_annual_mid, buy_put_std_mid),
        'real_write_call': (write_call_mean_annual_mid, write_call_std_mid),
        'real_write_put': (write_put_mean_annual_mid, write_put_std_mid),
        
        # Improvements
        'mean_improvement': mean_improvement_mid,
        'p_mean': p_mean_mid,
        'cer1_improvement': cer1_improvement_mid,
        'cer4_improvement': cer4_improvement_mid,
        'cer10_improvement': cer10_improvement_mid,
        
        # SSD
        'ssd_proportion': n_nonzeros_mid/n_dates_mid if n_dates_mid > 0 else np.nan
    },
    
    'High VIX': {
        'vix_mean': vix_mean_high,
        'n_dates': n_dates_high,
        'n_nonzeros': n_nonzeros_high,
        'breadth_median': breadth_median_high,
        'breadth_iqr': breadth_iqr_high,
        
        # Composition
        'comp_buy_call': comp_stats_high['buy_call'],
        'comp_buy_put': comp_stats_high['buy_put'],
        'comp_write_call': comp_stats_high['write_call'],
        'comp_write_put': comp_stats_high['write_put'],
        
        # Moneyness
        'mon_buy_call': moneyness_stats_high['buy_call'],
        'mon_buy_put': moneyness_stats_high['buy_put'],
        'mon_write_call': moneyness_stats_high['write_call'],
        'mon_write_put': moneyness_stats_high['write_put'],
        
        # Implied Vol
        'iv_buy_call': iv_stats_high['buy_call'],
        'iv_buy_put': iv_stats_high['buy_put'],
        'iv_write_call': iv_stats_high['write_call'],
        'iv_write_put': iv_stats_high['write_put'],
        
        # Projected Distribution
        'proj_passive_mean': (passive_mean_proj_high, passive_mean_proj_iqr_high),
        'proj_passive_std': (passive_std_proj_high, passive_std_proj_iqr_high),
        'proj_passive_skew': (passive_skew_proj_high, passive_skew_proj_iqr_high),
        'proj_passive_cvar': (passive_cvar_proj_high, passive_cvar_proj_iqr_high), 
        'premium_inc': (premium_inc_high, premium_inc_iqr_high), 
        'proj_active_mean': (active_mean_proj_high, active_mean_proj_iqr_high),
        'proj_active_std': (active_std_proj_high, active_std_proj_iqr_high),
        'proj_active_skew': (active_skew_proj_high, active_skew_proj_iqr_high),
        'proj_active_cvar': (active_cvar_proj_high, active_cvar_proj_iqr_high), 
        
        # Realized Performance
        'real_passive_mean': passive_mean_real_high,
        'real_active_mean': active_mean_real_high,
        'real_buy_call': (buy_call_mean_annual_high, buy_call_std_high),
        'real_buy_put': (buy_put_mean_annual_high, buy_put_std_high),
        'real_write_call': (write_call_mean_annual_high, write_call_std_high),
        'real_write_put': (write_put_mean_annual_high, write_put_std_high),
        
        # Improvements
        'mean_improvement': mean_improvement_high,
        'p_mean': p_mean_high,
        'cer1_improvement': cer1_improvement_high,
        'cer4_improvement': cer4_improvement_high,
        'cer10_improvement': cer10_improvement_high,
        
        # SSD
        'ssd_proportion': n_nonzeros_high/n_dates_high if n_dates_high > 0 else np.nan
    }
}

# ===========================================================================
# BUILD COMPARISON TABLE
# ===========================================================================

rows = []

def add_comparison_row(section, metric, all_val, low_val, mid_val, high_val):
    """Add a row to the comparison table"""
    rows.append({
        'Section': section,
        'Metric': metric,
        'All VIX': all_val,
        'Low VIX': low_val,
        'Mid VIX': mid_val,
        'High VIX': high_val
    })

# Header information
add_comparison_row('Objective', 'VIX',
                   'All Regimes',
                   'Low VIX',
                   'Mid VIX',
                   'High VIX')

add_comparison_row('Restrictions', 'Filters',
                   'C96/108; PVI04',
                   'C96/108; PVI04',
                   'C96/108; PVI04',
                   'C96/108; PVI04')

add_comparison_row('Specifications', 'Spread',
                   'Observed',
                   'Observed',
                   'Observed',
                   'Observed')

add_comparison_row('', 'TTE',
                   '28',
                   '28',
                   '28',
                   '28')

add_comparison_row('', 'VOX',
                   'All',
                   'All',
                   'All',
                   'All')

add_comparison_row('', 'Crisis',
                   'Included',
                   'Included',
                   'Included',
                   'Included')

add_comparison_row('', '', '', '', '', '')

# Count
add_comparison_row('Count', 'Dates',
                   str(regime_data['All VIX']['n_dates']),
                   str(regime_data['Low VIX']['n_dates']),
                   str(regime_data['Mid VIX']['n_dates']),
                   str(regime_data['High VIX']['n_dates']))

add_comparison_row('Count', 'Non-zeros',
                   str(regime_data['All VIX']['n_nonzeros']),
                   str(regime_data['Low VIX']['n_nonzeros']),
                   str(regime_data['Mid VIX']['n_nonzeros']),
                   str(regime_data['High VIX']['n_nonzeros']))

add_comparison_row('Count', 'Breadth',
                   f"{regime_data['All VIX']['breadth_median']} [{regime_data['All VIX']['breadth_iqr']}]",
                   f"{regime_data['Low VIX']['breadth_median']} [{regime_data['Low VIX']['breadth_iqr']}]",
                   f"{regime_data['Mid VIX']['breadth_median']} [{regime_data['Mid VIX']['breadth_iqr']}]",
                   f"{regime_data['High VIX']['breadth_median']} [{regime_data['High VIX']['breadth_iqr']}]")

add_comparison_row('', '', '', '', '', '')

# Composition
add_comparison_row('Composition', 'BuyCall',
                   format_with_iqr(regime_data['All VIX']['comp_buy_call'][0], 
                                  regime_data['All VIX']['comp_buy_call'][1]),
                   format_with_iqr(regime_data['Low VIX']['comp_buy_call'][0], 
                                  regime_data['Low VIX']['comp_buy_call'][1]),
                   format_with_iqr(regime_data['Mid VIX']['comp_buy_call'][0], 
                                  regime_data['Mid VIX']['comp_buy_call'][1]),
                   format_with_iqr(regime_data['High VIX']['comp_buy_call'][0], 
                                  regime_data['High VIX']['comp_buy_call'][1]))

add_comparison_row('Composition', 'BuyPut',
                   format_with_iqr(regime_data['All VIX']['comp_buy_put'][0], 
                                  regime_data['All VIX']['comp_buy_put'][1]),
                   format_with_iqr(regime_data['Low VIX']['comp_buy_put'][0], 
                                  regime_data['Low VIX']['comp_buy_put'][1]),
                   format_with_iqr(regime_data['Mid VIX']['comp_buy_put'][0], 
                                  regime_data['Mid VIX']['comp_buy_put'][1]),
                   format_with_iqr(regime_data['High VIX']['comp_buy_put'][0], 
                                  regime_data['High VIX']['comp_buy_put'][1]))

add_comparison_row('Composition', 'WriteCall',
                   format_with_iqr(regime_data['All VIX']['comp_write_call'][0], 
                                  regime_data['All VIX']['comp_write_call'][1]),
                   format_with_iqr(regime_data['Low VIX']['comp_write_call'][0], 
                                  regime_data['Low VIX']['comp_write_call'][1]),
                   format_with_iqr(regime_data['Mid VIX']['comp_write_call'][0], 
                                  regime_data['Mid VIX']['comp_write_call'][1]),
                   format_with_iqr(regime_data['High VIX']['comp_write_call'][0], 
                                  regime_data['High VIX']['comp_write_call'][1]))

add_comparison_row('Composition', 'WritePut',
                   format_with_iqr(regime_data['All VIX']['comp_write_put'][0], 
                                  regime_data['All VIX']['comp_write_put'][1]),
                   format_with_iqr(regime_data['Low VIX']['comp_write_put'][0], 
                                  regime_data['Low VIX']['comp_write_put'][1]),
                   format_with_iqr(regime_data['Mid VIX']['comp_write_put'][0], 
                                  regime_data['Mid VIX']['comp_write_put'][1]),
                   format_with_iqr(regime_data['High VIX']['comp_write_put'][0], 
                                  regime_data['High VIX']['comp_write_put'][1]))

add_comparison_row('', '', '', '', '', '')

# Moneyness
add_comparison_row('Moneyness', 'BuyCall',
                   format_with_iqr(regime_data['All VIX']['mon_buy_call'][0], 
                                  regime_data['All VIX']['mon_buy_call'][1]),
                   format_with_iqr(regime_data['Low VIX']['mon_buy_call'][0], 
                                  regime_data['Low VIX']['mon_buy_call'][1]),
                   format_with_iqr(regime_data['Mid VIX']['mon_buy_call'][0], 
                                  regime_data['Mid VIX']['mon_buy_call'][1]),
                   format_with_iqr(regime_data['High VIX']['mon_buy_call'][0], 
                                  regime_data['High VIX']['mon_buy_call'][1]))

add_comparison_row('Moneyness', 'BuyPut',
                   format_with_iqr(regime_data['All VIX']['mon_buy_put'][0], 
                                  regime_data['All VIX']['mon_buy_put'][1]),
                   format_with_iqr(regime_data['Low VIX']['mon_buy_put'][0], 
                                  regime_data['Low VIX']['mon_buy_put'][1]),
                   format_with_iqr(regime_data['Mid VIX']['mon_buy_put'][0], 
                                  regime_data['Mid VIX']['mon_buy_put'][1]),
                   format_with_iqr(regime_data['High VIX']['mon_buy_put'][0], 
                                  regime_data['High VIX']['mon_buy_put'][1]))

add_comparison_row('Moneyness', 'WriteCall',
                   format_with_iqr(regime_data['All VIX']['mon_write_call'][0], 
                                  regime_data['All VIX']['mon_write_call'][1]),
                   format_with_iqr(regime_data['Low VIX']['mon_write_call'][0], 
                                  regime_data['Low VIX']['mon_write_call'][1]),
                   format_with_iqr(regime_data['Mid VIX']['mon_write_call'][0], 
                                  regime_data['Mid VIX']['mon_write_call'][1]),
                   format_with_iqr(regime_data['High VIX']['mon_write_call'][0], 
                                  regime_data['High VIX']['mon_write_call'][1]))

add_comparison_row('Moneyness', 'WritePut',
                   format_with_iqr(regime_data['All VIX']['mon_write_put'][0], 
                                  regime_data['All VIX']['mon_write_put'][1]),
                   format_with_iqr(regime_data['Low VIX']['mon_write_put'][0], 
                                  regime_data['Low VIX']['mon_write_put'][1]),
                   format_with_iqr(regime_data['Mid VIX']['mon_write_put'][0], 
                                  regime_data['Mid VIX']['mon_write_put'][1]),
                   format_with_iqr(regime_data['High VIX']['mon_write_put'][0], 
                                  regime_data['High VIX']['mon_write_put'][1]))

add_comparison_row('', '', '', '', '', '')

# Implied Vol
add_comparison_row('Implied Vol', 'BuyCall',
                   format_with_iqr(regime_data['All VIX']['iv_buy_call'][0], 
                                  regime_data['All VIX']['iv_buy_call'][1]),
                   format_with_iqr(regime_data['Low VIX']['iv_buy_call'][0], 
                                  regime_data['Low VIX']['iv_buy_call'][1]),
                   format_with_iqr(regime_data['Mid VIX']['iv_buy_call'][0], 
                                  regime_data['Mid VIX']['iv_buy_call'][1]),
                   format_with_iqr(regime_data['High VIX']['iv_buy_call'][0], 
                                  regime_data['High VIX']['iv_buy_call'][1]))

add_comparison_row('Implied Vol', 'BuyPut',
                   format_with_iqr(regime_data['All VIX']['iv_buy_put'][0], 
                                  regime_data['All VIX']['iv_buy_put'][1]),
                   format_with_iqr(regime_data['Low VIX']['iv_buy_put'][0], 
                                  regime_data['Low VIX']['iv_buy_put'][1]),
                   format_with_iqr(regime_data['Mid VIX']['iv_buy_put'][0], 
                                  regime_data['Mid VIX']['iv_buy_put'][1]),
                   format_with_iqr(regime_data['High VIX']['iv_buy_put'][0], 
                                  regime_data['High VIX']['iv_buy_put'][1]))

add_comparison_row('Implied Vol', 'WriteCall',
                   format_with_iqr(regime_data['All VIX']['iv_write_call'][0], 
                                  regime_data['All VIX']['iv_write_call'][1]),
                   format_with_iqr(regime_data['Low VIX']['iv_write_call'][0], 
                                  regime_data['Low VIX']['iv_write_call'][1]),
                   format_with_iqr(regime_data['Mid VIX']['iv_write_call'][0], 
                                  regime_data['Mid VIX']['iv_write_call'][1]),
                   format_with_iqr(regime_data['High VIX']['iv_write_call'][0], 
                                  regime_data['High VIX']['iv_write_call'][1]))

add_comparison_row('Implied Vol', 'WritePut',
                   format_with_iqr(regime_data['All VIX']['iv_write_put'][0], 
                                  regime_data['All VIX']['iv_write_put'][1]),
                   format_with_iqr(regime_data['Low VIX']['iv_write_put'][0], 
                                  regime_data['Low VIX']['iv_write_put'][1]),
                   format_with_iqr(regime_data['Mid VIX']['iv_write_put'][0], 
                                  regime_data['Mid VIX']['iv_write_put'][1]),
                   format_with_iqr(regime_data['High VIX']['iv_write_put'][0], 
                                  regime_data['High VIX']['iv_write_put'][1]))

add_comparison_row('', '', '', '', '', '')

# Projected Distribution
add_comparison_row('Projected', 'Passive Mean',
                   format_with_iqr(annualize_return(regime_data['All VIX']['proj_passive_mean'][0]),
                                  annualize_return(regime_data['All VIX']['proj_passive_mean'][1]), True),
                   format_with_iqr(annualize_return(regime_data['Low VIX']['proj_passive_mean'][0]),
                                  annualize_return(regime_data['Low VIX']['proj_passive_mean'][1]), True),
                   format_with_iqr(annualize_return(regime_data['Mid VIX']['proj_passive_mean'][0]),
                                  annualize_return(regime_data['Mid VIX']['proj_passive_mean'][1]), True),
                   format_with_iqr(annualize_return(regime_data['High VIX']['proj_passive_mean'][0]),
                                  annualize_return(regime_data['High VIX']['proj_passive_mean'][1]), True))

add_comparison_row('Projected', 'Passive Std',
                   format_with_iqr(annualize_vol(regime_data['All VIX']['proj_passive_std'][0]),
                                  annualize_vol(regime_data['All VIX']['proj_passive_std'][1]), True),
                   format_with_iqr(annualize_vol(regime_data['Low VIX']['proj_passive_std'][0]),
                                  annualize_vol(regime_data['Low VIX']['proj_passive_std'][1]), True),
                   format_with_iqr(annualize_vol(regime_data['Mid VIX']['proj_passive_std'][0]),
                                  annualize_vol(regime_data['Mid VIX']['proj_passive_std'][1]), True),
                   format_with_iqr(annualize_vol(regime_data['High VIX']['proj_passive_std'][0]),
                                  annualize_vol(regime_data['High VIX']['proj_passive_std'][1]), True))

add_comparison_row('Projected', 'Passive Skew',
                   format_with_iqr(regime_data['All VIX']['proj_passive_skew'][0],
                                  regime_data['All VIX']['proj_passive_skew'][1]),
                   format_with_iqr(regime_data['Low VIX']['proj_passive_skew'][0],
                                  regime_data['Low VIX']['proj_passive_skew'][1]),
                   format_with_iqr(regime_data['Mid VIX']['proj_passive_skew'][0],
                                  regime_data['Mid VIX']['proj_passive_skew'][1]),
                   format_with_iqr(regime_data['High VIX']['proj_passive_skew'][0],
                                  regime_data['High VIX']['proj_passive_skew'][1]))

add_comparison_row('Projected', 'Passive CVaR',
                   format_with_iqr(passive_cvar_proj*math.sqrt(12), passive_cvar_proj_iqr*math.sqrt(12), True),  # All
                    format_with_iqr(passive_cvar_proj_low*math.sqrt(12), passive_cvar_proj_iqr_low*math.sqrt(12), True),  # Low
                    format_with_iqr(passive_cvar_proj_mid*math.sqrt(12), passive_cvar_proj_iqr_mid*math.sqrt(12), True),  # Mid
                    format_with_iqr(passive_cvar_proj_high*math.sqrt(12), passive_cvar_proj_iqr_high*math.sqrt(12), True))  # High
                

add_comparison_row('Projected', 'Premium Inc',
                    format_with_iqr(annualize_return(premium_inc), annualize_return(premium_inc_iqr), True),  # All
                    format_with_iqr(annualize_return(premium_inc_low), annualize_return(premium_inc_iqr_low), True),  # Low
                    format_with_iqr(annualize_return(premium_inc_mid), annualize_return(premium_inc_iqr_mid), True),  # Mid
                    format_with_iqr(annualize_return(premium_inc_high), annualize_return(premium_inc_iqr_high), True)) # High
                

add_comparison_row('Projected', 'Active Mean',
                   format_with_iqr(annualize_return(regime_data['All VIX']['proj_active_mean'][0]),
                                  annualize_return(regime_data['All VIX']['proj_active_mean'][1]), True),
                   format_with_iqr(annualize_return(regime_data['Low VIX']['proj_active_mean'][0]),
                                  annualize_return(regime_data['Low VIX']['proj_active_mean'][1]), True),
                   format_with_iqr(annualize_return(regime_data['Mid VIX']['proj_active_mean'][0]),
                                  annualize_return(regime_data['Mid VIX']['proj_active_mean'][1]), True),
                   format_with_iqr(annualize_return(regime_data['High VIX']['proj_active_mean'][0]),
                                  annualize_return(regime_data['High VIX']['proj_active_mean'][1]), True))

add_comparison_row('Projected', 'Active Std',
                   format_with_iqr(annualize_vol(regime_data['All VIX']['proj_active_std'][0]),
                                  annualize_vol(regime_data['All VIX']['proj_active_std'][1]), True),
                   format_with_iqr(annualize_vol(regime_data['Low VIX']['proj_active_std'][0]),
                                  annualize_vol(regime_data['Low VIX']['proj_active_std'][1]), True),
                   format_with_iqr(annualize_vol(regime_data['Mid VIX']['proj_active_std'][0]),
                                  annualize_vol(regime_data['Mid VIX']['proj_active_std'][1]), True),
                   format_with_iqr(annualize_vol(regime_data['High VIX']['proj_active_std'][0]),
                                  annualize_vol(regime_data['High VIX']['proj_active_std'][1]), True))

add_comparison_row('Projected', 'Active Skew',
                   format_with_iqr(regime_data['All VIX']['proj_active_skew'][0],
                                  regime_data['All VIX']['proj_active_skew'][1]),
                   format_with_iqr(regime_data['Low VIX']['proj_active_skew'][0],
                                  regime_data['Low VIX']['proj_active_skew'][1]),
                   format_with_iqr(regime_data['Mid VIX']['proj_active_skew'][0],
                                  regime_data['Mid VIX']['proj_active_skew'][1]),
                   format_with_iqr(regime_data['High VIX']['proj_active_skew'][0],
                                  regime_data['High VIX']['proj_active_skew'][1]))

add_comparison_row('Projected', 'Active CVaR',
                   format_with_iqr(active_cvar_proj*math.sqrt(12), active_cvar_proj_iqr*math.sqrt(12), True),  # All
                format_with_iqr(active_cvar_proj_low*math.sqrt(12), active_cvar_proj_iqr_low*math.sqrt(12), True),  # Low
                format_with_iqr(active_cvar_proj_mid*math.sqrt(12), active_cvar_proj_iqr_mid*math.sqrt(12), True),  # Mid
                format_with_iqr(active_cvar_proj_high*math.sqrt(12), active_cvar_proj_iqr_high*math.sqrt(12), True))  # High
                   # format_value(regime_data['All VIX']['proj_active_cvar']*math.sqrt(12), True),
                   # format_value(regime_data['Low VIX']['proj_active_cvar']*math.sqrt(12), True),
                   # format_value(regime_data['Mid VIX']['proj_active_cvar']*math.sqrt(12), True),
                   # format_value(regime_data['High VIX']['proj_active_cvar']*math.sqrt(12), True))

add_comparison_row('', '', '', '', '', '')

# Realized Performance
add_comparison_row('Realized', 'Passive Mean',
                   format_value(regime_data['All VIX']['real_passive_mean'], True),
                   format_value(regime_data['Low VIX']['real_passive_mean'], True),
                   format_value(regime_data['Mid VIX']['real_passive_mean'], True),
                   format_value(regime_data['High VIX']['real_passive_mean'], True))

add_comparison_row('Realized', 'Active Mean',
                   format_value(regime_data['All VIX']['real_active_mean'], True),
                   format_value(regime_data['Low VIX']['real_active_mean'], True),
                   format_value(regime_data['Mid VIX']['real_active_mean'], True),
                   format_value(regime_data['High VIX']['real_active_mean'], True))

# Individual option returns with std in parentheses
all_bc = regime_data['All VIX']['real_buy_call']
low_bc = regime_data['Low VIX']['real_buy_call']
mid_bc = regime_data['Mid VIX']['real_buy_call']
high_bc = regime_data['High VIX']['real_buy_call']
add_comparison_row('Realized', 'BuyCall Mean',
                   f"{format_value(all_bc[0], True)} ({format_value(annualize_vol(all_bc[1]), True)})" if not np.isnan(all_bc[0]) else 'N/A',
                   f"{format_value(low_bc[0], True)} ({format_value(annualize_vol(low_bc[1]), True)})" if not np.isnan(low_bc[0]) else 'N/A',
                   f"{format_value(mid_bc[0], True)} ({format_value(annualize_vol(mid_bc[1]), True)})" if not np.isnan(mid_bc[0]) else 'N/A',
                   f"{format_value(high_bc[0], True)} ({format_value(annualize_vol(high_bc[1]), True)})" if not np.isnan(high_bc[0]) else 'N/A')

all_bp = regime_data['All VIX']['real_buy_put']
low_bp = regime_data['Low VIX']['real_buy_put']
mid_bp = regime_data['Mid VIX']['real_buy_put']
high_bp = regime_data['High VIX']['real_buy_put']
add_comparison_row('Realized', 'BuyPut Mean',
                   f"{format_value(all_bp[0], True)} ({format_value(annualize_vol(all_bp[1]), True)})" if not np.isnan(all_bp[0]) else 'N/A',
                   f"{format_value(low_bp[0], True)} ({format_value(annualize_vol(low_bp[1]), True)})" if not np.isnan(low_bp[0]) else 'N/A',
                   f"{format_value(mid_bp[0], True)} ({format_value(annualize_vol(mid_bp[1]), True)})" if not np.isnan(mid_bp[0]) else 'N/A',
                   f"{format_value(high_bp[0], True)} ({format_value(annualize_vol(high_bp[1]), True)})" if not np.isnan(high_bp[0]) else 'N/A')

all_wc = regime_data['All VIX']['real_write_call']
low_wc = regime_data['Low VIX']['real_write_call']
mid_wc = regime_data['Mid VIX']['real_write_call']
high_wc = regime_data['High VIX']['real_write_call']
add_comparison_row('Realized', 'WriteCall Mean',
                   f"{format_value(all_wc[0], True)} ({format_value(annualize_vol(all_wc[1]), True)})" if not np.isnan(all_wc[0]) else 'N/A',
                   f"{format_value(low_wc[0], True)} ({format_value(annualize_vol(low_wc[1]), True)})" if not np.isnan(low_wc[0]) else 'N/A',
                   f"{format_value(mid_wc[0], True)} ({format_value(annualize_vol(mid_wc[1]), True)})" if not np.isnan(mid_wc[0]) else 'N/A',
                   f"{format_value(high_wc[0], True)} ({format_value(annualize_vol(high_wc[1]), True)})" if not np.isnan(high_wc[0]) else 'N/A')

all_wp = regime_data['All VIX']['real_write_put']
low_wp = regime_data['Low VIX']['real_write_put']
mid_wp = regime_data['Mid VIX']['real_write_put']
high_wp = regime_data['High VIX']['real_write_put']
add_comparison_row('Realized', 'WritePut Mean',
                   f"{format_value(all_wp[0], True)} ({format_value(annualize_vol(all_wp[1]), True)})" if not np.isnan(all_wp[0]) else 'N/A',
                   f"{format_value(low_wp[0], True)} ({format_value(annualize_vol(low_wp[1]), True)})" if not np.isnan(low_wp[0]) else 'N/A',
                   f"{format_value(mid_wp[0], True)} ({format_value(annualize_vol(mid_wp[1]), True)})" if not np.isnan(mid_wp[0]) else 'N/A',
                   f"{format_value(high_wp[0], True)} ({format_value(annualize_vol(high_wp[1]), True)})" if not np.isnan(high_wp[0]) else 'N/A')

add_comparison_row('', '', '', '', '', '')

# Improvements
add_comparison_row('Improvements', 'Mean',
                   f"{format_value(regime_data['All VIX']['mean_improvement'], True)} ({format_value(regime_data['All VIX']['p_mean'], decimals=2)})",
                   f"{format_value(regime_data['Low VIX']['mean_improvement'], True)} ({format_value(regime_data['Low VIX']['p_mean'], decimals=2)})",
                   f"{format_value(regime_data['Mid VIX']['mean_improvement'], True)} ({format_value(regime_data['Mid VIX']['p_mean'], decimals=2)})",
                   f"{format_value(regime_data['High VIX']['mean_improvement'], True)} ({format_value(regime_data['High VIX']['p_mean'], decimals=2)})")

add_comparison_row('Improvements', 'CER1',
                   f"{format_value(regime_data['All VIX']['cer1_improvement'], True)} ({format_value(p_cer1, decimals=2)})", 
                   f"{format_value(regime_data['Low VIX']['cer1_improvement'], True)} ({format_value(p_cer1_low, decimals=2)})", 
                   f"{format_value(regime_data['Mid VIX']['cer1_improvement'], True)} ({format_value(p_cer1_mid, decimals=2)})", 
                   f"{format_value(regime_data['High VIX']['cer1_improvement'], True)} ({format_value(p_cer1_high, decimals=2)})")


add_comparison_row('Improvements', 'CER4',
                   f"{format_value(regime_data['All VIX']['cer4_improvement'], True)} ({format_value(p_cer4, decimals=2)})", 
                   f"{format_value(regime_data['Low VIX']['cer4_improvement'], True)} ({format_value(p_cer4_low, decimals=2)})", 
                   f"{format_value(regime_data['Mid VIX']['cer4_improvement'], True)} ({format_value(p_cer4_mid, decimals=2)})", 
                   f"{format_value(regime_data['High VIX']['cer4_improvement'], True)} ({format_value(p_cer4_high, decimals=2)})")


add_comparison_row('Improvements', 'CER10',
                   f"{format_value(regime_data['All VIX']['cer10_improvement'], True)} ({format_value(p_cer10, decimals=2)})", 
                   f"{format_value(regime_data['Low VIX']['cer10_improvement'], True)} ({format_value(p_cer10_low, decimals=2)})", 
                   f"{format_value(regime_data['Mid VIX']['cer10_improvement'], True)} ({format_value(p_cer10_mid, decimals=2)})", 
                   f"{format_value(regime_data['High VIX']['cer10_improvement'], True)} ({format_value(p_cer10_high, decimals=2)})")


add_comparison_row('', '', '', '', '', '')

# SSD
add_comparison_row('SSD test', 'Proportion',
                   format_value(regime_data['All VIX']['ssd_proportion']),
                   format_value(regime_data['Low VIX']['ssd_proportion']),
                   format_value(regime_data['Mid VIX']['ssd_proportion']),
                   format_value(regime_data['High VIX']['ssd_proportion']))

# ===========================================================================
# CREATE AND DISPLAY TABLE
# ===========================================================================

comparison_df = pd.DataFrame(rows)

# Set display options for better formatting
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(comparison_df.to_string(index=False))
print("\n" + "="*140)

# # ============================================================================
# # CORRECTED ELR TEST FOR SSD DOMINANCE
# # ============================================================================
# import numpy as np
# import cvxpy as cp
# from scipy.stats import chi2

# def create_blocks(data, block_size=12, step_size=3):
#     """
#     Create overlapping blocks for time series data.
    
#     Parameters:
#     -----------
#     data : array-like
#         Time series data
#     block_size : int
#         Number of observations per block (default: 12 for monthly data = 1 year)
#     step_size : int
#         Step between block starts (default: 3 for quarterly overlap)
    
#     Returns:
#     --------
#     blocks : list of arrays
#         List of overlapping blocks
#     """
#     T = len(data)
#     blocks = []
#     start = 0
    
#     while start + block_size <= T:
#         blocks.append(data[start:start + block_size])
#         start += step_size
    
#     if len(blocks) == 0 and T > 0:
#         # If not enough data for even one block, use all available data
#         blocks.append(data)
    
#     return blocks

# def compute_elr_test(passive_returns, active_returns, block_size=12, step_size=3, 
#                                 n_thresholds=20, verbose=True):
#     """
#     Compute ELR test following Arvanitis & Post (2024) more closely.
    
#     Key insight: The test statistic scaling should account for the effective sample size
#     given the block bootstrap structure, not just multiply by T.
#     """
    
#     T = len(passive_returns)
    
#     if T < block_size:
#         if verbose:
#             print(f"Warning: T={T} < block_size={block_size}. Adjusting block_size to {T//2}")
#         block_size = max(T // 2, 6)
#         step_size = max(block_size // 4, 1)
    
#     # Create blocks
#     blocks_passive = create_blocks(passive_returns, block_size, step_size)
#     blocks_active = create_blocks(active_returns, block_size, step_size)
#     T_star = len(blocks_passive)
#     B = len(blocks_passive[0]) if T_star > 0 else block_size
    
#     if T_star < 5:
#         if verbose:
#             print(f"ERROR: Only {T_star} blocks. Need at least 5.")
#         return np.nan, np.nan, None, {'error': 'insufficient_blocks'}
    
#     if verbose:
#         print(f"\nELR Test Setup:")
#         print(f"  Total observations (T): {T}")
#         print(f"  Number of blocks (T*): {T_star}")
#         print(f"  Block size (B): {B}")
#         print(f"  Step size: {step_size}")
    
#     # Uniform empirical probabilities
#     G_T = np.ones(T_star) / T_star
    
#     # Decision variable
#     G = cp.Variable(T_star, nonneg=True)
    
#     # Objective: minimize KL divergence
#     objective = cp.Minimize(-G_T @ cp.log(G))
    
#     # Constraints
#     constraints = [cp.sum(G) == 1]
    
#     # Define thresholds
#     all_returns = np.concatenate([passive_returns, active_returns])
#     thresholds = np.linspace(np.percentile(all_returns, 5), 
#                              np.percentile(all_returns, 95), 
#                              n_thresholds)
    
#     # SSD constraints
#     for tau in thresholds:
#         lpm_passive_blocks = np.array([
#             np.mean(np.maximum(tau - block, 0)) for block in blocks_passive
#         ])
#         lpm_active_blocks = np.array([
#             np.mean(np.maximum(tau - block, 0)) for block in blocks_active
#         ])
        
#         constraints.append(lpm_active_blocks @ G <= lpm_passive_blocks @ G)
    
#     # Solve
#     problem = cp.Problem(objective, constraints)
    
#     try:
#         problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        
#         if problem.status not in ['optimal', 'optimal_inaccurate']:
#             if verbose:
#                 print(f"WARNING: Optimization status: {problem.status}")
#             return np.nan, np.nan, None, {'error': 'optimization_failed'}
        
#         G_opt = G.value
#         G_opt = np.maximum(G_opt, 1e-12)
#         G_opt = G_opt / np.sum(G_opt)
        
#         # Compute KL divergence
#         kl_value = np.sum(G_T * np.log(G_T / G_opt))
        
#         # CRITICAL: Proper scaling for block bootstrap
#         # The effective degrees of freedom is T_star (number of blocks), not T
#         # Standard ELR: 2 * n * KL where n is the effective sample size
#         # With overlapping blocks, effective n is closer to T_star than T
        
#         # Option 1: Use T_star as effective sample size (conservative)
#         elr_stat_conservative = 2 * T_star * (T / B) * kl_value
        
#         # Option 2: Adjust for overlap structure (less conservative)
#         # Effective sample size ≈ T_star * sqrt(B/step_size)
#         overlap_adj = np.sqrt(B / step_size)
#         effective_n = T_star * overlap_adj
#         elr_stat_adjusted = 2 * effective_n * kl_value
        
#         # Use the more conservative estimate
#         elr_stat = elr_stat_conservative
        
#         # Count binding constraints
#         binding_constraints = []
#         for tau in thresholds:
#             lpm_p = np.array([np.mean(np.maximum(tau - block, 0)) for block in blocks_passive])
#             lpm_a = np.array([np.mean(np.maximum(tau - block, 0)) for block in blocks_active])
            
#             lpm_diff = (lpm_a @ G_opt) - (lpm_p @ G_opt)
#             if abs(lpm_diff) < 1e-5:
#                 binding_constraints.append(tau)
        
#         n_binding = max(len(binding_constraints), 1)
        
#         # P-value
#         p_value = 1 - chi2.cdf(elr_stat, df=n_binding)
        
#         # Diagnostics
#         diagnostics = {
#             'kl_divergence': kl_value,
#             'elr_statistic_conservative': elr_stat_conservative,
#             'elr_statistic_adjusted': elr_stat_adjusted,
#             'elr_statistic_used': elr_stat,
#             'p_value': p_value,
#             'df': n_binding,
#             'n_blocks': T_star,
#             'block_size': B,
#             'effective_sample_size': T_star,
#             'max_prob_deviation': np.max(np.abs(G_opt - 1/T_star)),
#             'n_binding_constraints': n_binding,
#             'optimization_status': problem.status
#         }
        
#         if verbose:
#             print(f"\nELR Test Results:")
#             print(f"  KL Divergence: {kl_value:.6f}")
#             print(f"  ELR Statistic (Conservative, using T*={T_star}): {elr_stat_conservative:.4f}")
#             print(f"  ELR Statistic (Adjusted, eff_n={effective_n:.1f}): {elr_stat_adjusted:.4f}")
#             print(f"  ELR Statistic (Used): {elr_stat:.4f}")
#             print(f"  Degrees of Freedom: {n_binding}")
#             print(f"  P-value: {p_value:.4f}")
#             print(f"  Max deviation from uniform: {diagnostics['max_prob_deviation']:.6f}")
#             print(f"  Binding constraints: {n_binding} / {n_thresholds}")
            
#             if p_value > 0.10:
#                 print(f"  ✓ Fail to reject H0: Consistent with SSD dominance (p={p_value:.4f})")
#             elif p_value > 0.05:
#                 print(f"  ? Weak evidence against dominance (p={p_value:.4f})")
#             else:
#                 print(f"  ✗ Reject H0: Active does NOT dominate passive (p={p_value:.4f})")
        
#         return elr_stat, p_value, G_opt, diagnostics
        
#     except Exception as e:
#         if verbose:
#             print(f"ERROR in ELR optimization: {str(e)}")
#         return np.nan, np.nan, None, {'error': str(e)}


# # ============================================================================
# # CHECK SSD VIOLATIONS DIRECTLY (BEFORE RUNNING ELR)
# # ============================================================================
# def check_ssd_violations(passive_returns, active_returns, n_points=50, verbose=True):
#     """
#     Check for empirical SSD violations.
#     For SSD: Active dominates Passive if cumulative shortfalls satisfy:
#         sum_{i=1}^k (tau - R_active)^+ <= sum_{i=1}^k (tau - R_passive)^+  for all tau
    
#     Returns number of violations and diagnostic info.
#     """
#     all_returns = np.concatenate([passive_returns, active_returns])
#     thresholds = np.linspace(np.min(all_returns), np.max(all_returns), n_points)
    
#     violations = []
#     cumulative_shortfalls_passive = []
#     cumulative_shortfalls_active = []
    
#     for tau in thresholds:
#         # Cumulative shortfall (integral of LPM)
#         shortfall_passive = np.sum(np.maximum(tau - passive_returns, 0))
#         shortfall_active = np.sum(np.maximum(tau - active_returns, 0))
        
#         cumulative_shortfalls_passive.append(shortfall_passive)
#         cumulative_shortfalls_active.append(shortfall_active)
        
#         # Violation occurs if active has MORE cumulative shortfall
#         if shortfall_active > shortfall_passive + 1e-8:
#             violations.append((tau, shortfall_active - shortfall_passive))
    
#     if verbose:
#         print(f"\nSSD Violation Check:")
#         print(f"  Threshold points evaluated: {n_points}")
#         print(f"  SSD violations found: {len(violations)}")
#         if len(violations) > 0:
#             print(f"  Max violation magnitude: {max(v[1] for v in violations):.6f}")
#             print(f"  Violations at thresholds: {[f'{v[0]:.4f}' for v in violations[:5]]}")
#         else:
#             print(f"  ✓ No SSD violations detected - Active may dominate Passive")
        
#         print(f"\nBasic Statistics:")
#         print(f"  Mean Passive: {np.mean(passive_returns):.4f}")
#         print(f"  Mean Active: {np.mean(active_returns):.4f}")
#         print(f"  Std Passive: {np.std(passive_returns):.4f}")
#         print(f"  Std Active: {np.std(active_returns):.4f}")
#         print(f"  Active > Passive: {np.sum(active_returns > passive_returns)} / {len(passive_returns)} months")
    
#     return violations, cumulative_shortfalls_passive, cumulative_shortfalls_active


# # ============================================================================
# # RUN ELR TEST FOR EACH VIX REGIME
# # ============================================================================

# print("\n" + "="*80)
# print("ELR TEST FOR SSD DOMINANCE - BY VIX REGIME")
# print("="*80)

# # Initialize results storage
# elr_results = {}

# # 1. ALL VIX
# print("\n" + "-"*80)
# print("ALL VIX REGIME")
# print("-"*80)
# if len(oos_data) >= 24:
#     passive_rets = oos_data['realized_index_return'].values
#     active_rets = oos_data['realized_enhanced_return'].values
    
#     # First check for violations
#     check_ssd_violations(passive_rets, active_rets)
    
#     # Then run ELR test
#     elr_stat, elr_pvalue, G_opt, diagnostics = compute_elr_test(
#         passive_rets, active_rets, 
#         block_size=12, step_size=3, n_thresholds=20
#     )
    
#     elr_results['all'] = {
#         'statistic': elr_stat,
#         'p_value': elr_pvalue,
#         'G_optimal': G_opt,
#         'diagnostics': diagnostics,
#         'n_obs': len(passive_rets)
#     }
    
#     # if G_opt is not None:
#     #     np.save('results/G_optimal_all.npy', G_opt)
# else:
#     print(f"Insufficient data: {len(oos_data)} < 24 observations")
#     elr_results['all'] = None

# # 2. LOW VIX
# print("\n" + "-"*80)
# print("LOW VIX REGIME")
# print("-"*80)
# if len(low_vix_oos) >= 18:  # Need at least 18 for 2 blocks of size 12 with step 3
#     passive_rets_low = low_vix_oos['realized_index_return'].values
#     active_rets_low = low_vix_oos['realized_enhanced_return'].values
    
#     check_ssd_violations(passive_rets_low, active_rets_low)
    
#     # Adjust block size for smaller sample
#     block_size_low = min(12, len(passive_rets_low) // 2)
#     step_size_low = max(block_size_low // 4, 1)
    
#     elr_stat_low, elr_pvalue_low, G_optimal_low, diagnostics_low = compute_elr_test(
#         passive_rets_low, active_rets_low,
#         block_size=block_size_low, step_size=step_size_low, n_thresholds=15
#     )
    
#     elr_results['low'] = {
#         'statistic': elr_stat_low,
#         'p_value': elr_pvalue_low,
#         'G_optimal': G_optimal_low,
#         'diagnostics': diagnostics_low,
#         'n_obs': len(passive_rets_low)
#     }
    
#     # if G_optimal_low is not None:
#     #     np.save('results/G_optimal_low.npy', G_optimal_low)
# else:
#     print(f"Insufficient data: {len(low_vix_oos)} < 18 observations")
#     elr_results['low'] = None

# # 3. MEDIUM VIX
# print("\n" + "-"*80)
# print("MEDIUM VIX REGIME")
# print("-"*80)
# if len(mid_vix_oos) >= 18:
#     passive_rets_med = mid_vix_oos['realized_index_return'].values
#     active_rets_med = mid_vix_oos['realized_enhanced_return'].values
    
#     check_ssd_violations(passive_rets_med, active_rets_med)
    
#     block_size_med = min(12, len(passive_rets_med) // 2)
#     step_size_med = max(block_size_med // 4, 1)
    
#     elr_stat_med, elr_pvalue_med, G_optimal_med, diagnostics_med = compute_elr_test(
#         passive_rets_med, active_rets_med,
#         block_size=block_size_med, step_size=step_size_med, n_thresholds=15
#     )
    
#     elr_results['med'] = {
#         'statistic': elr_stat_med,
#         'p_value': elr_pvalue_med,
#         'G_optimal': G_optimal_med,
#         'diagnostics': diagnostics_med,
#         'n_obs': len(passive_rets_med)
#     }
    
#     # if G_optimal_med is not None:
#     #     np.save('results/G_optimal_med.npy', G_optimal_med)
# else:
#     print(f"Insufficient data: {len(mid_vix_oos)} < 18 observations")
#     elr_results['med'] = None

# # 4. HIGH VIX
# print("\n" + "-"*80)
# print("HIGH VIX REGIME")
# print("-"*80)
# if len(high_vix_oos) >= 18:
#     passive_rets_high = high_vix_oos['realized_index_return'].values
#     active_rets_high = high_vix_oos['realized_enhanced_return'].values
    
#     check_ssd_violations(passive_rets_high, active_rets_high)
    
#     block_size_high = min(12, len(passive_rets_high) // 2)
#     step_size_high = max(block_size_high // 4, 1)
    
#     elr_stat_high, elr_pvalue_high, G_optimal_high, diagnostics_high = compute_elr_test(
#         passive_rets_high, active_rets_high,
#         block_size=block_size_high, step_size=step_size_high, n_thresholds=15
#     )
    
#     elr_results['high'] = {
#         'statistic': elr_stat_high,
#         'p_value': elr_pvalue_high,
#         'G_optimal': G_optimal_high,
#         'diagnostics': diagnostics_high,
#         'n_obs': len(passive_rets_high)
#     }
    
#     # if G_optimal_high is not None:
#     #     np.save('results/G_optimal_high.npy', G_optimal_high)
# else:
#     print(f"Insufficient data: {len(high_vix_oos)} < 18 observations")
#     elr_results['high'] = None
  

