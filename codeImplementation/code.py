import pandas as pd
import numpy as np
from scipy.stats import expon, norm, gamma
import matplotlib.pyplot as plt

# Load and clean the data
def load_data(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, dtype=str)
    data.columns = ['observation', 'death_status', 'age_entry', 'age_exit', 'age_diff', 'gender']
    
    # Convert relevant columns to numeric, forcing errors to NaN to handle any irregularities
    data['death_status'] = pd.to_numeric(data['death_status'], errors='coerce')
    data['age_entry'] = pd.to_numeric(data['age_entry'], errors='coerce')
    data['age_exit'] = pd.to_numeric(data['age_exit'], errors='coerce')
    data['age_diff'] = pd.to_numeric(data['age_diff'], errors='coerce')
    data['gender'] = pd.to_numeric(data['gender'], errors='coerce')

    # Drop rows with NaN values
    data = data.dropna()
    return data

# Corrected M-spline basis for testing purposes
def m_spline_basis(t, knots, degree):
    return np.array([t**i for i in range(degree)])  # 'degree' should match the number of gamma_params

# Define the baseline hazard function h0(t) as a sum of M-splines
def baseline_hazard(t, gamma_params, knots, degree):
    m_splines = m_spline_basis(t, knots, degree)
    return np.dot(gamma_params, m_splines)

# Define the cumulative hazard function H0(t) as the integral of h0(t)
def cumulative_hazard(t, gamma_params, knots, degree):
    return np.cumsum(baseline_hazard(t, gamma_params, knots, degree))

# Define the likelihood function for different observation types
def likelihood(data, gamma_params, beta_params, knots, degree):
    likelihood_value = 1.0
    for i, row in data.iterrows():
        li = row['age_entry']
        ti = row['age_exit']
        xi = row[['age_diff', 'gender']].to_numpy()  # Use covariates 'age_diff' and 'gender'
        delta = row['death_status']
        nu = 1 if row['age_entry'] > 0 else 0  # Assumption: truncation if age_entry > 0
        
        h0_ti = baseline_hazard(ti, gamma_params, knots, degree)
        H0_ti = cumulative_hazard(ti, gamma_params, knots, degree)[-1]  # Use last cumulative value
        H0_li = cumulative_hazard(li, gamma_params, knots, degree)[-1]  # Use last cumulative value
        
        # Ensure the hazard and cumulative hazard values are valid
        if np.isnan(h0_ti) or np.isnan(H0_ti) or np.isnan(H0_li):
            return 0  # Return zero likelihood if values are invalid

        if nu == 1 and delta == 1:
            likelihood_value *= h0_ti * np.exp(np.dot(beta_params, xi)) * \
                                np.exp(-(H0_ti - H0_li) * np.exp(np.dot(beta_params, xi)))
        elif nu == 1 and delta == 0:
            likelihood_value *= np.exp(-(H0_ti - H0_li) * np.exp(np.dot(beta_params, xi)))
        elif nu == 0 and delta == 1:
            likelihood_value *= h0_ti * np.exp(np.dot(beta_params, xi)) * \
                                np.exp(-H0_ti * np.exp(np.dot(beta_params, xi)))
        else:
            likelihood_value *= np.exp(-H0_ti * np.exp(np.dot(beta_params, xi)))

        # Prevent extremely small likelihood values
        if likelihood_value < 1e-300:  # Threshold to avoid underflow
            return 0

    return likelihood_value

# Define the prior distributions
def prior(gamma_params, beta_params, eta, sigma):
    prior_gamma = np.prod([expon(scale=1/eta).pdf(g) for g in gamma_params])
    prior_beta = np.prod([norm(0, sigma).pdf(b) for b in beta_params])
    return prior_gamma * prior_beta

# Define the posterior distribution with checks for invalid values
def posterior(data, gamma_params, beta_params, eta, sigma, knots, degree):
    likelihood_val = likelihood(data, gamma_params, beta_params, knots, degree)
    prior_val = prior(gamma_params, beta_params, eta, sigma)
    
    # Check for zero or NaN values in likelihood and prior
    if likelihood_val == 0 or np.isnan(likelihood_val) or prior_val == 0 or np.isnan(prior_val):
        return 0  # If invalid, return zero to avoid division by NaN or zero
    return likelihood_val * prior_val

# Implement the Metropolis-Hastings algorithm for sampling
def metropolis_hastings(data, initial_gamma, initial_beta, eta, sigma, knots, degree, iterations=1000):
    gamma_samples = [initial_gamma]
    beta_samples = [initial_beta]

    for _ in range(iterations):
        # Propose new gamma and beta
        gamma_proposal = np.random.normal(gamma_samples[-1], 0.1)
        beta_proposal = np.random.normal(beta_samples[-1], 0.1)
        
        # Calculate acceptance ratio
        posterior_current = posterior(data, gamma_samples[-1], beta_samples[-1], eta, sigma, knots, degree)
        posterior_proposal = posterior(data, gamma_proposal, beta_proposal, eta, sigma, knots, degree)

        if posterior_current == 0:  # To avoid division by zero
            acceptance_ratio = 0
        else:
            acceptance_ratio = posterior_proposal / posterior_current
        
        # Accept or reject the proposal
        if np.random.rand() < acceptance_ratio:
            gamma_samples.append(gamma_proposal)
            beta_samples.append(beta_proposal)
        else:
            gamma_samples.append(gamma_samples[-1])
            beta_samples.append(beta_samples[-1])
    
    return gamma_samples, beta_samples

# Plot the MCMC results
def plot_mcmc_results(gamma_samples, beta_samples):
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Plot gamma parameter samples
    axs[0].plot(gamma_samples)
    axs[0].set_title('Gamma Parameter MCMC Samples')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Gamma')

    # Plot beta parameter samples
    axs[1].plot(beta_samples)
    axs[1].set_title('Beta Parameter MCMC Samples')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Beta')

    plt.tight_layout()
    plt.show()

# Example usage
file_path = './Channing_20House_20data.txt'
data = load_data(file_path)

# Define initial parameters and hyperparameters
initial_gamma = np.ones(5)
initial_beta = np.ones(2)  # Two covariates: age_diff and gender
knots = np.linspace(min(data['age_entry']), max(data['age_exit']), 5)
degree = len(initial_gamma)  # Degree matches the number of gamma parameters
eta = 1.0  # Set eta (hyperparameter for the gamma prior)
sigma = 1.0  # Set sigma (standard deviation for the normal prior)

# Run Metropolis-Hastings algorithm and plot results
gamma_samples, beta_samples = metropolis_hastings(data, initial_gamma, initial_beta, eta, sigma, knots, degree)
plot_mcmc_results(gamma_samples, beta_samples)
