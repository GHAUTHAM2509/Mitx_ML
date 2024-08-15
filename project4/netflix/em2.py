"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K = len(mixture.p)
    
    # Initialize the log responsibilities (soft counts) matrix
    log_responsibilities = np.zeros((n, K))
    
    # Compute the log responsibilities
    for k in range(K):
        mean = mixture.mu[k]
        var = mixture.var[k]
        weight = mixture.p[k]
        
        # Create a mask for non-zero (non-missing) entries
        valid_mask = X != 0
        valid_count = np.sum(valid_mask, axis=1)
        
        # Compute log-probabilities for the Gaussian distribution
        diff = (X - mean) * valid_mask
        log_norm_const = -0.5 * valid_count * np.log(2 * np.pi * var)
        log_exponent = -0.5 * np.sum(diff ** 2, axis=1) / var
        
        # Log responsibility for component k
        log_responsibilities[:, k] = np.log(weight + 1e-16) + log_norm_const + log_exponent
    
    # Compute log-likelihood using logsumexp for numerical stability
    log_total_responsibilities = logsumexp(log_responsibilities, axis=1, keepdims=True)
    log_likelihood = np.sum(log_total_responsibilities)
    
    # Convert log responsibilities to regular responsibilities
    responsibilities = np.exp(log_responsibilities - log_total_responsibilities)
    
    return responsibilities, log_likelihood
    

def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    mu_rev, _, _ = mixture
    K = mu_rev.shape[0]
    
    # Calculate revised pi(j): same expression as in the naive case
    pi_rev = np.sum(post, axis=0)/n
    
    # Create delta matrix indicating where X is non-zero
    delta = X.astype(bool).astype(int)
    
    # Update means only when sum_u(p(j|u)*delta(l,Cu)) >= 1
    denom = post.T @ delta # Denominator (K,d): Only include dims that have information
    numer = post.T @ X  # Numerator (K,d)
    update_indices = np.where(denom >= 1)   # Indices for update
    mu_rev[update_indices] = numer[update_indices]/denom[update_indices] # Only update where necessary (denom>=1)
    
    # Update variances
    denom_var = np.sum(post*np.sum(delta, axis=1).reshape(-1,1), axis=0) # Shape: (K,)
    norms = np.sum(X**2, axis=1)[:,None] + (delta @ mu_rev.T**2) - 2*(X @ mu_rev.T)
    var_rev = np.maximum(np.sum(post*norms, axis=0)/denom_var, min_variance)  
    
    return GaussianMixture(mu_rev, var_rev, pi_rev)
def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log_lh = None
    new_log_lh = None  # Keep track of log likelihood to check convergence
    
    # Start the main loop
    while old_log_lh is None or (new_log_lh - old_log_lh > 1e-6*np.abs(new_log_lh)):
        
        old_log_lh = new_log_lh
        
        # E-step
        post, new_log_lh = estep(X, mixture)
        
        # M-step
        mixture = mstep(X, post, mixture)
            
    return mixture, post, new_log_lh
    raise NotImplementedError



def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    X_pred = X.copy()
    n,d = X.shape
    K = mixture.mu.shape[0]
    
    # Log probabilities for numerical stability
    log_p = np.log(mixture.p + 1e-10)
    log_var = np.log(mixture.var + 1e-10)

    # Initialize the filled matrix with the original data
    X_pred = np.copy(X)
    
    for i in range(n):
        # Identify missing entries (where X is 0)
        missing_mask = X[i] == 0
        observed_mask = ~missing_mask
        
        if np.sum(missing_mask) == 0:
            continue  # Skip if no missing data

        # Calculate log-probabilities for observed data
        log_prob = np.zeros(K)
        for k in range(K):
            log_prob[k] = log_p[k] - 0.5 * np.sum(np.log(2 * np.pi * mixture.var[k]) + 
                                                  (X[i, observed_mask] - mixture.mu[k, observed_mask])**2 / mixture.var[k])

        # Compute responsibilities in log-domain and normalize
        log_resp = log_prob - logsumexp(log_prob)
        resp = np.exp(log_resp)

        # Fill in the missing values using the weighted mean of the Gaussians
        for j in range(d):
            if missing_mask[j]:
                X_pred[i, j] = np.sum(resp * mixture.mu[:, j])
    
    return X_pred
    raise NotImplementedError
