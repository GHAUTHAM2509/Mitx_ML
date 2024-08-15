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
    
    
    # Initialize the responsibilities (soft counts) matrix
    responsibilities = np.zeros((n, K))
    
    # Compute the responsibilities
    for k in range(K):
        mean = mixture.mu[k]
        var = mixture.var[k]
        weight = mixture.p[k]
        diff = X - mean
        norm_const = 1.0 / np.sqrt((2 * np.pi * var) ** d)
        exponent = -0.5 * np.sum(diff ** 2, axis=1) / var
        
        responsibilities[:, k] = weight * norm_const * np.exp(exponent)
    
    # Normalize responsibilities across all components for each data point
    total_responsibilities = np.sum(responsibilities, axis=1, keepdims=True)
    responsibilities /= total_responsibilities
    
    # Compute the log-likelihood of the data
    log_likelihood = np.sum(np.log(total_responsibilities))
    
    return responsibilities, log_likelihood
    raise NotImplementedError
    



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
    K = post.shape[1]

    # Update the mixing coefficients
    n_k = np.sum(post, axis=0)  # Sum of responsibilities for each component
    p = n_k / n

    # Update the means
    mu = np.dot(post.T, X) / n_k[:, np.newaxis]

    # Update the variances
    var = np.zeros(K)
    for k in range(K):
        diff = X - mu[k]
        var[k] = np.sum(post[:, k] * np.sum(diff ** 2, axis=1)) / (n_k[k] * d)

    # Return the updated Gaussian mixture
    updated_mixture = GaussianMixture(mu, var, p)
    return updated_mixture
    raise NotImplementedError


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
    tol = 1e-6
    max_iters = 100
    prev_log_likelihood = None

    for _ in range(max_iters):
        # E-step: Compute the responsibilities and log-likelihood
        post, log_likelihood = estep(X, mixture)

        # Check for convergence
        if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) <= tol*abs(log_likelihood):
            break

        prev_log_likelihood = log_likelihood

        # M-step: Update the parameters of the mixture model
        mixture = mstep(X, post , mixture)

    return mixture, post, log_likelihood
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
