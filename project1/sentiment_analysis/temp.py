import numpy as np

def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the perceptron algorithm. Returns new parameters rather than
    modifying in-place.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns a tuple containing two values:
        the updated feature-coefficient parameter `theta` as a numpy array
        the updated offset parameter `theta_0` as a floating point number
    """
    # Compute the dot product of the current theta and the feature vector
    dot_product = np.dot(current_theta, feature_vector) + current_theta_0
    
    # Check if the feature vector is correctly classified
    if label * dot_product <= 0:
        # Update theta and theta_0
        current_theta = current_theta + label * feature_vector
        current_theta_0 = current_theta_0 + label
    ans = (current_theta, current_theta_0)
    return ans


def get_order(n_samples):
    """
    Helper function to get a permutation order of indices from 0 to n_samples-1
    """
    return np.arange(n_samples)

def perceptron(feature_matrix, labels, T):
    a =1
   
# Given input
feature_matrix = np.array([
    [0.36408373, 0.45931872, 0.25493036, 0.0802452, 0.11372254, 0.06142737, 0.23101311, 0.18812316, -0.33922125, -0.29069062],
    [-0.44030317, 0.20188938, -0.32034618, -0.18573883, 0.03560079, -0.38051959, 0.11301165, 0.4653683, -0.25113617, 0.04280678],
    [-0.17139716, -0.40986839, -0.24796231, -0.37682381, -0.34289676, -0.39789466, -0.32742261, -0.35427073, 0.20715504, -0.26338259],
    [0.10356584, 0.45262288, 0.32812414, 0.35935103, 0.07603167, 0.44369335, -0.48034201, -0.23386943, 0.35248617, -0.14776253],
    [0.20869015, 0.26267048, 0.28632761, 0.09730821, 0.20717326, 0.41067929, 0.01551328, 0.41018638, -0.25000273, -0.08491717]
])
labels = np.array([-1, 1, 1, -1, 1])
T = 600

# Run the perceptron algorithm
print(perceptron(feature_matrix, labels, T))


