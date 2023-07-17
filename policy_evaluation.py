"""In this file we will define the policy iteration algorithm."""
import numpy as np


def iteration(policy: function, currentValue: np.array, env) -> np.array:
    """This function performs one iteration of the policy evaluation algorithm.

    Args:
        policy (function): The policy of the agent.
        currentValue (np.array): The current value function.

    Returns:
        np.array: The new estimated value function.
    """
