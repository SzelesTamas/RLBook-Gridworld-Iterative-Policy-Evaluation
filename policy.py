"""In this file we will define the policy of the agent."""
import numpy as np


# we can use more complicated policies, but for now we will use a simple one
def policy(state: int, action: int) -> float:
    """This function returns the probability of taking an action in a given state.

    Args:
        state int: The state of the environment.
        action (int): The action to take.

    Returns:
        float: The probability of taking an action in a given state.
    """
    return 0.25
