"""In this file we will define the policy iteration algorithm."""
import numpy as np
from environment import Environment


def iteration(policy, currentValue: np.array, env: Environment) -> tuple:
    """This function performs one iteration of the policy evaluation algorithm.

    Args:
        policy (function): The policy of the agent.
        currentValue (np.array): The current value function.

    Returns:
        tuple: The new value function and the maximum difference between the old and new value function.
    """
    newValue = np.zeros(env.stateCount)
    for state in range(env.stateCount):
        if state in env.terminalStates:
            continue
        for action in range(len(env.actions)):
            # get the next state and reward
            nextState, reward, _ = env.makeMove(state, action)
            # calculate the new value
            newValue[state] += policy(state, action) * (
                reward + currentValue[nextState]
            )

    return newValue, np.max(np.abs(newValue - currentValue))
