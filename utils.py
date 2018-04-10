"""
This file contains general helper functions for the Risk environment
"""
import numpy as np


def validate_q_func_for_argmax(q_func, valid_mask):
    """
    Returns a q function with negative infinity in indices zero'd by valid_mask
    :param q_func: float vector to correct for validation
    :param valid_mask: int vector of valid actions (binary)
    :return valid_q_func: vector of size as q_func, with -inf in places set by valid_mask
    """
    nA = len(valid_mask)
    valid_q_func = np.zeros(nA)

    any_valid_action = False

    if not (len(q_func) == nA):
        print("Q function and mask different sizes")
        return -1
    for ii in range(nA):
        if valid_mask[ii] == 1:
            any_valid_action = True
            valid_q_func[ii] = q_func[ii]
        else:
            valid_q_func[ii] = float("-inf")

    if any_valid_action == False:
        print("Warning: no valid actions")
        return -1

    return valid_q_func



def choose_by_weight(q_func):
    """
    Returns an index in q_func (action choice) based on relative weights in the q function
    """
    min_q = min(q_func)
    sum_q = np.sum(q_func)
    if (sum_q == 0):
        print("WARNING: weighted choices sum to zero, sampling randomly")
        return np.random.randint(0, len(q_func))

    if (min_q < 0):
        print("WARNING: weighted choices non-negative, sampling incorrect")
        q_func = np.array(q_func)  # In case passed a list, which doesn't support '-=''
        q_func -= min_q

    probs = q_func/np.sum(q_func)
    choice = np.random.choice(a=range(len(q_func)), size=1, p=probs)

    return choice[0]


def epsilon_greedy_valid(q_func, valid_mask, epsilon):
    """
    Returns an epsilon greedy action from a subset of function defined by mask
    Only chooses valid actions as specified by the mask
    :param q_func: float vector to return argmax in greedy case
    :param valid_mask: int vector of valid actions
    :param epsilon: probability under which to choose non-greedily
    :return arg: int choice
    """
    nA = len(valid_mask)
    if not (len(q_func) == nA):
        print("Q function and mask different sizes")
        return -1
    eps_choices = np.sum(valid_mask) - 1

    valid_q_func = []
    valid_q_to_orig_q_map = []
    for ii in range(nA):
        if valid_mask[ii] == 1:
            valid_q_func.append(q_func[ii])
            valid_q_to_orig_q_map.append(ii)

    if len(valid_q_func) == 0:
        print("No valid actions")
        return -1

    # print(valid_q_func)
    # print(valid_q_to_orig_q_map)
    valid_action = epsilon_greedy(valid_q_func, epsilon)
    # print(valid_action)
    action = valid_q_to_orig_q_map[valid_action]

    return action


def epsilon_greedy(q_func, epsilon):
    """
    Defines a policy which acts greedily except for epsilon exceptions
    :param q_func: q function returned by an attack network
    :param epsilon: the threshold value
    :return index: int the index of the corresponding action
    """

    eps_choices = len(q_func) - 1
    if eps_choices == 0:
        return -1

    choice = np.random.uniform()
    max_action = np.argmax(q_func)

    # print(choice)
    # print("Max action is {}".format(max_action))

    if choice > epsilon:
        return max_action
    else:
        eps_slice = epsilon/eps_choices
        for act_slice in range(eps_choices):
            # print(eps_slice*(1+act_slice))
            if choice < (eps_slice*(1+act_slice)):
                action = act_slice
                break


    if action >= max_action:  # Increment if past max_action
        action += 1

    return action