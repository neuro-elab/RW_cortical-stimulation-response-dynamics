import numpy as np


# use named function to be able to use in in multiprocessing
def five_param_sigmoid(x, lower_plateau, upper_plateau, midpoint, steepness_raw, shape):
    exponent = steepness_raw * (x - midpoint)
    base = 1 + shape * np.exp(exponent)
    base = np.clip(base, 1e-10, 1e100)  # avoid overflow in power
    denominator = (base) ** (1 / shape)
    return upper_plateau + ((lower_plateau - upper_plateau) / denominator)


# use named function to be able to use in in multiprocessing
def six_param_sigmoid(
    x, lower_plateau, upper_plateau, midpoint, steepness_raw, shape, shape_2
):
    exponent = steepness_raw * (x - midpoint)
    base = 1 + shape_2 * np.exp(exponent)
    base = np.clip(base, 1e-10, 1e100)  # avoid overflow in power
    denominator = (base) ** (1 / shape)
    return upper_plateau + ((lower_plateau - upper_plateau) / denominator)


CURVES = {
    "2P": {
        # linear model
        "name": "2P",
        "function": lambda x, intercept, slope: intercept + slope * x,
        "param_names": ["intercept", "slope"],
        "initial_values": [0.2, 1],
        "bounds": None,
    },
    "3P": {  # TODO check if we need some lower plateau
        # 3P sigmoid
        "name": "3P",
        "function": lambda x, upper_plateau, midpoint, steepness: +(
            (upper_plateau) / (1 + np.exp(-steepness * (x - midpoint)))
        ),
        "param_names": ["upper_plateau", "midpoint", "steepness"],
        "initial_values": [1, 0.4, 5],
        "bounds": None,
    },
    "4P": {
        # classic sigmoid
        "name": "4P",
        "function": lambda x, upper_plateau, lower_plateau, midpoint, steepness: lower_plateau
        + ((upper_plateau - lower_plateau) / (1 + np.exp(-steepness * (x - midpoint)))),
        "param_names": ["upper_plateau", "lower_plateau", "midpoint", "steepness"],
        "initial_values": [1, 0.3, 0.4, 5],
        "bounds": None,
    },
    "5P": {
        # 5 parameter model
        "name": "5P",
        "function": five_param_sigmoid,  # lambda x, lower_plateau, upper_plateau, midpoint, steepness_raw, shape: upper_plateau
        # + (
        #     (lower_plateau - upper_plateau)
        #     / (1 + shape * np.exp(steepness_raw * (x - midpoint))) ** (1 / shape)
        # ),
        "param_names": [
            "lower_plateau",
            "upper_plateau",
            "midpoint",
            "steepness_raw",
            "shape",
        ],
        "initial_values": [0.3, 1, 0.35, 15, 10],
        "bounds": (
            [-np.inf, -np.inf, -np.inf, 0, 0.01],
            [np.inf, np.inf, np.inf, 200, 200],
        ),
    },
}
