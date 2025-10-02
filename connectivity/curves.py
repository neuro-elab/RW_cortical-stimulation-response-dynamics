import numpy as np


# use named function to be able to use in in multiprocessing
def five_param_sigmoid(
    x, lower_plateau, upper_plateau, inflection_point, steepness, shape
):
    exponent = steepness * (x - inflection_point)
    base = 1 + shape * np.exp(exponent)
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
        "function": lambda x, upper_plateau, inflection_point, steepness: +(
            (upper_plateau) / (1 + np.exp(-steepness * (x - inflection_point)))
        ),
        "param_names": ["upper_plateau", "inflection_point", "steepness"],
        "initial_values": [1, 0.4, 5],
        "bounds": (
            [0, -np.inf, -np.inf],
            [np.inf, np.inf, np.inf],
        ),
    },
    "4P": {
        # classic sigmoid
        "name": "4P",
        "function": lambda x, upper_plateau, lower_plateau, inflection_point, steepness: lower_plateau
        + (
            (upper_plateau - lower_plateau)
            / (1 + np.exp(-steepness * (x - inflection_point)))
        ),
        "param_names": [
            "upper_plateau",
            "lower_plateau",
            "inflection_point",
            "steepness",
        ],
        "initial_values": [1, 0.3, 0.4, 5],
        "bounds": (
            [0, 0, -np.inf, -np.inf],
            [1, np.inf, np.inf, np.inf],
        ),
    },
    "5P": {
        # 5 parameter model
        "name": "5P",
        "function": five_param_sigmoid,  # lambda x, lower_plateau, upper_plateau, inflection_point, steepness_raw, shape: upper_plateau
        # + (
        #     (lower_plateau - upper_plateau)
        #     / (1 + shape * np.exp(steepness_raw * (x - inflection_point))) ** (1 / shape)
        # ),
        "param_names": [
            "lower_plateau",
            "upper_plateau",
            "inflection_point",
            "steepness_raw",
            "shape",
        ],
        "initial_values": [0.3, 1, 0.35, 15, 10],
        "bounds": (
            [0, 0, -np.inf, 0, 0.01],
            [1, np.inf, np.inf, 200, 200],
        ),
    },
}
