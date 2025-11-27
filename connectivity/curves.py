import numpy as np
import scipy


# rational one
# p = expit(z)
# inv_denom = ((1 - p) / (1 + (shape - 1) * p)) ** (1 / shape)
# y = upper_plateau + (lower_plateau - upper_plateau) * inv_denom


# use named funciton to be able to use in in multiprocessing
def four_param_sigmoid(x, lower_plateau, upper_plateau, inflection_point, steepness):
    return lower_plateau + (
        (upper_plateau - lower_plateau)
        / (1 + np.exp(-steepness * (x - inflection_point)))
    )


# use named function to be able to use in in multiprocessing
def five_param_sigmoid(
    x, lower_plateau, upper_plateau, inflection_point, steepness, shape
):

    z = steepness * (x - inflection_point)
    # log(1 + shape * e^z) stably
    log_base = np.logaddexp(0.0, np.log(shape) + z)  # = log(1 + shape*exp(z))

    # inv_denom = exp(-(1/shape) * log_base)  in (0,1]
    log_inv_denom = -(1.0 / shape) * log_base
    inv_denom = np.exp(log_inv_denom)  # safe: exponent <= 0

    return upper_plateau + (lower_plateau - upper_plateau) * inv_denom


Curve = dict[str, object]

CURVES: dict[str, Curve] = {
    "2P": {
        # linear model
        "name": "2P",
        "function": lambda x, intercept, slope: intercept + slope * x,
        "param_names": ["intercept", "slope"],
        "initial_values": [0.2, 1],
        "bounds": None,
    },
    "3P": {
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
        "function": four_param_sigmoid,  # lambda x, lower_plateau, upper_plateau, inflection_point, steepness: lower_plateau
        # + (
        #     (upper_plateau - lower_plateau)
        #     / (1 + np.exp(-steepness * (x - inflection_point)))
        # ),
        "param_names": [
            "lower_plateau",
            "upper_plateau",
            "inflection_point",
            "steepness",
        ],
        "initial_values": [0.3, 1, 0.4, 5],
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
            "steepness",
            "shape",
        ],
        "initial_values": [0.3, 1, 0.35, 15, 10],
        "bounds": (
            [0, 0, -np.inf, 0, 0.01],
            [1, np.inf, np.inf, np.inf, 200],
        ),
    },
}
