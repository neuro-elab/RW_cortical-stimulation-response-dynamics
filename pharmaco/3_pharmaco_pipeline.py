import math
import os
import sys
from filelock import FileLock
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

from connectivity.analyze import (
    calculate_continuous_line_length,
    calculate_continuous_line_length,
    calculate_ll_baseline,
    calculate_ll_baseline,
    calculate_model_performance,
    calculate_stimulation_response_curves,
    fallback_fit_curve,
    filter_logs,
    find_max_n_replications,
    fit_curve,
    normalize_ll_values,
    significant_exi_difference_testing,
)
from connectivity.curves import CURVES
from connectivity.load import (
    MultipleHDFResponseLoader,
    get_h5_names_of_patient,
    parsed_list_to_numpy_array,
)

base_path = "D:/data_paper"
out_path = "output/pharmaco"
RESPONSES_FILE = f"{out_path}/response_channels_lf.json"
n_replications = 3
SIGNIFICANCE_LEVEL = 0.05
CURVE = CURVES["5P"]
FALLBACK_CURVE = CURVES["4P"]
N_SURROGATES = 100
MAX_ITERATIONS = 20000
MAX_ITERATIONS_SURR = 1000
R_SQUARED_THRESHOLD = 0.6

x_fit = np.linspace(0, 1, 1000)

# load
patient_ids = sys.argv[1:]  # ["EL010"]  # , "EL014"]
stim_blocks = {"EL010": {"off": 1, "on": 3}, "EL014": {"off": 1, "on": 2}}

INTENSITIES = [0, 0.2, 0.6, 0.8, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12]
norm_intensities = np.array(INTENSITIES) / max(INTENSITIES)


responses_df = pd.read_json(RESPONSES_FILE, orient="records")

for patient_id in patient_ids:
    result_rows = []
    print(f"{pd.Timestamp.now()}: Processing patient {patient_id}")
    patient_responses = responses_df[responses_df["patient_id"] == patient_id]

    stim_channel_names = patient_responses["stim_channel_name"].unique().tolist()
    for stim_channel_name in stim_channel_names:
        significant_response_channels = patient_responses[
            (patient_responses["combined_significant"] == True)
            & (patient_responses["stim_channel_name"] == stim_channel_name)
        ][
            "response_channel_name"
        ].unique()  # "OR logic"

        n_cols = 6
        n_plots = len(significant_response_channels)

        n_rows = math.ceil(n_plots / n_cols)
        fig = plt.figure(figsize=(35, 4.5 * n_rows), constrained_layout=True)
        if n_plots == 0:
            continue
        gs = GridSpec(
            n_rows,
            n_cols,
            figure=fig,
        )

        for i, response_channel_name in enumerate(significant_response_channels):
            print(
                f"  Plotting stim: {stim_channel_name}, response: {response_channel_name}"
            )
            channel_paths = [
                f"{base_path}/{patient_id}/responses/lf/{stim_channel_name}/{response_channel_name}.h5"
            ]

            off_row = responses_df[
                (responses_df["patient_id"] == patient_id)
                & (responses_df["stim_channel_name"] == stim_channel_name)
                & (responses_df["response_channel_name"] == response_channel_name)
                & (responses_df["condition"] == "off")
            ].squeeze()

            on_row = responses_df[
                (responses_df["patient_id"] == patient_id)
                & (responses_df["stim_channel_name"] == stim_channel_name)
                & (responses_df["response_channel_name"] == response_channel_name)
                & (responses_df["condition"] == "on")
            ].squeeze()

            row, col = divmod(i, n_cols)
            ax = fig.add_subplot(gs[row, col])

            # ll dots
            on_norm_ll_values = parsed_list_to_numpy_array(on_row["norm_ll_values"])
            on_norm_med_ll_values = np.nanmedian(on_norm_ll_values, axis=1)
            off_norm_ll_values = parsed_list_to_numpy_array(off_row["norm_ll_values"])
            off_norm_med_ll_values = np.nanmedian(off_norm_ll_values, axis=1)

            for i_rep in range(on_norm_ll_values.shape[1]):
                ax.scatter(
                    norm_intensities,
                    on_norm_ll_values[:, i_rep],
                    c="green",
                    alpha=0.3,
                    # label="ON Replications" if i_rep == 0 else None,
                    marker="o",
                )
                ax.scatter(
                    norm_intensities,
                    off_norm_ll_values[:, i_rep],
                    c="red",
                    alpha=0.3,
                    # label="OFF Replications" if i_rep == 0 else None,
                    marker="o",
                )
            ax.scatter(
                norm_intensities,
                on_norm_med_ll_values,
                c="green",
                label="ON",
            )
            ax.scatter(
                norm_intensities,
                off_norm_med_ll_values,
                c="red",
                label="OFF",
            )

            # check if we find significant difference
            delta_empirical_exis_null, delta_exis_null = (
                significant_exi_difference_testing(
                    norm_ll_values_1=on_norm_ll_values,
                    norm_ll_values_2=off_norm_ll_values,
                    n_surrogates=N_SURROGATES,
                    intensities=norm_intensities,
                    curve=CURVE,
                    # ax=ax_hist,
                    parallelize=False,
                    max_iterations=MAX_ITERATIONS_SURR,
                    return_null_distributions=True,
                )
            )

            on_empirical_exi = np.trapezoid(on_norm_med_ll_values, norm_intensities)
            off_empirical_exi = np.trapezoid(off_norm_med_ll_values, norm_intensities)

            delta_empirical_exi = on_empirical_exi - off_empirical_exi
            surr_p_value_empirical = (
                np.sum(delta_empirical_exis_null >= np.abs(delta_empirical_exi)) + 1
            ) / (N_SURROGATES + 1)

            try:
                main_initial_values = {"shape": 1}
                on_params, _ = fallback_fit_curve(
                    main_curve=CURVE,
                    fallback_curve=FALLBACK_CURVE,
                    x=norm_intensities,
                    y=on_norm_med_ll_values,
                    max_iterations=MAX_ITERATIONS,
                    main_initial_values=main_initial_values,
                )
                # on_params = fit_curve(
                #     curve_function=CURVE["function"],
                #     x=norm_intensities,
                #     y=on_norm_med_ll_values,
                #     initial_values=CURVE["initial_values"],
                #     bounds=CURVE["bounds"],
                #     max_iterations=MAX_ITERATIONS,
                # )
                on_y_fit = CURVE["function"](x_fit, *on_params)
                on_y_pred = CURVE["function"](norm_intensities, *on_params)
                on_exi = np.trapezoid(on_y_fit, x_fit)

                # off_params = fit_curve(
                #     curve_function=CURVE["function"],
                #     x=norm_intensities,
                #     y=off_norm_med_ll_values,
                #     initial_values=CURVE["initial_values"],
                #     bounds=CURVE["bounds"],
                #     max_iterations=MAX_ITERATIONS,
                # )
                on_performance_dict = calculate_model_performance(
                    y=on_norm_med_ll_values,
                    y_pred=on_y_pred,
                    num_params=len(CURVE["initial_values"]) + 1,
                )
                on_r_squared = on_performance_dict["r_squared"]

                off_params, _ = fallback_fit_curve(
                    main_curve=CURVE,
                    fallback_curve=FALLBACK_CURVE,
                    x=norm_intensities,
                    y=off_norm_med_ll_values,
                    max_iterations=MAX_ITERATIONS,
                    main_initial_values=main_initial_values,
                )

                off_y_fit = CURVE["function"](x_fit, *off_params)
                off_y_pred = CURVE["function"](norm_intensities, *off_params)
                off_performance_dict = calculate_model_performance(
                    y=off_norm_med_ll_values,
                    y_pred=off_y_pred,
                    num_params=len(CURVE["initial_values"]) + 1,
                )
                off_r_squared = off_performance_dict["r_squared"]
                off_exi = np.trapezoid(off_y_fit, x_fit)

                delta_exi = on_exi - off_exi  # decrease in excitability = negative

                # calculate p-value
                surr_p_value = (np.sum(delta_exis_null >= np.abs(delta_exi)) + 1) / (
                    N_SURROGATES + 1
                )

                if surr_p_value < SIGNIFICANCE_LEVEL:
                    if delta_exi < 0:
                        ax.set_facecolor("#f3aeae")
                    else:
                        ax.set_facecolor("#b5f3ba")

                ax.plot(
                    x_fit,
                    on_y_fit,
                    label=f"ON Fit, ExI: {on_exi:.2f}",
                    color="darkgreen",
                    linestyle="--",
                )
                ax.plot(
                    x_fit,
                    off_y_fit,
                    label=f"OFF Fit, ExI: {off_exi:.2f}",
                    color="darkred",
                    linestyle="--",
                )

            except Exception as e:
                delta_exi = np.nan
                on_r_squared = 0
                off_r_squared = 0
                print(
                    f"{patient_id}: {stim_channel_name}/{response_channel_name}: Optimization failed. {e}"
                )
            low_r_squared = (
                on_r_squared < R_SQUARED_THRESHOLD
                or off_r_squared < R_SQUARED_THRESHOLD
            )
            if low_r_squared:
                ax.axhspan(
                    -0.1,
                    1.4,
                    transform=ax.transAxes,
                    facecolor="none",  # keep the color from set_facecolor
                    hatch="xx",  # choose any pattern
                    edgecolor="lightgray",  # must not be "none", otherwise hatch won't draw
                    linewidth=0.0,
                    zorder=-1,
                )
            result_rows.append(
                {
                    "patient_id": patient_id,
                    "stim_channel_name": stim_channel_name,
                    "response_channel_name": response_channel_name,
                    "delta_exi": delta_exi,
                    "delta_empirical_exi": delta_empirical_exi,
                    "p_empirical": surr_p_value_empirical,
                    "p_surrogate": surr_p_value,
                    "on_r_squared": on_r_squared,
                    "off_r_squared": off_r_squared,
                }
            )

            ax.set_title(
                f"{patient_id}: {stim_channel_name}/{response_channel_name}\n"
                + f"ON R²: {on_r_squared:.2f}, OFF R²: {off_r_squared:.2f} \n"
                + f"Delta ExI: {delta_exi:.2f}, p_emp: {surr_p_value_empirical:.3f}, p_surr: {surr_p_value:.3f}"
            )
            ax.set_ylim(-0.1, 1.4)

        plt.savefig(f"{out_path}/result_{patient_id}_{stim_channel_name}.png")
        plt.close()

    results_df = pd.DataFrame(result_rows)

    json_path = f"{out_path}/pharmaco_result_lf.json"
    lock_path = json_path + ".lock"

    with FileLock(lock_path):
        if os.path.exists(json_path):
            df = pd.read_json(json_path, orient="records")
            # Remove old entries
            mask = df["patient_id"] == patient_id

            # Drop old entries
            df = df[~mask]
            # Combine
            df = pd.concat([df, results_df], ignore_index=True)
        else:
            df = pd.DataFrame(result_rows)

        df.to_json(json_path, orient="records", indent=4)
