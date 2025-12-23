import json
import math
import os
import sys
from dotenv import load_dotenv
from filelock import FileLock
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

from connectivity.analyze import (
    calculate_model_performance,
    fallback_fit_curve,
    fit_curve,
)

from connectivity.curves import CURVES
from connectivity.load import MultipleHDFResponseLoader, get_h5_names_of_patient

load_dotenv()

r_squared_theshold = 0.6
n_replications = 12
base_path = os.getenv("BASE_PATH_PAPER", "/default/path")


# base_path = "D:/data"
# patients_id = ["EL022", "EL027", "EL019", "EL026"]  # EL019
patients_id = [arg for arg in sys.argv[1:]]
print(f"Patients: {patients_id}")
out_path = "output/curve_fitting"
response_file = "output/significant_responses/response_channels_lf.json"
curve_fitting_file = "output/curve_fitting/curve_fitting_lf.json"


curves = [CURVES["2P"], CURVES["3P"], CURVES["4P"], CURVES["5P"]]
MAIN_CURVE_NAME = "5P"
SECONDARY_CURVE_NAME = "4P"

LOSSES = ["linear"]  # , "soft_l1", "cauchy"]
R_SQUARED_THRESHOLD = 0.6
MAX_ITERATIONS = 20000

with open(f"{out_path}/params_{'_'.join(patients_id)}.json", "w") as f:
    json.dump(
        {
            "r_squared_theshold": r_squared_theshold,
            "n_replications": n_replications,
            "base_path": base_path,
            "patients_id": patients_id,
            "curves": [curve["name"] for curve in curves],
            "losses": LOSSES,
            "main_curve": MAIN_CURVE_NAME,
            "r_squared_threshold": R_SQUARED_THRESHOLD,
            "max_iterations": MAX_ITERATIONS,
            "secondary_curve": SECONDARY_CURVE_NAME,
        },
        f,
        indent=4,
    )

## LOAD RESPONSE CHANNEL FILE

result_df = pd.read_json(response_file)

# add new cols
for curve in curves:
    cols = ["r_squared", "params", "d_aic"]
    dtypes = [float, object, float]
    for dtype, col in zip(dtypes, cols):
        if f"{curve['name']}_{col}" not in result_df.columns:
            result_df[f"{curve['name']}_{col}"] = pd.Series(dtype=dtype)

results = {}


for patient_id in patients_id:
    names_h5 = get_h5_names_of_patient(base_path, patient_id, protocol="CR")
    patient_df = result_df[result_df["patient_id"] == patient_id]

    path_lookup = f"{base_path}/{patient_id}/Electrodes/Lookup.xlsx"
    paths_h5 = [f"{base_path}/{patient_id}/Electrophy/{name}.h5" for name in names_h5]
    paths_logs = [f"{base_path}/{patient_id}/out/{name}_logs.csv" for name in names_h5]

    mrl = MultipleHDFResponseLoader(
        paths_h5=paths_h5,
        paths_logs=paths_logs,
        recording_names=names_h5,
        path_lookup=path_lookup,
    )

    logs = mrl.get_logs()
    intensities = logs[logs["type"] == "CR_IO"]["Int_prob"].drop_duplicates().tolist()
    intensities.sort()
    intensities.insert(0, 0)
    intensities = np.array(intensities)
    norm_intensities = intensities / np.max(intensities)

    # results[patient_id] = {}
    stim_channel_names = patient_df["stim_channel_name"].unique()
    for stim_channel_name in stim_channel_names:
        stim_channel_df = patient_df[
            patient_df["stim_channel_name"] == stim_channel_name
        ]

        n_connections = stim_channel_df["is_significant"].value_counts().get(True, 0)
        n_response_channels = len(stim_channel_df)

        n_cols = 6
        n_plots = n_connections
        n_rows = np.max([math.ceil(n_plots / n_cols), 1])
        fig = plt.figure(figsize=(25, n_rows * 4))
        gs = GridSpec(n_rows, n_cols, figure=fig)

        n_r_squared_significant = 0

        params_across = {}
        for curve in curves:
            params_across[curve["name"]] = []

        connection_df = stim_channel_df[stim_channel_df["is_significant"] == True]
        rows = []
        for i, (df_i, row) in enumerate(connection_df.iterrows()):
            channel_path = row["response_channel_path"]
            channel_name = row["response_channel_name"]
            res_row = {
                "patient_id": patient_id,
                "stim_channel_name": stim_channel_name,
                "response_channel_name": channel_name,
            }
            destrieux_label = mrl.get_destrieux_labels_from_names(
                channel_names=[channel_name], short_form=True
            )[0]

            norm_med_lls = np.array(row["norm_med_lls"])

            plot_row, plot_col = divmod(i, n_cols)  # Determine row and column
            ax = fig.add_subplot(gs[plot_row, plot_col])

            ax.scatter(
                norm_intensities, norm_med_lls, color="black", label="Median", s=5
            )

            colors = [
                "blue",
                "orange",
                "green",
                "purple",
                "orange",
                "green",
                "purple",
                "orange",
                "green",
                "purple",
            ]
            # linestyles = [
            #     "solid",
            #     "solid",
            #     "solid",
            #     "solid",
            #     "dashed",
            #     "dashed",
            #     "dashed",
            #     "dotted",
            #     "dotted",
            #     "dotted",
            # ]
            aic_main = np.nan
            aic_secondary = np.nan
            curve_fittings = {}
            for j, curve in enumerate(curves):
                linestyles = ["solid", "dashed", "dotted"]
                for k, loss in enumerate(LOSSES):
                    try:
                        if curve["name"] == "5P":
                            main_initial_values = {"shape": 1}
                            params, nfev = fallback_fit_curve(
                                main_curve=CURVES["5P"],
                                fallback_curve=CURVES["4P"],
                                x=norm_intensities,
                                y=norm_med_lls,
                                loss=loss,
                                max_iterations=MAX_ITERATIONS,
                                main_initial_values=main_initial_values,
                            )
                        else:
                            params, nfev = fit_curve(
                                curve_function=curve["function"],
                                x=norm_intensities,
                                y=norm_med_lls,
                                initial_values=curve["initial_values"],
                                bounds=curve["bounds"],
                                loss=loss,
                                full_output=True,
                                max_iterations=MAX_ITERATIONS,
                            )

                        x_fit = np.linspace(0, 1, 1000)
                        y_fit = curve["function"](x_fit, *params)
                        exi = np.trapezoid(y_fit, x_fit)
                        y_pred = curve["function"](norm_intensities, *params)

                        # num params +1 for variance of errors: https://en.wikipedia.org/wiki/Akaike_information_criterion#Counting_parameters
                        performance_dict = calculate_model_performance(
                            y=norm_med_lls,
                            y_pred=y_pred,
                            num_params=len(curve["initial_values"]) + 1,
                        )

                        if curve["name"] == MAIN_CURVE_NAME:
                            aic_main = performance_dict["dAIC"]
                            if performance_dict["r_squared"] > R_SQUARED_THRESHOLD:
                                n_r_squared_significant += 1
                            else:
                                ax.set_facecolor("#f8ffc9")
                        elif curve["name"] == SECONDARY_CURVE_NAME:
                            aic_secondary = performance_dict["dAIC"]
                        res_row.update(
                            {
                                f"{curve['name']}_params": params,
                                f"{curve['name']}_r_squared": performance_dict[
                                    "r_squared"
                                ],
                                f"{curve['name']}_d_aic": performance_dict["dAIC"],
                                f"{curve['name']}_d_aicc": performance_dict["dAICc"],
                                f"{curve['name']}_d_bic": performance_dict["dBIC"],
                                f"{curve['name']}_nfev": nfev,
                                f"{curve['name']}_exi": exi,
                            }
                        )

                        ax.plot(
                            x_fit,
                            y_fit,
                            label=f"{curve['name']}/{loss}: {performance_dict['r_squared']: .2f}, n_iter={nfev}",
                            color=colors[j],
                            linestyle=linestyles[k],
                        )
                        ax.set_xlabel("Intensity")
                        ax.set_ylabel("Normalized LL")
                        ax.set_ylim(-0.1, 1.1)
                        ax.set_xlim(0, 1)
                    except Exception as e:
                        print(
                            f"{channel_name}/{curve['name']}: Optimization failed.", e
                        )
                        if curve["name"] == MAIN_CURVE_NAME:
                            ax.set_facecolor("#f7c1c1")

                            # try:
                            #     new_init = np.append(res_row["4P_params"], 1)
                            #     params, nfev = fit_curve(
                            #         curve_function=curve["function"],
                            #         x=norm_intensities,
                            #         y=norm_med_lls,
                            #         initial_values=new_init,
                            #         bounds=curve["bounds"],
                            #         loss=loss,
                            #         full_output=True,
                            #         max_iterations=MAX_ITERATIONS + 20000,
                            #     )
                            #     print(f"{channel_name} SUCCESS WITH 4P INIT")
                            # except RuntimeError as e:
                            #     print(f"{channel_name} FAILED WITH 4P INIT")

            rows.append(res_row)

            ax.legend()

            ax.set_title(
                f"{channel_name} ({destrieux_label}):\nSNR {row['snr']:.2f}"  # FIXME dAIC {aic_main-aic_secondary: .2f} ({MAIN_CURVE_NAME}/{SECONDARY_CURVE_NAME})"
            )

        plt.suptitle(
            f"{base_path} - Stimulation response curves for {patient_id} - {stim_channel_name}\nn_replications={n_replications}"
            + f"\nn_responses={n_response_channels}, n_connections={n_connections}, n_sigmoidal={n_r_squared_significant} ({n_r_squared_significant/n_connections if n_connections else 0:.2%})"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{out_path}/curve_fittings_{patient_id}_{stim_channel_name}.png")
        plt.close()

        curve_fitting_df = pd.DataFrame(rows)

        lock_path = curve_fitting_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(curve_fitting_file):
                df = pd.read_json(curve_fitting_file, orient="records")
                # Remove old entries
                mask = (df["patient_id"] == patient_id) & (
                    df["stim_channel_name"] == stim_channel_name
                )

                # Drop old entries
                df = df[~mask]

                # Combine
                curve_fitting_df = pd.concat([df, curve_fitting_df], ignore_index=True)

            curve_fitting_df.to_json(curve_fitting_file, orient="records", indent=4)
