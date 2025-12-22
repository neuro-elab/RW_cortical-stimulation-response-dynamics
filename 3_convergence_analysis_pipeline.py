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

from connectivity.analyze import calculate_model_performance, fit_curve

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
out_path = "output/convergence_analysis"
response_file = "output/significant_responses/response_channels_lf.json"
convergence_analysis_file = "output/convergence_analysis/convergence_analysis_lf.json"
curve_fitting_file = "output/curve_fitting/curve_fitting_lf.json"


LOSSES = ["linear", "soft_l1"]  # , "cauchy"]
CURVE = CURVES["5P"]
MAX_ITERATIONS = 20000
x_fit = np.linspace(0, 1, 1000)

with open(f"{out_path}/params_{'_'.join(patients_id)}.json", "w") as f:
    json.dump(
        {
            "r_squared_theshold": r_squared_theshold,
            "n_replications": n_replications,
            "base_path": base_path,
            "patients_id": patients_id,
            "losses": LOSSES,
            "max_iterations": MAX_ITERATIONS,
        },
        f,
        indent=4,
    )

## LOAD RESPONSE CHANNEL FILE

result_df = pd.read_json(response_file)
curve_fitting_df = pd.read_json(curve_fitting_file)

# add new cols
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

            linestyles = ["solid", "dashed", "dotted"]
            for k, loss in enumerate(LOSSES):
                curve_fitting_row = curve_fitting_df[
                    (curve_fitting_df["patient_id"] == patient_id)
                    & (curve_fitting_df["stim_channel_name"] == stim_channel_name)
                    & (curve_fitting_df["response_channel_name"] == channel_name)
                ].iloc[0]
                initial_values_list = [
                    CURVE["initial_values"],
                    curve_fitting_row["4P_params"] + [1],
                ]
                initial_value_names = ["Ordinary", "4P"]
                # plot 4P initial fit
                if loss == LOSSES[0]:
                    ax.plot(
                        x_fit,
                        CURVE["function"](
                            x_fit, *(curve_fitting_row["4P_params"] + [1])
                        ),
                        label=f"4P init",
                        color="gray",
                        linestyle="dashdot",
                    )
                for j, initial_values in enumerate(initial_values_list):
                    try:
                        params, nfev = fit_curve(
                            curve_function=CURVE["function"],
                            x=norm_intensities,
                            y=norm_med_lls,
                            initial_values=initial_values,
                            bounds=CURVE["bounds"],
                            loss=loss,
                            full_output=True,
                            max_iterations=MAX_ITERATIONS,
                        )

                        y_fit = CURVE["function"](x_fit, *params)
                        y_pred = CURVE["function"](norm_intensities, *params)

                        # num params +1 for variance of errors: https://en.wikipedia.org/wiki/Akaike_information_criterion#Counting_parameters
                        performance_dict = calculate_model_performance(
                            y=norm_med_lls,
                            y_pred=y_pred,
                            num_params=len(CURVE["initial_values"]) + 1,
                        )

                        res_row.update(
                            {
                                f"{initial_value_names[j]}_init_{loss}_5P_params": params,
                                f"{initial_value_names[j]}_init_{loss}_5P_r_squared": performance_dict[
                                    "r_squared"
                                ],
                                f"{initial_value_names[j]}_init_{loss}_5P_d_aic": performance_dict[
                                    "dAIC"
                                ],
                                f"{initial_value_names[j]}_init_{loss}_5P_nfev": nfev,
                            }
                        )

                        ax.plot(
                            x_fit,
                            y_fit,
                            label=f"{initial_value_names[j]}-5P/{loss}:\n{performance_dict['r_squared']: .2f}, n_iter={nfev}",
                            color=colors[j],
                            linestyle=linestyles[k],
                        )
                        ax.set_xlabel("Intensity")
                        ax.set_ylabel("Normalized LL")
                        ax.set_ylim(-0.1, 1.1)
                        ax.set_xlim(0, 1)
                    except Exception as e:
                        print(
                            f"{channel_name}/{initial_value_names[j]}_init_{loss}_5P/{loss}: Optimization failed. {e}"
                        )
                        ax.set_facecolor("#f7c1c1")
                        res_row.update(
                            {
                                f"{initial_value_names[j]}_init_{loss}_5P_nfev": -1,
                            }
                        )
            rows.append(res_row)

            ax.legend(fontsize=8)

            ax.set_title(f"{channel_name} ({destrieux_label}):\nSNR {row['snr']:.2f}")

        plt.suptitle(
            f"{base_path} - Stimulation response curves for {patient_id} - {stim_channel_name}\nn_replications={n_replications}"
            + f"\nn_responses={n_response_channels}, n_connections={n_connections}"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(
            f"{out_path}/convergence_analysis_{patient_id}_{stim_channel_name}.png"
        )
        plt.close()

        convergence_analysis_df = pd.DataFrame(rows)

        lock_path = convergence_analysis_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(convergence_analysis_file):
                df = pd.read_json(convergence_analysis_file, orient="records")
                # Remove old entries
                mask = (df["patient_id"] == patient_id) & (
                    df["stim_channel_name"] == stim_channel_name
                )

                # Drop old entries
                df = df[~mask]

                # Combine
                convergence_analysis_df = pd.concat(
                    [df, convergence_analysis_df], ignore_index=True
                )

            convergence_analysis_df.to_json(
                convergence_analysis_file, orient="records", indent=4
            )
