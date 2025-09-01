import json
import sys
import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from filelock import FileLock


from connectivity.load import (
    MultipleHDFResponseLoader,
    get_h5_names_of_patient,
    parsed_list_to_numpy_array,
)
from connectivity.analyze import (
    bootstrap_curve_fitting,
    evaluate_bootstrap_result,
    find_max_n_replications,
    filter_logs,
    fit_curve,
    normalize_ll_values,
    subset_bootstrap_curve_fitting,
)
from connectivity.plot import plot_curve_fittings
from connectivity.enums import SleepStage
from connectivity.curves import CURVES

if __name__ == "__main__":
    load_dotenv()

    DATA_PAPER = True

    if DATA_PAPER:
        BASE_PATH = os.getenv("BASE_PATH_PAPER", "/default/path")
    else:
        BASE_PATH = os.getenv("BASE_PATH", "/default/path")
    # base_path = "D:/data"
    PATIENTS_ID = [arg for arg in sys.argv[1:]]  # ["EL027", "EL019", "EL022", "EL026"]
    print(f"Patients: {PATIENTS_ID}")

    OUT_PATH = (
        "output/paper/bootstrap_curve_analysis"
        if DATA_PAPER
        else "output/bootstrap_curve_analysis"
    )

    CURVE = CURVES["5P"]
    STORE_ALL_DATA = True

    N_BOOTSTRAP_GROUND_TRUTH = 500
    N_BOOTSTRAP_SUBSETS = 500
    R_SQUARED_THRESHOLD = 0.6
    N_REPLICATIONS_ORIGINAL = 12
    BOOTSTRAP_PERCENTILES = [2.5, 97.5]  #
    CURVE_FITTING_RESULT_FILE = (
        f"output/paper/curve_fitting/curve_fitting_lf.json"
        if DATA_PAPER
        else f"output/curve_fitting/curve_fitting_lf.json"
    )
    SURROGATE_RESULT_FILE = (
        f"output/paper/significant_responses/response_channels_lf.json"
        if DATA_PAPER
        else f"output/significant_responses/response_channels_lf.json"
    )
    CLEAN_DATA = True
    CLEAN_DATA_FILE = (
        ("out/clean_CR_IO/bad_responses_dict.json")
        if DATA_PAPER
        else ("out/clean/bad_responses_dict.json")
    )
    SLEEP_STAGES = [SleepStage.AWAKE, SleepStage.QWAKE]
    LOSS = "linear"
    MAX_ITERATIONS = 1000
    INCLUDE_REPLICATIONS_LIST = [9, 6, 5, 4, 3, 2, 1]
    USE_CACHE_GT = False
    USE_CACHE_SUBSET = False
    USE_MIN_NORMALIZATION = True

    BOOT_R_SQUARED_THRESHOLD = 0.6

    EXCLUDE_INTENSITIES_MATRICES = {
        17: [
            [],  # all intensities
            [1],
            [1, 3],
            [1, 3, 5],
            [1, 3, 5, 7],
            [1, 3, 5, 7, 12],
            [1, 3, 5, 7, 12, 14],
            [1, 3, 5, 7, 12, 14, 10],
            [1, 3, 5, 7, 12, 14, 10, 8],
            [1, 3, 5, 7, 12, 14, 10, 15],
            [1, 3, 5, 7, 12, 14, 10, 8, 15],
        ],
        18: [
            [],  # all intensities
            [2],
            [2, 1],
            [2, 1, 4],
            [2, 1, 4, 6],
            [2, 1, 4, 6, 8],
            [2, 1, 4, 6, 8, 13],
            [2, 1, 4, 6, 8, 13, 15],
            [2, 1, 4, 6, 8, 13, 15, 11],
            [2, 1, 4, 6, 8, 13, 15, 11, 9],
            [2, 1, 4, 6, 8, 13, 15, 11, 16],
            [2, 1, 4, 6, 8, 13, 15, 11, 9, 16],
        ],
    }

    import multiprocessing as mp

    mp.freeze_support()

    with open(f"{OUT_PATH}/params_{'_'.join(PATIENTS_ID)}.json", "w") as f:
        json.dump(
            {
                "n_iterations_ground_truth": N_BOOTSTRAP_GROUND_TRUTH,
                "n_iterations_subsets": N_BOOTSTRAP_SUBSETS,
                "r_squared_theshold": R_SQUARED_THRESHOLD,
                "n_replications": N_REPLICATIONS_ORIGINAL,
                "base_path": BASE_PATH,
                "patients_id": PATIENTS_ID,
                "curves": CURVE["name"],
                "curve_fitting_result_file": CURVE_FITTING_RESULT_FILE,
                "bootstrap_percentiles": BOOTSTRAP_PERCENTILES,
                "clean_data": CLEAN_DATA,
                "clean_data_file": CLEAN_DATA_FILE,
                "sleep_stages": [stage.name for stage in SLEEP_STAGES],
                "loss": LOSS,
                "max_iterations": MAX_ITERATIONS,
                "include_replications_list": INCLUDE_REPLICATIONS_LIST,
                "exclude_intensities_matrices": EXCLUDE_INTENSITIES_MATRICES,
                "use_cache_gt": USE_CACHE_GT,
                "use_cahe_subset": USE_CACHE_SUBSET,
                "boot_r_squared_threshold": BOOT_R_SQUARED_THRESHOLD,
                "use_min_normalization": USE_MIN_NORMALIZATION,
            },
            f,
            indent=4,
        )

    results = {}

    complete_curve_name = CURVE["name"] + "_" + LOSS

    responses_df = pd.read_json(SURROGATE_RESULT_FILE, orient="records")
    curve_fitting_df = pd.read_json(CURVE_FITTING_RESULT_FILE, orient="records")

    merged_df = pd.merge(
        responses_df,
        curve_fitting_df,
        on=["patient_id", "stim_channel_name", "response_channel_name"],
        how="left",
    )

    results = {}

    for patient_id in PATIENTS_ID:
        print(f"{pd.Timestamp.now()}: Processing patient {patient_id}")
        names_h5 = get_h5_names_of_patient(
            BASE_PATH,
            patient_id,
            protocol="CR",
            new_overview_format=True if DATA_PAPER else False,
        )
        results[patient_id] = {}

        path_lookup = f"{BASE_PATH}/{patient_id}/Electrodes/Lookup.xlsx"
        paths_h5 = [
            f"{BASE_PATH}/{patient_id}/Electrophy/{name}.h5" for name in names_h5
        ]
        paths_logs = [
            f"{BASE_PATH}/{patient_id}/out/{name}_logs.csv" for name in names_h5
        ]
        if CLEAN_DATA:
            path_excluded_responses = f"{BASE_PATH}/{patient_id}/{CLEAN_DATA_FILE}"
        else:
            path_excluded_responses = None

        mrl = MultipleHDFResponseLoader(
            paths_h5=paths_h5,
            paths_logs=paths_logs,
            recording_names=names_h5,
            path_lookup=path_lookup,
            path_excluded_responses=path_excluded_responses,
        )
        mrl.add_sleep_score_to_logs()

        logs = mrl.get_logs()

        intensities = (
            logs[logs["type"] == "CR_IO"]["Int_prob"].drop_duplicates().tolist()
        )
        intensities.insert(0, 0)
        intensities.sort()
        norm_intensities = intensities / np.max(intensities)

        exclude_intensities_matrix = EXCLUDE_INTENSITIES_MATRICES.get(
            len(intensities), None
        )

        stim_channels = (
            logs[logs["type"] == "CR_IO"][["name_pos", "name_neg"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        stim_channel_names = stim_channels.agg("-".join, axis=1).tolist()
        stim_channel_paths = mrl.get_channel_paths_from_names(stim_channel_names)
        for stim_channel_index, stim_channel in stim_channels.iterrows():
            result_rows = []

            stim_channel_name = stim_channel_names[stim_channel_index]
            results[patient_id][stim_channel_name] = {
                "connections": {},
                "norm_intensities": list(
                    norm_intensities
                ),  # just for convenience (also in curve fitting)
            }

            print(" ... working on stim channel: ", stim_channel_name)

            _, max_n_replications = find_max_n_replications(
                complete_logs=logs,
                selected_stim_channel_name_neg=stim_channel["name_neg"],
                selected_stim_channel_name_pos=stim_channel["name_pos"],
                stim_protocol="CR_IO",
                sleep_states=SLEEP_STAGES,
            )
            if max_n_replications < N_REPLICATIONS_ORIGINAL:
                print(f"Not enough replications for {patient_id} - {stim_channel}")
                continue

            ## PREPARATION
            filtered_logs = filter_logs(
                complete_logs=logs,
                n_replications=N_REPLICATIONS_ORIGINAL,
                selected_stim_channel_name_pos=stim_channel["name_pos"],
                selected_stim_channel_name_neg=stim_channel["name_neg"],
                sleep_stages=SLEEP_STAGES,
            )

            ## FOCUS ON SIGNIFICANT SIGMOID RESPONSE CHANNELS
            responses_df = merged_df[
                (merged_df["patient_id"] == patient_id)
                & (merged_df["stim_channel_name"] == stim_channel_name)
            ]

            for df_i, df_row in responses_df.iterrows():
                if df_row[CURVE["name"] + "_r_squared"] > R_SQUARED_THRESHOLD:
                    response_channel_name = df_row["response_channel_name"]
                    print(
                        "  ... working on",
                        stim_channel_name,
                        "-",
                        response_channel_name,
                    )

                    response_channel_path = mrl.get_channel_paths_from_names(
                        [response_channel_name]
                    )[0]

                    ll_med_values = parsed_list_to_numpy_array(df_row["med_lls"])
                    ll_values = parsed_list_to_numpy_array(df_row["ll_values"])

                    normalized_ll_med_values = normalize_ll_values(
                        ll_values=ll_med_values, axis=0, use_min=USE_MIN_NORMALIZATION
                    )
                    shared_ll_min = np.nanmin(ll_med_values, axis=0)
                    shared_ll_max = np.nanpercentile(ll_med_values, 95, axis=0)
                    norm_ll_values = normalize_ll_values(
                        ll_values=ll_values,
                        max=shared_ll_max,
                        min=shared_ll_min,
                        use_min=USE_MIN_NORMALIZATION,
                        axis=0,
                    )

                    # if not np.allclose(ll_med_values, df_row["med_lls"]):
                    #     print(
                    #         f"Warning: Median LL values from surrogate file and freshly calculated do not match for {patient_id} - {stim_channel_name} - {response_channel_name}, difference: {np.sum(np.abs(ll_med_values - df_row['med_lls']))}"
                    #     )

                    # if not np.allclose(
                    #     normalized_ll_med_values,
                    #     df_row["norm_med_lls"],
                    # ):
                    #     print(
                    #         f"Normalized median LL values from surrogate file and freshly calculated do not match for {patient_id} - {stim_channel_name} - {response_channel_name}, difference: {np.sum(np.abs(normalized_ll_med_values - df_row['norm_med_lls']))}"
                    #     )

                    x_fit = np.linspace(
                        min(norm_intensities),
                        max(norm_intensities),
                        1000,
                    )

                    ### ORIGINAL FIT
                    try:
                        params_original = fit_curve(
                            curve_function=CURVE["function"],
                            x=norm_intensities,
                            y=normalized_ll_med_values,
                            initial_values=CURVE["initial_values"],
                            bounds=CURVE["bounds"],
                        )
                        y_fit_original = CURVE["function"](
                            x_fit, *params_original
                        )  # shape (1000,)
                        auc_original = np.trapezoid(y=y_fit_original, x=x_fit)
                        fig = plt.figure(figsize=(5, 9))  # Adjust figure size as needed
                        r_squared_original = plot_curve_fittings(
                            curve_function=CURVE["function"],
                            x=norm_intensities,
                            y=normalized_ll_med_values,
                            params_list=[params_original],
                            names=["All datapoints"],
                            fig=fig,
                        )[0]
                        plt.suptitle(
                            f"Curve fitting\nRecording: {patient_id}\nStimulation: {stim_channel_name}\nResponse: {response_channel_name}"
                        )
                        plt.tight_layout()
                        plt.savefig(
                            f"{OUT_PATH}/original_fit_{patient_id}_{stim_channel_name}_{response_channel_name}.png"
                        )
                        plt.close("all")

                    except RuntimeError:
                        print("Could not fit curve")
                        continue

                    ## DO BOOTSTRAPPING FOR RESPONSE CHANNELS
                    fig = plt.figure(figsize=(50, 40))  # Adjust figure size as needed

                    width_ratios = [
                        2 if i % 2 == 0 else 1
                        for i in range(len(INCLUDE_REPLICATIONS_LIST) * 2 + 2)
                    ]
                    gs = GridSpec(
                        len(exclude_intensities_matrix) + 1,
                        len(INCLUDE_REPLICATIONS_LIST) * 2 + 2,
                        figure=fig,
                        width_ratios=width_ratios,
                    )

                    metrics = {
                        "misfit": [],
                        "delta_auc": [],
                        "relative_error": [],
                        "relative_width": [],
                    }

                    ### CREATE GROUND TRUTH CURVE
                    if not USE_CACHE_GT:
                        params_list = bootstrap_curve_fitting(
                            curve_function=CURVE["function"],
                            x=norm_intensities,
                            y=norm_ll_values,
                            initial_values=CURVE["initial_values"],
                            bounds=CURVE["bounds"],
                            max_optimizer_iterations=MAX_ITERATIONS,
                            loss=LOSS,
                            n_bootstrap=N_BOOTSTRAP_GROUND_TRUTH,
                            parallelize=True,
                            normalize=False,
                        )
                        if STORE_ALL_DATA:
                            np.savez_compressed(
                                f"{OUT_PATH}/params_list_{patient_id}_{stim_channel_name}_{response_channel_name}_gt.npz",
                                params_list=params_list,
                            )
                    else:
                        try:
                            params_list = np.load(
                                f"{OUT_PATH}/params_list_{patient_id}_{stim_channel_name}_{response_channel_name}_gt.npz",
                                allow_pickle=True,
                            )["params_list"]
                        except:
                            print("Error loading cached file.")

                    if len(params_list) == 0:
                        print(
                            f"No params found for {patient_id} - {stim_channel_name} - {response_channel_name} for GT."
                        )
                        continue

                    ground_truth_result = evaluate_bootstrap_result(
                        x_fit=x_fit,
                        curve_function=CURVE["function"],
                        params_list=params_list,
                        percentiles=BOOTSTRAP_PERCENTILES + [50],  # median at the end
                        r_squared_threshold=BOOT_R_SQUARED_THRESHOLD,
                        x_dp=norm_intensities,
                        y_dp_true=normalized_ll_med_values,
                    )
                    print("GT - excluded", ground_truth_result["excluded_r_squared"])

                    if len(ground_truth_result["aucs"]) < N_BOOTSTRAP_GROUND_TRUTH:
                        if len(ground_truth_result["aucs"]) == 0:
                            # TODO find better solution
                            continue

                        # Sometimes it does not converege -> use filler value of 0
                        pad_length = N_BOOTSTRAP_GROUND_TRUTH - len(
                            ground_truth_result["aucs"]
                        )
                        ground_truth_result["aucs"] = np.concatenate(
                            [ground_truth_result["aucs"], np.full(pad_length, 0)]
                        )

                    ground_truth_ci = ground_truth_result[
                        "percentiles"
                    ]  # shape (3, len(x_fit))
                    ground_truth_auc_ci = ground_truth_result["auc_percentiles"]

                    for i, exclude_intensities in enumerate(exclude_intensities_matrix):
                        misfit_arr = []
                        delta_auc_arr = []
                        relative_error_arr = []
                        relative_width_arr = []
                        inaccuracy_size = []

                        # remove intensities
                        intensity_reduced_intensities = np.delete(
                            norm_intensities, exclude_intensities
                        )
                        intensity_reduced_norm_ll_values = np.delete(
                            norm_ll_values, exclude_intensities, axis=0
                        )
                        intensity_reduced_norm_ll_med_values = np.nanmedian(
                            intensity_reduced_norm_ll_values, axis=1
                        )

                        # create plot with CI (first column)
                        ax_curve = fig.add_subplot(gs[i, 0])
                        ax_hist = fig.add_subplot(gs[i, 1])

                        ax_curve.set_title(
                            f"Ground truth: {len(intensity_reduced_intensities)} intensities"
                        )
                        ax_curve.text(
                            0.5,
                            -0.2,  # y < 0 means outside below the axis
                            s=", ".join(
                                str(x)  # f"{x:.1f}"
                                for x in np.delete(
                                    np.array(intensities), exclude_intensities
                                )
                            ),
                            fontsize=6,
                            ha="center",
                            va="top",
                            transform=ax_curve.transAxes,
                            clip_on=False,
                        )
                        ax_curve.scatter(
                            np.repeat(
                                intensity_reduced_intensities, N_REPLICATIONS_ORIGINAL
                            ),
                            intensity_reduced_norm_ll_values,
                            s=2,
                            alpha=0.5,
                        )
                        ax_curve.plot(
                            intensity_reduced_intensities,
                            intensity_reduced_norm_ll_med_values,
                            label="Median LL",
                        )
                        ax_curve.plot(
                            norm_intensities,
                            normalized_ll_med_values,
                            label="GT Med. LL",
                            color="grey",
                            linestyle=":",
                        )
                        ax_curve.plot(
                            x_fit,
                            CURVE["function"](x_fit, *params_original),
                            label="Original fit",
                            color="black",
                            linestyle=":",
                        )
                        ax_curve.fill_between(
                            x_fit,
                            ground_truth_ci[0],
                            ground_truth_ci[1],
                            color="blue",
                            alpha=0.2,
                            label=f"{round(BOOTSTRAP_PERCENTILES[1] - BOOTSTRAP_PERCENTILES[0])}% Bootstrap CI",
                        )
                        if i == 0:
                            ax_curve.legend()
                        ax_curve.set_xlabel("Norm. Intensitiy")
                        ax_curve.set_ylabel("Norm. LL values")
                        ax_curve.set_ylim(-0.1, 1.1)
                        ax_curve.set_xlim(-0.1, 1.1)
                        try:
                            ax_hist.hist(
                                ground_truth_result["aucs"],
                                bins=30,
                                label="GT AUCs",
                                histtype="step",
                            )
                        except:
                            print("Histogram plotting error")
                        ax_hist.axvline(
                            x=auc_original,
                            color="black",
                            linestyle="--",
                            label="GT AUC",
                        )
                        ax_hist.axvline(
                            x=ground_truth_auc_ci[0],
                            color="blue",
                            linestyle=":",
                            label=f"{BOOTSTRAP_PERCENTILES[0]}% GT AUC",
                        )
                        ax_hist.axvline(
                            x=ground_truth_auc_ci[1],
                            color="blue",
                            linestyle=":",
                            label=f"{BOOTSTRAP_PERCENTILES[1]}% GT AUC",
                        )
                        # ax_hist.set_xlim(0, 1)
                        if i == 0:
                            ax_hist.legend()

                        ### CREATE SUBSET DISTRIBUTIONS
                        subset_params_matrix = []
                        if USE_CACHE_SUBSET:
                            try:
                                subset_params_matrix = np.load(
                                    f"{OUT_PATH}/params_list_{patient_id}_{stim_channel_name}_{response_channel_name}_subset_{i}.npz",
                                    allow_pickle=True,
                                )["subset_params_matrix"]
                            except:
                                print("Cached file not found")
                                continue

                        for j, n_subset_replications in enumerate(
                            INCLUDE_REPLICATIONS_LIST
                        ):
                            if not USE_CACHE_SUBSET:
                                subset_params_list = subset_bootstrap_curve_fitting(
                                    x=intensity_reduced_intensities,
                                    y=intensity_reduced_norm_ll_values,
                                    curve_function=CURVE["function"],
                                    initial_values=CURVE["initial_values"],
                                    bounds=CURVE["bounds"],
                                    n_subset_replications=n_subset_replications,
                                    max_optimizer_iterations=MAX_ITERATIONS,
                                    loss=LOSS,
                                    n_bootstrap=N_BOOTSTRAP_SUBSETS,
                                    parallelize=True,
                                    normalize=False,
                                )
                                subset_params_matrix.append(subset_params_list)
                            else:
                                subset_params_list = subset_params_matrix[j]

                            # print(len(params_list), "params found")
                            if len(subset_params_list) == 0:
                                continue

                            ax_curve = fig.add_subplot(gs[i, j * 2 + 2])
                            ax_hist = fig.add_subplot(gs[i, j * 2 + 3])

                            subset_result = evaluate_bootstrap_result(
                                x_fit=x_fit,
                                curve_function=CURVE["function"],
                                params_list=subset_params_list,
                                percentiles=BOOTSTRAP_PERCENTILES,
                                r_squared_threshold=BOOT_R_SQUARED_THRESHOLD,
                                x_dp=norm_intensities,
                                y_dp_true=normalized_ll_med_values,
                            )  # shape (2, len(x_fit))

                            if len(subset_result["aucs"]) == 0:
                                misfit = 1
                                relative_error = np.nan
                                relative_error = np.nan
                                relative_width = np.nan
                                delta_auc = np.nan
                            else:
                                if len(subset_result["aucs"]) < N_BOOTSTRAP_SUBSETS:
                                    # Sometimes it does not converege -> use filler value of 0
                                    pad_length = N_BOOTSTRAP_SUBSETS - len(
                                        subset_result["aucs"]
                                    )
                                    subset_result["aucs"] = np.concatenate(
                                        [subset_result["aucs"], np.full(pad_length, 0)]
                                    )

                                subset_ci = subset_result["percentiles"]
                                misfit = np.mean(
                                    (subset_result["aucs"] < ground_truth_auc_ci[0])
                                    | (subset_result["aucs"] > ground_truth_auc_ci[1])
                                )
                                relative_error = (
                                    np.mean(subset_result["aucs"]) - auc_original
                                ) / auc_original
                                # relative_error = np.mean(
                                #     np.abs(subset_result["aucs"] - auc_original)
                                #     / auc_original
                                # )
                                relative_width = (
                                    subset_result["auc_percentiles"][1]
                                    - subset_result["auc_percentiles"][0]
                                ) / (ground_truth_auc_ci[1] - ground_truth_auc_ci[0])
                                delta_auc = np.mean(
                                    np.abs(
                                        subset_result["aucs"] - ground_truth_auc_ci[2]
                                    )  # auc_original)
                                )
                                # inaccuracy_size =
                            # misfit_percentage = subset_result["aucs"]
                            ax_curve.fill_between(
                                x_fit,
                                subset_ci[0],
                                subset_ci[1],
                                color="red",
                                alpha=0.2,
                                label=f"{round(BOOTSTRAP_PERCENTILES[1] - BOOTSTRAP_PERCENTILES[0])}% Subset Bootstrap CI",
                            )
                            ax_curve.fill_between(
                                x_fit,
                                ground_truth_ci[0],
                                ground_truth_ci[1],
                                color="blue",
                                alpha=0.2,
                                label=f"{round(BOOTSTRAP_PERCENTILES[1] - BOOTSTRAP_PERCENTILES[0])}% Ground truth Bootstrap CI",
                            )
                            ax_curve.plot(
                                x_fit,
                                CURVE["function"](x_fit, *params_original),
                                label="Original fit",
                                color="black",
                                linestyle=":",
                            )
                            if i == 0 and j == 0:
                                ax_curve.legend()
                            ax_curve.set_xlabel("Norm. Intensitiy")
                            ax_curve.set_ylabel("Norm. LL values")
                            ax_curve.set_ylim(-0.1, 1.1)
                            ax_curve.set_xlim(-0.1, 1.1)
                            ax_curve.set_title(
                                f"Subset: {n_subset_replications} replications"
                            )
                            try:
                                bins = np.histogram_bin_edges(
                                    ground_truth_result["aucs"], bins=30
                                )
                                ax_hist.hist(
                                    ground_truth_result["aucs"],
                                    bins=bins,
                                    histtype="step",
                                    label="GT AUCs",
                                )
                                ax_hist.hist(
                                    subset_result["aucs"],
                                    bins=bins,
                                    histtype="step",
                                    label="Subset AUCs",
                                )
                                outside_mask = (
                                    subset_result["aucs"] < ground_truth_auc_ci[0]
                                ) | (subset_result["aucs"] > ground_truth_auc_ci[1])
                                ax_hist.hist(
                                    subset_result["aucs"][outside_mask],
                                    bins=bins,
                                    histtype="step",
                                    label="Exclusio",
                                    linestyle=":",
                                )
                            except:
                                print("Histogram plotting error")
                            ax_hist.axvline(
                                x=auc_original,
                                color="black",
                                linestyle="--",
                                label="GT AUC",
                            )
                            ax_hist.axvline(
                                x=ground_truth_auc_ci[0],
                                color="blue",
                                linestyle=":",
                                label=f"{BOOTSTRAP_PERCENTILES[0]}% GT AUC",
                            )
                            ax_hist.axvline(
                                x=ground_truth_auc_ci[1],
                                color="blue",
                                linestyle=":",
                                label=f"{BOOTSTRAP_PERCENTILES[1]}% GT AUC",
                            )
                            ax_hist.set_title(
                                f"Misfit: {misfit:.2%}, {delta_auc:.2f},\nAcc: {relative_error:.2%}, P: {relative_width:.2}x"
                            )
                            # ax_hist.set_xlim(0, 1)
                            if i == 0 and j == 0:
                                ax_hist.legend()

                            misfit_arr.append(misfit)
                            delta_auc_arr.append(delta_auc)
                            relative_error_arr.append(relative_error)
                            relative_width_arr.append(relative_width)

                        if not USE_CACHE_SUBSET:
                            subset_params_matrix = (
                                np.array(  # since param_lists sizes are inhomogeneous
                                    subset_params_matrix, dtype=object
                                )
                            )
                            if STORE_ALL_DATA:
                                np.savez_compressed(
                                    f"{OUT_PATH}/params_list_{patient_id}_{stim_channel_name}_{response_channel_name}_subset_{i}.npz",
                                    subset_params_matrix=subset_params_matrix,
                                )

                        metrics["misfit"].append(misfit_arr)
                        metrics["delta_auc"].append(delta_auc_arr)
                        metrics["relative_error"].append(relative_error_arr)
                        metrics["relative_width"].append(relative_width_arr)

                    result_rows.append(
                        {
                            "patient_id": patient_id,
                            "stim_channel_name": stim_channel_name,
                            "response_channel_name": response_channel_name,
                            "params_original": params_original,
                            "misfit": metrics["misfit"],
                            "delta_auc": metrics["delta_auc"],
                            "relative_error": metrics["relative_error"],
                            "relative_width": metrics["relative_width"],
                        }
                    )

                    plt.suptitle(
                        f"Bootstrap\nRecording: {patient_id}\nStimulation: {stim_channel_name}\nResponse: {response_channel_name}\n\nGT: {N_BOOTSTRAP_GROUND_TRUTH}, n_subsets: {N_BOOTSTRAP_SUBSETS}",
                    )
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    plt.savefig(
                        f"{OUT_PATH}/result_{patient_id}_{stim_channel_name}_{response_channel_name}.png"
                    )
                    plt.close("all")

            boostrap_df = pd.DataFrame(result_rows)

            if len(boostrap_df) > 0:
                json_path = f"{OUT_PATH}/bootstrap_curve_analysis_lf.json"
                lock_path = json_path + ".lock"
                with FileLock(lock_path):
                    if os.path.exists(json_path):
                        df = pd.read_json(json_path, orient="records")
                        # Remove old entries
                        mask = (df["patient_id"] == patient_id) & (
                            df["stim_channel_name"] == stim_channel_name
                        )

                        # Drop old entries
                        df = df[~mask]

                        # Combine
                        boostrap_df = pd.concat([df, boostrap_df], ignore_index=True)

                    boostrap_df.to_json(json_path, orient="records", indent=4)
