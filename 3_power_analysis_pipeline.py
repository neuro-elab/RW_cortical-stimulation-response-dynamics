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
    find_params_for_given_effect_size,
    fit_curve,
    calculate_model_performance,
    normalize_ll_values,
    pick_random_replications,
    significant_exi_difference_testing,
)

from connectivity.curves import CURVES


if __name__ == "__main__":
    load_dotenv()
    BASE_PATH = os.getenv("BASE_PATH_PAPER", "/default/path")

    PATIENTS_ID = [sys.argv[1]]
    EFFECT_SIZES = [float(arg) for arg in sys.argv[3:]]
    PARAM_INDEX_TO_CHANGE = int(
        sys.argv[2]
    )  # 2 = midpoint, 0 = lower plateau, 1 = upper plateau
    IS_PARAM_TO_CHANGE_RECIPROCAL = {
        2: True,
        0: False,
        1: False,
    }[PARAM_INDEX_TO_CHANGE]
    CURVE = CURVES["5P"]

    print(f"Patients: {PATIENTS_ID}")
    print(f"Effect sizes: {EFFECT_SIZES}")
    print(f"Parameter to change: {CURVE['param_names'][PARAM_INDEX_TO_CHANGE]}")
    for var in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        print(var, os.getenv(var))

    OUT_PATH = "output/full_power_analysis_test"

    N_REPLICATIONS_ORIGINAL = 12
    BOOTSTRAP_PERCENTILES = [2.5, 97.5]  #
    R_SQUARED_THRESHOLD = 0.6
    CURVE_FITTING_RESULT_FILE = "output/curve_fitting/curve_fitting_lf.json"
    CONNECTIONS_RESULT_FILE = "output/significant_responses/response_channels_lf.json"

    USE_MIN_NORMALIZATION = True
    MAX_ITERATIONS = 20000
    MAX_ITERATIONS_SURR = 1000
    EFFECT_SIZE_PRECISION = 0.001
    N_SURROGATES = 500

    BASELINE_CORRECTION = True
    SUBSET_CURVE_THRESHOLD = -1

    REDUCE_ALONG_ONE_DIM_ONLY = True
    # INCLUDE_REPLICATIONS_LIST = [12, 9, 6, 5, 4, 3, 2, 1]
    INCLUDE_REPLICATIONS_LIST = [6, 5, 4, 3]
    PLOT = True
    # EXCLUDE_INTENSITIES_MATRICES = {
    #     17: [
    #         [],  # all intensities
    #         [12],
    #         [12, 14],
    #         [12, 14, 1],
    #         [12, 14, 1, 3],
    #         [12, 14, 1, 3, 5],
    #         [12, 14, 1, 3, 5, 7],
    #         [12, 14, 1, 3, 5, 7, 10],
    #         [12, 14, 1, 3, 5, 7, 10, 8],
    #         [12, 14, 1, 3, 5, 7, 10, 15],
    #         [12, 14, 1, 3, 5, 7, 10, 8, 15],
    #     ],
    #     18: [
    #         [],  # all intensities
    #         [2],
    #         [2, 13],
    #         [2, 13, 15],
    #         [2, 13, 15, 1],
    #         [2, 13, 15, 1, 4],
    #         [2, 13, 15, 1, 4, 6],
    #         [2, 13, 15, 1, 4, 6, 8],
    #         [2, 13, 15, 1, 4, 6, 8, 11],
    #         [2, 13, 15, 1, 4, 6, 8, 11, 9],
    #         [2, 13, 15, 1, 4, 6, 8, 11, 16],
    #         [2, 13, 15, 1, 4, 6, 8, 11, 9, 16],
    #     ],
    #     15: [
    #         [],  # all intensities
    #         [1],
    #         [1, 3],
    #         [1, 3, 5],
    #         [1, 3, 5, 7],
    #         [1, 3, 5, 7, 10],
    #         [1, 3, 5, 7, 10, 8],
    #         [1, 3, 5, 7, 10, 13],
    #         [1, 3, 5, 7, 10, 8, 13],
    #     ],
    # }
    EXCLUDE_INTENSITIES_MATRICES = {
        17: [
            [12, 14],
            [12, 14, 1, 3],
            [12, 14, 1, 3, 5, 7, 10],
            [12, 14, 1, 3, 5, 7, 10, 8, 15],
        ],
        18: [
            [2, 13, 15],
            [2, 13, 15, 1, 4],
            [2, 13, 15, 1, 4, 6, 8, 11],
            [2, 13, 15, 1, 4, 6, 8, 11, 9, 16],
        ],
        15: [[], [1, 3], [1, 3, 5, 7, 10], [1, 3, 5, 7, 10, 8, 13]],  # all intensities
    }

    # EXCLUDE_INTENSITIES_MATRICES = {
    #     17: [
    #         [12, 14, 1, 3, 5, 7, 10, 8, 15],
    #     ],
    #     18: [
    #         [2, 13, 15, 1, 4, 6, 8, 11, 9, 16],
    #     ],
    #     15: [[1, 3, 5, 7, 10, 8, 13]],  # all intensities
    # }

    import multiprocessing as mp

    mp.freeze_support()

    with open(f"{OUT_PATH}/params_{'_'.join(PATIENTS_ID)}.json", "w") as f:
        json.dump(
            {
                "n_replications": N_REPLICATIONS_ORIGINAL,
                "base_path": BASE_PATH,
                "patients_id": PATIENTS_ID,
                "curves": CURVE["name"],
                "curve_fitting_result_file": CURVE_FITTING_RESULT_FILE,
                "bootstrap_percentiles": BOOTSTRAP_PERCENTILES,
                "max_iterations_surr": MAX_ITERATIONS_SURR,
                "max_iterations": MAX_ITERATIONS,
                "include_replications_list": INCLUDE_REPLICATIONS_LIST,
                "exclude_intensities_matrices": EXCLUDE_INTENSITIES_MATRICES,
                "use_min_normalization": USE_MIN_NORMALIZATION,
                "effect_sizes": ",".join(str(x) for x in EFFECT_SIZES),
                "effect_size_precision": EFFECT_SIZE_PRECISION,
                "r_squared_threshold": R_SQUARED_THRESHOLD,
                "n_surrogates": N_SURROGATES,
                "param_to_change": CURVE["param_names"][PARAM_INDEX_TO_CHANGE],
                "param_reciprocal": IS_PARAM_TO_CHANGE_RECIPROCAL,
                "baseline_correction": BASELINE_CORRECTION,
                "subset_curve_threshold": SUBSET_CURVE_THRESHOLD,
                "reduce_along_one_dim_only": REDUCE_ALONG_ONE_DIM_ONLY,
                "plot": PLOT,
            },
            f,
            indent=4,
        )

    results = {}

    complete_curve_name = CURVE["name"]

    responses_df = pd.read_json(CONNECTIONS_RESULT_FILE, orient="records")
    curve_fitting_df = pd.read_json(CURVE_FITTING_RESULT_FILE, orient="records")

    merged_df = pd.merge(
        responses_df,
        curve_fitting_df,
        on=["patient_id", "stim_channel_name", "response_channel_name"],
        how="left",
    )

    x_fit = np.linspace(0, 1, 1000)
    results = {}

    for effect_size in EFFECT_SIZES:
        for patient_id in PATIENTS_ID:
            print(f"{pd.Timestamp.now()}: Processing patient {patient_id}")

            names_h5 = get_h5_names_of_patient(BASE_PATH, patient_id, protocol="CR")
            results[patient_id] = {}

            path_lookup = f"{BASE_PATH}/{patient_id}/Electrodes/Lookup.xlsx"
            paths_h5 = [
                f"{BASE_PATH}/{patient_id}/Electrophy/{name}.h5" for name in names_h5
            ]
            paths_logs = [
                f"{BASE_PATH}/{patient_id}/out/{name}_logs.csv" for name in names_h5
            ]

            mrl = MultipleHDFResponseLoader(
                paths_h5=paths_h5,
                paths_logs=paths_logs,
                recording_names=names_h5,
                path_lookup=path_lookup,
            )
            mrl.add_sleep_score_to_logs()

            logs = mrl.get_logs()

            io_intensities = (
                logs[logs["type"] == "CR_IO"]["Int_prob"].drop_duplicates().tolist()
            )
            io_intensities.insert(0, 0)
            io_intensities.sort()
            norm_intensities = io_intensities / np.max(io_intensities)

            exclude_intensities_matrix = EXCLUDE_INTENSITIES_MATRICES.get(
                len(io_intensities), None
            )

            stim_channels = (
                logs[logs["type"] == "CR_IO"][["name_pos", "name_neg"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            stim_channel_names = stim_channels.agg("-".join, axis=1).tolist()
            stim_channel_paths = mrl.get_channel_paths_from_names(stim_channel_names)

            df_patient = merged_df[
                (merged_df["patient_id"] == patient_id)
                & (merged_df["is_significant"] == True)
                & (merged_df[CURVE["name"] + "_r_squared"] > R_SQUARED_THRESHOLD)
            ]
            for stim_channel_index, stim_channel in stim_channels.iterrows():
                result_rows = []

                stim_channel_name = stim_channel_names[stim_channel_index]

                print("WORKING ON ", patient_id, effect_size, stim_channel_name)

                df_subset = df_patient[
                    df_patient["stim_channel_name"] == stim_channel_name
                ]
                for i_row, df_row in df_subset.iterrows():
                    response_channel_name = df_row["response_channel_name"]

                    if PLOT:
                        fig = plt.figure(
                            figsize=(50, 40)
                        )  # Adjust figure size as needed

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

                    # 1. get row and med LL values (does this make sense?)
                    original_lls = parsed_list_to_numpy_array(
                        df_row["ll_values"]
                    )  # shape: (n_intensities, n_replications)
                    original_med_lls = np.nanmedian(original_lls, axis=1)
                    original_norm_med_lls = normalize_ll_values(
                        original_med_lls,
                        axis=0,
                        min=0 if BASELINE_CORRECTION else None,
                        use_min=USE_MIN_NORMALIZATION,
                    )
                    shared_original_ll_max = np.nanpercentile(
                        original_med_lls, 95, axis=0
                    )
                    shared_original_ll_min = np.nanmin(original_med_lls, axis=0)
                    original_norm_lls = normalize_ll_values(
                        original_lls,
                        max=shared_original_ll_max,
                        min=0 if BASELINE_CORRECTION else shared_original_ll_min,
                        use_min=USE_MIN_NORMALIZATION,
                        axis=0,
                    )
                    # get original ExI
                    try:
                        original_params = fit_curve(
                            curve_function=CURVE["function"],
                            x=norm_intensities,
                            y=original_norm_med_lls,
                            initial_values=CURVE["initial_values"],
                            bounds=CURVE["bounds"],
                            max_iterations=MAX_ITERATIONS,
                        )

                        original_y_fit = CURVE["function"](x_fit, *original_params)
                        original_y_pred = CURVE["function"](
                            norm_intensities, *original_params
                        )

                        original_exi = np.trapezoid(original_y_fit, x_fit)
                    except Exception as e:
                        print(
                            f"{patient_id}: {stim_channel_name}/{response_channel_name}: Optimization failed."
                        )
                        continue

                    if PARAM_INDEX_TO_CHANGE == 2:  # inflection point
                        if original_params[2] < 0.3:
                            _target_exi = original_exi - effect_size
                        else:
                            _target_exi = original_exi + effect_size
                    elif PARAM_INDEX_TO_CHANGE == 0:  # lower plateau
                        _target_exi = original_exi + effect_size
                    elif PARAM_INDEX_TO_CHANGE == 1:  # upper plateau
                        _target_exi = original_exi - effect_size

                    try:
                        new_params = find_params_for_given_effect_size(
                            curve=CURVE,
                            orig_params=original_params,
                            selected_param_index=PARAM_INDEX_TO_CHANGE,
                            is_reciprocal=IS_PARAM_TO_CHANGE_RECIPROCAL,
                            target_exi=_target_exi,
                            x=x_fit,
                            precision=EFFECT_SIZE_PRECISION,
                        )
                    except Exception as e:
                        print(
                            f"{patient_id}: {stim_channel_name}/{response_channel_name}: No ad-hoc curve found."
                        )
                        continue

                    new_y_fit = CURVE["function"](x_fit, *new_params)
                    # generate new datapoints

                    original_y_pred = CURVE["function"](
                        norm_intensities, *original_params
                    )
                    new_y_pred = CURVE["function"](norm_intensities, *new_params)
                    new_norm_lls = []
                    for i_int, _ in enumerate(norm_intensities):
                        residuals = original_norm_lls[i_int] - original_y_pred[i_int]
                        target_std = np.std(residuals)

                        # generated_lls = np.random.normal(
                        #     loc=new_y_pred[i_int],
                        #     scale=std,
                        #     size=N_REPLICATIONS_ORIGINAL,
                        # )
                        generated_lls = np.random.normal(
                            loc=0,
                            scale=1,
                            size=N_REPLICATIONS_ORIGINAL,
                        )
                        # target_iqr = np.subtract(
                        #     *np.percentile(residuals, [75, 25])
                        # )  # 75th - 25th percentile
                        # gen_iqr = np.subtract(*np.percentile(generated_lls, [75, 25]))
                        # b = target_iqr / gen_iqr  # scale factor to adjust spread
                        b = target_std / np.std(generated_lls)
                        # new_y_pred[i_int] == target median
                        a = new_y_pred[i_int] - b * np.median(
                            generated_lls
                        )  # median shift (centered around new predicted value)
                        generated_lls = a + b * generated_lls

                        new_norm_lls.append(generated_lls)
                    new_norm_lls = np.array(new_norm_lls)

                    delta_exis = []
                    surr_p_values = []
                    surr_p_values_empirical = []
                    r_squared_original = []
                    r_squared_new = []

                    for i, exclude_intensities in enumerate(exclude_intensities_matrix):
                        delta_exis_row = []
                        surr_p_values_row = []
                        surr_p_values_empirical_row = []
                        r_squared_original_row = []
                        r_squared_new_row = []

                        # remove intensities
                        int_reduced_intensities = np.delete(
                            norm_intensities, exclude_intensities
                        )
                        # ORIGINAL
                        int_reduced_original_norm_lls = np.delete(
                            original_norm_lls, exclude_intensities, axis=0
                        )
                        int_reduced_new_norm_lls = np.delete(
                            new_norm_lls, exclude_intensities, axis=0
                        )

                        for j, n_subset_replications in enumerate(
                            INCLUDE_REPLICATIONS_LIST
                        ):
                            if REDUCE_ALONG_ONE_DIM_ONLY:
                                if not (i == 0 or j == 0):
                                    delta_exis_row.append(np.nan)
                                    surr_p_values_row.append(np.nan)
                                    surr_p_values_empirical_row.append(np.nan)
                                    r_squared_original_row.append(np.nan)
                                    r_squared_new_row.append(np.nan)
                                    continue

                            if PLOT:
                                ax_curve = fig.add_subplot(gs[i, j * 2 + 2])
                                ax_hist = fig.add_subplot(gs[i, j * 2 + 3])

                            # all normalized
                            original_subset_lls = pick_random_replications(
                                int_reduced_original_norm_lls,
                                n_replications_to_select=n_subset_replications,
                            )
                            original_subset_med_lls = np.nanmedian(
                                original_subset_lls, axis=1
                            )
                            new_subset_lls = pick_random_replications(
                                int_reduced_new_norm_lls,
                                n_replications_to_select=n_subset_replications,
                            )
                            new_subset_med_lls = np.nanmedian(new_subset_lls, axis=1)

                            surr_p_value_empirical, surr_p_value = (
                                significant_exi_difference_testing(
                                    norm_ll_values_1=original_subset_lls,
                                    norm_ll_values_2=new_subset_lls,
                                    n_surrogates=N_SURROGATES,
                                    intensities=int_reduced_intensities,
                                    curve=CURVE,
                                    ax=ax_hist if PLOT else None,
                                    parallelize=True,
                                    max_iterations=MAX_ITERATIONS_SURR,
                                )
                            )
                            surr_p_values_empirical_row.append(surr_p_value_empirical)
                            surr_p_values_row.append(surr_p_value)

                            if PLOT:
                                ax_curve.scatter(
                                    int_reduced_intensities,
                                    original_subset_med_lls,
                                    color="green",
                                    label="original",
                                )
                                ax_curve.scatter(
                                    int_reduced_intensities,
                                    new_subset_med_lls,
                                    color="blue",
                                    label="generated",
                                )
                                if i == 0 and j == 0:
                                    ax_curve.legend()
                                for src in original_subset_lls.T:
                                    ax_curve.scatter(
                                        int_reduced_intensities,
                                        src,
                                        color="green",
                                        label="_nolegend_",
                                        s=1.5,
                                    )
                                for src in new_subset_lls.T:
                                    ax_curve.scatter(
                                        int_reduced_intensities,
                                        src,
                                        color="blue",
                                        label="_nolegend_",
                                        s=1.5,
                                    )
                                ax_curve.set_title(
                                    f"{len(int_reduced_intensities)} int., {n_subset_replications} rep.: {surr_p_value: .3f}"
                                )
                                ax_curve.set_ylim([-0.1, 1.2])

                            # fit curves
                            r_squared_original_value = np.nan
                            r_squared_new_value = np.nan
                            try:
                                original_subset_params = fit_curve(
                                    curve_function=CURVE["function"],
                                    x=int_reduced_intensities,
                                    y=original_subset_med_lls,
                                    initial_values=CURVE["initial_values"],
                                    bounds=CURVE["bounds"],
                                    max_iterations=MAX_ITERATIONS,
                                )
                                original_subset_y_fit = CURVE["function"](
                                    x_fit, *original_subset_params
                                )
                                original_subset_exi = np.trapezoid(
                                    original_subset_y_fit, x_fit
                                )
                                original_subset_y_pred = CURVE["function"](
                                    int_reduced_intensities, *original_subset_params
                                )
                                original_subset_performance_dict = (
                                    calculate_model_performance(
                                        y=original_subset_med_lls,
                                        y_pred=original_subset_y_pred,
                                        num_params=len(CURVE["initial_values"]) + 1,
                                    )
                                )
                                r_squared_original_value = (
                                    original_subset_performance_dict["r_squared"]
                                )
                                new_subset_params = fit_curve(
                                    curve_function=CURVE["function"],
                                    x=int_reduced_intensities,
                                    y=new_subset_med_lls,
                                    initial_values=CURVE["initial_values"],
                                    bounds=CURVE["bounds"],
                                    max_iterations=MAX_ITERATIONS,
                                )
                                new_subset_y_fit = CURVE["function"](
                                    x_fit, *new_subset_params
                                )
                                new_subset_y_pred = CURVE["function"](
                                    int_reduced_intensities, *new_subset_params
                                )
                                new_subset_performance_dict = (
                                    calculate_model_performance(
                                        y=new_subset_med_lls,
                                        y_pred=new_subset_y_pred,
                                        num_params=len(CURVE["initial_values"]) + 1,
                                    )
                                )
                                r_squared_new_value = new_subset_performance_dict[
                                    "r_squared"
                                ]
                                new_subset_exi = np.trapezoid(new_subset_y_fit, x_fit)
                                if (
                                    r_squared_original_value > SUBSET_CURVE_THRESHOLD
                                ) and (r_squared_new_value > SUBSET_CURVE_THRESHOLD):
                                    delta_exis_row.append(
                                        new_subset_exi - original_subset_exi
                                    )
                                else:
                                    delta_exis_row.append(np.nan)
                                    if PLOT:
                                        ax_curve.set_facecolor("#5a5a5a")
                                if PLOT:
                                    ax_curve.plot(
                                        x_fit,
                                        original_subset_y_fit,
                                        color="green",
                                        label="original",
                                    )
                                    ax_curve.plot(
                                        x_fit,
                                        original_y_fit,
                                        color="green",
                                        label="original",
                                        linestyle=":",
                                    )
                                    ax_curve.plot(
                                        x_fit,
                                        new_subset_y_fit,
                                        color="blue",
                                        label="generated",
                                    )
                                    ax_curve.plot(
                                        x_fit,
                                        new_y_fit,
                                        color="blue",
                                        label="generated",
                                        linestyle=":",
                                    )
                                    ax_hist.set_title(
                                        f"Orig: {effect_size:.2f} / {original_subset_exi-new_subset_exi:.2f}"
                                    )

                            except Exception as e:
                                delta_exis_row.append(np.nan)
                            r_squared_original_row.append(r_squared_original_value)
                            r_squared_new_row.append(r_squared_new_value)

                        delta_exis.append(delta_exis_row)
                        surr_p_values.append(surr_p_values_row)
                        surr_p_values_empirical.append(surr_p_values_empirical_row)
                        r_squared_original.append(r_squared_original_row)
                        r_squared_new.append(r_squared_new_row)
                    result_rows.append(
                        {
                            "patient_id": patient_id,
                            "stim_channel_name": stim_channel_name,
                            "response_channel_name": response_channel_name,
                            "effect_size": effect_size,
                            "param_name": CURVE["param_names"][PARAM_INDEX_TO_CHANGE],
                            "delta_exis": delta_exis,
                            "surr_p_values": surr_p_values,
                            "surr_p_values_empirical": surr_p_values_empirical,
                            "r_squared_original": r_squared_original,
                            "r_squared_new": r_squared_new,
                        }
                    )
                    if PLOT:
                        plt.suptitle(
                            f"Power analysis\nRecording: {patient_id}\nStimulation: {stim_channel_name}\nResponse: {response_channel_name}\nEffect size: {effect_size}, Param. changed: {CURVE['param_names'][PARAM_INDEX_TO_CHANGE]}",
                        )
                        plt.tight_layout(rect=[0, 0, 1, 0.96])
                        plt.savefig(
                            f"{OUT_PATH}/result_{patient_id}_{stim_channel_name}_{response_channel_name}_{effect_size}_{CURVE['param_names'][PARAM_INDEX_TO_CHANGE]}.png"
                        )
                        plt.close("all")

                boostrap_df = pd.DataFrame(result_rows)

                if len(boostrap_df) > 0:
                    json_path = f"{OUT_PATH}/power_analysis_lf.json"
                    lock_path = json_path + ".lock"
                    with FileLock(lock_path):
                        if os.path.exists(json_path):
                            df = pd.read_json(json_path, orient="records")
                            # Remove old entries
                            mask = (
                                (df["patient_id"] == patient_id)
                                & (df["stim_channel_name"] == stim_channel_name)
                                & (df["effect_size"] == effect_size)
                                & (
                                    df["param_name"]
                                    == CURVE["param_names"][PARAM_INDEX_TO_CHANGE]
                                )
                            )

                            # Drop old entries
                            df = df[~mask]

                            # Combine
                            boostrap_df = pd.concat(
                                [df, boostrap_df], ignore_index=True
                            )

                        boostrap_df.to_json(json_path, orient="records", indent=4)
