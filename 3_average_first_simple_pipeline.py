import json
import sys
import os
from dotenv import load_dotenv
import math

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from filelock import FileLock


from connectivity.load import MultipleHDFResponseLoader, get_h5_names_of_patient
from connectivity.analyze import (
    calculate_AUC,
    calculate_pointwise_line_length_max,
    find_max_n_replications,
    filter_logs,
    normalize_ll_values,
)
from connectivity.enums import SleepStage


if __name__ == "__main__":
    load_dotenv()

    DATA_PAPER = True
    if DATA_PAPER:
        base_path = os.getenv("BASE_PATH_PAPER", "/default/path")
    else:
        base_path = os.getenv("BASE_PATH", "/default/path")
    patients_id = [arg for arg in sys.argv[1:]]  # ["EL027", "EL019", "EL022", "EL026"]
    print(f"Patients: {patients_id}")
    OUT_PATH = (
        "output/paper/average_first_simple"
        if DATA_PAPER
        else "output/average_first_simple"
    )
    # if not os.path.exists(OUT_PATH):
    #    os.makedirs(OUT_PATH)
    SLEEP_STAGE = [SleepStage.AWAKE, SleepStage.QWAKE]

    N_REPLICATIONS = 12
    N_SUBSET_REPLICATIONS = 3
    N_ITERATIONS = 1000
    USE_CACHE_SURROGATES = True
    USE_CACHE = False
    CLEAN_DATA = True
    CLEAN_DATA_FILE = (
        (
            "out/clean_CR_IO/bad_responses_dict.json"  # "out/clean/bad_responses_dict.json"
        )
        if DATA_PAPER
        else (
            "out/clean/bad_responses_dict.json"  # "out/clean/bad_responses_dict.json"
        )
    )
    RESPONSES_FILE = (
        "output/paper/significant_responses/response_channels_lf.json"
        if DATA_PAPER
        else "output/significant_responses/response_channels_lf.json"
    )
    USE_MIN_NORMALIZATION = True

    import multiprocessing as mp

    mp.freeze_support()

    with open(f"{OUT_PATH}/params_{'_'.join(patients_id)}.json", "w") as f:
        json.dump(
            {
                "n_replications": N_REPLICATIONS,
                "n_subset_replications": N_SUBSET_REPLICATIONS,
                "n_iterations": N_ITERATIONS,
                "base_path": base_path,
                "patients_id": patients_id,
                "sleep_stage": [stage.name for stage in SLEEP_STAGE],
                "metric": "AUC",
                "use_cache": USE_CACHE,
                "use_cache_surrogates": USE_CACHE_SURROGATES,
                "clean_data": CLEAN_DATA,
                "clean_data_file": CLEAN_DATA_FILE,
                "use_min_normalization": USE_MIN_NORMALIZATION,
            },
            f,
            indent=4,
        )

    response_channels_result = pd.read_json(RESPONSES_FILE, orient="records")

    for patient_id in patients_id:
        print(f"Processing {patient_id}")
        ### PREPARATION

        names_h5 = get_h5_names_of_patient(
            base_path,
            patient_id,
            protocol="CR",
            new_overview_format=True if DATA_PAPER else False,
        )

        if CLEAN_DATA:
            path_excluded_responses = f"{base_path}/{patient_id}/{CLEAN_DATA_FILE}"
        else:
            path_excluded_responses = None

        mrl = MultipleHDFResponseLoader(
            recording_names=names_h5,
            paths_h5=[
                f"{base_path}/{patient_id}/Electrophy/{name_h5}.h5"
                for name_h5 in names_h5
            ],
            paths_logs=[
                f"{base_path}/{patient_id}/out/{name_h5}_logs.csv"
                for name_h5 in names_h5
            ],
            path_lookup=f"{base_path}/{patient_id}/Electrodes/Lookup.xlsx",
            path_excluded_responses=path_excluded_responses,
        )
        mrl.add_sleep_score_to_logs()

        logs = mrl.get_logs()
        io_logs = logs[logs["type"] == "CR_IO"]
        io_stim_channels = io_logs[io_logs["type"] == "CR_IO"][
            ["name_pos", "name_neg"]
        ].drop_duplicates()[:]

        all_intensities = (
            io_logs[io_logs["type"] == "CR_IO"]["Int_prob"].sort_values().unique()
        )
        all_response_channel_paths = mrl.get_channel_paths(
            exclude_noisy_channels=True,
            exclude_wm_only_channels=True,
            exclude_out_channels=True,
        )
        results = {}
        for selected_stim_channel_index in range(len(io_stim_channels)):
            selected_stim_channel = io_stim_channels.iloc[selected_stim_channel_index]
            stim_channel_name = (
                selected_stim_channel["name_pos"]
                + "-"
                + selected_stim_channel["name_neg"]
            )

            response_channel_paths = mrl.get_channel_paths(
                exclude_noisy_channels=True,
                stim_channel_name_pos=selected_stim_channel["name_pos"],
                stim_channel_name_neg=selected_stim_channel["name_neg"],
                exclude_stim_channels=True,
                exclude_wm_only_channels=True,
                exclude_out_channels=True,
            )

            results[stim_channel_name] = {
                "exis_ll_first": [],
                "exis_avg_first": [],
                "p": 0,
                "response_channel_paths": list(response_channel_paths),
            }

            print(
                f" ... working on stim channel: {stim_channel_name}",
            )

            _, max_n_replications = find_max_n_replications(
                complete_logs=logs,
                selected_stim_channel_name_neg=selected_stim_channel["name_neg"],
                selected_stim_channel_name_pos=selected_stim_channel["name_pos"],
                stim_protocol="CR_IO",
                sleep_states=SLEEP_STAGE,
            )
            if max_n_replications < N_REPLICATIONS:
                print(
                    f"Not enough replications for {patient_id} - {io_stim_channels.iloc[selected_stim_channel_index]}"
                )
                continue

            filtered_logs = filter_logs(
                complete_logs=logs,
                n_replications=N_REPLICATIONS,
                selected_stim_channel_name_pos=selected_stim_channel["name_pos"],
                selected_stim_channel_name_neg=selected_stim_channel["name_neg"],
                stim_protocol="CR_IO",
                triplet_protocol="CR_triplet",
                sleep_stages=SLEEP_STAGE,
            )

            # FIXME only for EL019 (= old nels)
            if patient_id == "EL019":
                filtered_logs_triplets = logs[
                    (logs["type"] == "CR_BM") & (logs["IPI_ms"] == 100.0)
                ][:N_REPLICATIONS]
            else:
                filtered_logs_triplets = filtered_logs[
                    filtered_logs["type"] == "CR_triplet"
                ][:N_REPLICATIONS]

            filtered_logs = filtered_logs[filtered_logs["type"] != "CR_triplet"]
            intensities = filtered_logs["Int_prob"].sort_values().unique()

            #### GET RESPONSE TRACES
            # get responses
            for response_channel_path in response_channel_paths:
                response_channel_name = response_channel_path.split("/")[-1]
                response_res = response_channels_result[
                    (response_channels_result["patient_id"] == patient_id)
                    & (
                        response_channels_result["stim_channel_name"]
                        == stim_channel_name
                    )
                    & (
                        response_channels_result["response_channel_name"]
                        == response_channel_name
                    )
                ]
                assert (
                    len(response_res) == 1
                ), f"More than one row for {patient_id}: {stim_channel_name}/{response_channel_name}"
                response_res = response_res.squeeze()

            traces = []
            id_matrix = []

            for intensity in intensities:
                trace, stim_ids = mrl.get_responses(
                    stim_indices=filtered_logs[
                        filtered_logs["Int_prob"] == intensity
                    ].index,
                    response_channel_paths=response_channel_paths,
                    t_start=-1,
                    t_stop=1,
                    return_stim_ids=True,
                )
                traces.append(trace)
                id_matrix.append(stim_ids)

            # 0mA
            subset_stimlist = filtered_logs.sample(
                n=N_REPLICATIONS, replace=False, random_state=42
            )  # random state for reproducibility

            # get traces from -1.6 up to 0.6
            stim_indices = subset_stimlist.index.tolist()
            assert len(stim_indices) > 0, "No stimulations found for intensity 0 mA. "
            zero_res, zero_stim_ids = mrl.get_responses(
                stim_indices=stim_indices,
                response_channel_paths=response_channel_paths,
                overwrite_excluded_recordings=CLEAN_DATA,
                t_start=-1 - 0.6,  # -0.6 w.r.t. stimulation time
                t_stop=1 - 0.6,
                return_stim_ids=True,
            )
            zero_stim_ids = zero_stim_ids + "-0.6s"
            if np.isnan(np.min(zero_res)):
                print("Warning: NaN values in traces for intensity 0mA")

            traces.insert(0, zero_res)
            id_matrix.insert(0, zero_stim_ids)

            traces = np.array(traces)
            id_matrix = np.array(id_matrix)
            intensities = np.insert(intensities, 0, 0)  # add 0mA component
            norm_intensities = intensities / np.max(intensities)

            traces = np.array(traces)
            id_matrix = np.array(id_matrix)

            avg_traces = np.nanmean(traces, axis=1)
            # calculate line lengths on single trials, calculate average on multiple averages
            single_ll_values = calculate_pointwise_line_length_max(
                data=traces, offset_stim_seconds=1, f_sample=mrl.f_sample
            )
            ll_ll_first = np.nanmedian(
                single_ll_values, axis=1
            )  # shape: (n_intensties, n_reponses)
            # ll_ll_first[0] = surrogate_result["percentiles_dp"]["med"][indices]
            norm_ll_ll_first = normalize_ll_values(
                ll_values=ll_ll_first, axis=0, use_min=USE_MIN_NORMALIZATION
            )

            ll_avg_first = calculate_pointwise_line_length_max(
                data=avg_traces, offset_stim_seconds=1, f_sample=mrl.f_sample
            )
            norm_ll_avg_first = normalize_ll_values(
                ll_values=ll_avg_first, axis=0, use_min=USE_MIN_NORMALIZATION
            )
            # ll_avg_first[0] = surrogate_result_avg_first["percentiles_agg"]["med"][
            #     indices
            # ]

            aucs_avg_first = np.trapezoid(y=ll_avg_first, x=intensities, axis=0)
            aucs_ll_first = calculate_AUC(
                ll_values=single_ll_values, intensities=intensities
            )

            exis_avg_first = np.trapezoid(
                y=norm_ll_avg_first, x=norm_intensities, axis=0
            )

            exis_ll_first = np.trapezoid(y=norm_ll_ll_first, x=norm_intensities, axis=0)

            print(exis_ll_first.shape)

            stat, p = scipy.stats.wilcoxon(exis_avg_first, exis_ll_first)
            results[stim_channel_name]["p"] = float(p)

            results[stim_channel_name]["exis_avg_first"] = exis_avg_first.tolist()
            results[stim_channel_name]["exis_ll_first"] = exis_ll_first.tolist()

            #### CONTROL PLOT
            # create grid spec
            n_cols = 6
            n_plots = traces.shape[2]
            n_rows = math.ceil(n_plots / n_cols)
            fig = plt.figure(figsize=(20, n_rows * 4))
            gs = GridSpec(n_rows, n_cols, figure=fig)

            for i, channel in enumerate(response_channel_paths):
                row, col = divmod(i, n_cols)  # Determine row and column
                ax = fig.add_subplot(gs[row, col])
                ax.set_title(
                    f"{channel.split('/')[-1]}\nMag-first: {exis_ll_first[i]:.2f}, Avg-first: {exis_avg_first[i]:.2f}",
                    # color=(
                    #     "red"
                    #     if real_aucs_avg_first[i]
                    #     < upper_bounds_95_avg_first[index_mapping[i]]
                    #     else "green"
                    # ),
                )
                for j in range(traces.shape[1]):
                    # iterate through replications
                    ax.scatter(intensities, single_ll_values[:, j, i], s=2)
                ax.plot(intensities, ll_avg_first[:, i], color="black")
                ax.plot(intensities, ll_ll_first[:, i], color="grey")
                # is_significant_response = real_aucs[i] > upper_bounds[index_mapping[i]]
                # if not is_significant_response:
                #     ax.set_facecolor(color="lightcoral")
                ax.set_xlabel("Intensity [mA]")
                ax.set_ylabel("Line length")

            fig.suptitle(
                f"{patient_id} - Stim. channel: {selected_stim_channel['name_pos']} - {selected_stim_channel['name_neg']}, n_replications: {N_REPLICATIONS}"
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(
                f"{OUT_PATH}/comparison_{patient_id}_{selected_stim_channel['name_pos']}-{selected_stim_channel['name_neg']}.png"
            )
            plt.close()

            #### CONTROL PLOT NORMALIZED
            # create grid spec
            n_cols = 6
            n_plots = traces.shape[2]
            n_rows = math.ceil(n_plots / n_cols)
            fig = plt.figure(figsize=(20, n_rows * 4))
            gs = GridSpec(n_rows, n_cols, figure=fig)

            comparison = exis_ll_first > exis_avg_first

            for i, channel in enumerate(response_channel_paths):
                row, col = divmod(i, n_cols)  # Determine row and column
                ax = fig.add_subplot(gs[row, col])
                ax.set_title(
                    f"{channel.split('/')[-1]}\nMag-first: {exis_ll_first[i]:.2f}, Avg-first: {exis_avg_first[i]:.2f}",
                    # color=(
                    #     "red"
                    #     if real_aucs_avg_first[i]
                    #     < upper_bounds_95_avg_first[index_mapping[i]]
                    #     else "green"
                    # ),
                )
                ax.plot(
                    norm_intensities,
                    norm_ll_avg_first[:, i],
                    color="black",
                    label="Avg-first",
                )
                ax.plot(
                    norm_intensities,
                    norm_ll_ll_first[:, i],
                    color="grey",
                    label="Mag-first",
                )
                # is_significant_response = real_aucs[i] > upper_bounds[index_mapping[i]]
                # if not is_significant_response:
                #     ax.set_facecolor(color="lightcoral")
                ax.set_xlabel("Normalized intensity [mA]")
                ax.set_ylabel("Normalized LL")
                if i == 0:
                    ax.legend()
            fig.suptitle(
                f"{patient_id} - Stim. channel: {selected_stim_channel['name_pos']} - {selected_stim_channel['name_neg']}, n_replications: {N_REPLICATIONS}\n"
                + f"Mag-first larger: {np.sum(comparison)}, Avg-first larger:  {len(comparison) - np.sum(comparison)}"
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(
                f"{OUT_PATH}/norm_comparison_{patient_id}_{selected_stim_channel['name_pos']}-{selected_stim_channel['name_neg']}.png"
            )
            plt.close()

        pat_exis_ll_first = []
        pat_exis_avg_first = []

        # Combine data into list of groups
        data = []
        labels = []
        for stim_channel_name, res in results.items():
            if len(res["exis_ll_first"]) > 0 and len(res["exis_avg_first"]):
                labels.append(stim_channel_name + f"Mag-f ({res['p']:.3f})")
                data.append(res["exis_ll_first"])
                pat_exis_ll_first.extend(res["exis_ll_first"])
                labels.append(stim_channel_name + "Avg-f")
                data.append(res["exis_avg_first"])
                pat_exis_avg_first.extend(res["exis_avg_first"])
        stat, p_pat = scipy.stats.wilcoxon(pat_exis_ll_first, pat_exis_avg_first)
        print("Patient-p", p_pat)
        # Create the violin plot
        plt.figure(figsize=(8, 4))
        parts = plt.violinplot(
            data, showmeans=False, showmedians=True, showextrema=True
        )

        # Customize x-axis
        plt.xticks(np.arange(len(labels)) + 1, labels, rotation=90)
        plt.ylabel("Value")
        plt.title(f"{patient_id} - p-value patient-level {p_pat}")
        plt.grid(axis="y")

        plt.tight_layout()
        plt.savefig(f"{OUT_PATH}/violin_{patient_id}.png")
        plt.close()

        json_path = f"{OUT_PATH}/average_first_results.json"
        lock_path = json_path + ".lock"

        with FileLock(lock_path):
            # Read existing data or start fresh
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    results_dict = json.load(f)
            else:
                results_dict = {}

            # Perform your update
            key = f"{selected_stim_channel['name_pos']}-{selected_stim_channel['name_neg']}"
            if patient_id not in results_dict:
                results_dict[patient_id] = results

            # Write back
            with open(json_path, "w") as f:
                json.dump(results_dict, f, indent=4)
