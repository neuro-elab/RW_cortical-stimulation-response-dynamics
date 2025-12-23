import json
import math
import os
import sys
from dotenv import load_dotenv
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
    calculate_stimulation_response_curves,
    fallback_fit_curve,
    filter_logs,
    find_max_n_replications,
    fit_curve,
    normalize_ll_values,
    significant_exi_difference_testing,
)
from connectivity.curves import CURVES
from connectivity.enums import SleepStage
from connectivity.load import (
    MultipleHDFResponseLoader,
    get_h5_names_of_patient,
    parsed_list_to_numpy_array,
)

load_dotenv()


base_path = os.getenv("BASE_PATH_PAPER", "/default/path")
out_path = "output/window_analysis"
patient_ids = sys.argv[1:]  # ["EL010"]  # , "EL014"]

RESPONSES_FILE = f"output/significant_responses/response_channels_lf.json"
N_REPLICATES = 12
SIGNIFICANCE_LEVEL_SPEAR = 0.995
SLEEP_STAGES = [SleepStage.AWAKE, SleepStage.QWAKE]
PROTOCOL = "CR_IO"
PROTOCOL_SHORT = "CR"
CLEAN_DATA = True
CLEAN_DATA_FILE = f"out/clean_{PROTOCOL}/bad_responses_dict.json"  # "out/clean/bad_responses_dict.json"

WINDOWS_TO_TEST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]


responses_df = pd.read_json(RESPONSES_FILE, orient="records")

for patient_id in patient_ids:
    result_rows = []

    with open(f"{out_path}/params_{patient_id}.json", "w") as f:
        json.dump(
            {
                "n_replicates": N_REPLICATES,
                "base_path": base_path,
                "patient_id": patient_id,
                "sleep_stages": [stage.name for stage in SLEEP_STAGES],
                "clean_data": CLEAN_DATA,
                "clean_data_file": CLEAN_DATA_FILE,
                "protocol": PROTOCOL,
                "protocol_short": PROTOCOL_SHORT,
                "windows_to_test": WINDOWS_TO_TEST,
                "significance_level_spear": SIGNIFICANCE_LEVEL_SPEAR,
            },
            f,
            indent=4,
        )

    print(f"{pd.Timestamp.now()}: Processing patient {patient_id}")
    names_h5 = get_h5_names_of_patient(base_path, patient_id, protocol=PROTOCOL_SHORT)
    path_lookup = f"{base_path}/{patient_id}/Electrodes/Lookup.xlsx"
    paths_h5 = [f"{base_path}/{patient_id}/Electrophy/{name}.h5" for name in names_h5]
    paths_logs = [f"{base_path}/{patient_id}/out/{name}_logs.csv" for name in names_h5]
    if CLEAN_DATA:
        path_excluded_responses = f"{base_path}/{patient_id}/{CLEAN_DATA_FILE}"
    else:
        path_excluded_responses = None

    mrl = MultipleHDFResponseLoader(
        paths_h5=paths_h5,
        paths_logs=paths_logs,
        recording_names=names_h5,
        path_lookup=path_lookup,
        path_excluded_responses=path_excluded_responses,
    )
    if SLEEP_STAGES is not None and len(SLEEP_STAGES) > 0:
        mrl.add_sleep_score_to_logs()

        logs = mrl.get_logs()

    io_stim_channels = logs[logs["type"] == PROTOCOL][
        ["name_pos", "name_neg"]
    ].drop_duplicates()
    io_stim_channel_names = io_stim_channels.agg("-".join, axis=1).tolist()
    io_stim_channel_paths = mrl.get_channel_paths_from_names(io_stim_channel_names)
    for chosen_stim_channel, io_stim_channel in io_stim_channels.iterrows():
        for window_width in WINDOWS_TO_TEST:
            stim_channel_name = (
                f"{io_stim_channel['name_pos']}-{io_stim_channel['name_neg']}"
            )

            _, max_n_replications = find_max_n_replications(
                complete_logs=logs,
                selected_stim_channel_name_neg=io_stim_channel["name_neg"],
                selected_stim_channel_name_pos=io_stim_channel["name_pos"],
                stim_protocol=PROTOCOL,
                sleep_states=SLEEP_STAGES,
            )
            print(max_n_replications)
            if max_n_replications < N_REPLICATES:
                print(f"Not enough replications for {patient_id} - {io_stim_channel}")
                continue

            ## PREPARATION
            io_stimlist = filter_logs(
                complete_logs=logs,
                n_replications=N_REPLICATES,
                selected_stim_channel_name_pos=io_stim_channel["name_pos"],
                selected_stim_channel_name_neg=io_stim_channel["name_neg"],
                sleep_stages=SLEEP_STAGES,
                triplet_protocol=None,
            )

            channel_paths = mrl.get_channel_paths(
                exclude_stim_channels=True,
                exclude_noisy_channels=True,
                stim_channel_name_pos=io_stim_channel["name_pos"],
                stim_channel_name_neg=io_stim_channel["name_neg"],
                exclude_wm_only_channels=True,
                exclude_out_channels=True,
            )

            io_intensities = (
                logs[logs["type"] == PROTOCOL]["Int_prob"].drop_duplicates().tolist()
            )
            io_intensities.sort()
            io_intensities.insert(0, 0)

            ## CALCULATE STIMULUS RESPONSE CURVES
            # TODO iterate through windows
            ll_values, traces, id_matrix = calculate_stimulation_response_curves(
                stimlist=io_stimlist,
                response_loader=mrl,
                selected_stim_channel_name_pos=io_stim_channel["name_pos"],
                selected_stim_channel_name_neg=io_stim_channel["name_neg"],
                selected_channel_paths=channel_paths,
                selected_intensities=io_intensities,
                exclude_responses=CLEAN_DATA,
            )
            original_ll_values = ll_values.copy()
            # (n_intensities, n_replications, n_response_channel_paths)

            baseline_ll = calculate_ll_baseline(
                data=traces,
                offset_stim_seconds=1,
                f_sample=mrl.f_sample,
                window_width_seconds=window_width,
            )  # shape: (n_intensities, n_replications, n_response_channel_paths)
            ll_values = ll_values - baseline_ll
            ll_med_values = np.nanmedian(
                ll_values, axis=1
            )  # shape (n_intensities, n_response_channel_paths)

            intensities_reshaped = np.array(io_intensities)[
                :, None
            ]  # shape (n_intensities, 1)

            real_aucs = np.trapezoid(
                y=ll_med_values, x=intensities_reshaped, axis=0
            )  # shape (n_channels)

            normalized_ll_med_values = normalize_ll_values(
                ll_values=ll_med_values, min=0, axis=0  # no min normalization
            )

            norm_io_intensities = io_intensities / np.max(io_intensities)

            # SIGNAL-TO-NOISE RATIO based on surrogates
            # snr = (
            #     np.nanpercentile(ll_med_values, q=95, axis=(0))
            #     / surrogates_percentiles_med["95"]
            # )
            # SNR based on baseline
            baseline_95_percentiles = np.nanpercentile(
                np.nanmedian(baseline_ll, axis=1), q=95, axis=(0)
            )
            snr = (
                np.nanpercentile(np.nanmedian(original_ll_values, axis=1), q=95, axis=0)
                / baseline_95_percentiles
            )

            # RANKS
            ideal_ranks = np.arange(len(io_intensities) - 1) + 1

            spearman_p_values = []
            spearman_rhos = []
            surrogate_p_values = []

            for i, response_channel_path in enumerate(channel_paths):
                response_channel_name = response_channel_path.split("/")[-1]

                spearman_rho, spearman_p_value = scipy.stats.spearmanr(
                    # ignore at 0mA
                    ll_med_values[1:, i],
                    ideal_ranks,
                )
                spearman_p_values.append(float(spearman_p_value))
                spearman_rhos.append(float(spearman_rho))

            # FDR CORRECTION
            spearman_p_values_fdr_corrected = scipy.stats.false_discovery_control(
                spearman_p_values, method="bh"
            )
            surrogate_p_values_fdr_corrected = scipy.stats.false_discovery_control(
                surrogate_p_values, method="bh"
            )

            significance = (
                spearman_p_values_fdr_corrected < 1 - SIGNIFICANCE_LEVEL_SPEAR
            )

            cont_ll = calculate_continuous_line_length(
                data=traces,
                start_index=0,
                end_index=round(2 * mrl.f_sample),
                window_width_indices=round(window_width * mrl.f_sample),
                f_sample=mrl.f_sample,
            )

            n_significant = 0
            n_r_squared_significant = 0

            n_cols = 6
            n_plots = ll_med_values.shape[1]
            n_rows = math.ceil(n_plots / n_cols)
            fig = plt.figure(figsize=(35, 4.5 * 2 * n_rows), constrained_layout=True)
            gs = GridSpec(
                3 * n_rows,
                n_cols,
                figure=fig,
                height_ratios=[1, 1, 3] * n_rows,
            )

            for i, response_channel_path in enumerate(channel_paths):
                response_channel_name = response_channel_path.split("/")[-1]
                destrieux_label = mrl.get_destrieux_labels_from_names(
                    channel_names=[response_channel_name], short_form=True
                )[0]

                row, col = divmod(i, n_cols)
                ax_upper = fig.add_subplot(gs[3 * row, col])
                ax_middle = fig.add_subplot(gs[3 * row + 1, col])
                ax_lower = fig.add_subplot(gs[3 * row + 2, col])

                manual_label = 0
                mask = (
                    (responses_df["patient_id"] == patient_id)
                    & (responses_df["stim_channel_name"] == stim_channel_name)
                    & (responses_df["response_channel_name"] == response_channel_name)
                )

                if "label" in responses_df.columns:
                    s = responses_df.loc[mask, "label"].dropna()
                    manual_label = s.iloc[0] if not s.empty else 0
                else:
                    manual_label = 0

                ll_window_start = round(
                    1 * mrl.f_sample
                )  # we only want to have the [0, 0.5s] window to display, as it is used for LL calculation
                ll_max_window_offset = round(
                    window_width * mrl.f_sample
                )  # max in [0.25, 0.5s]
                ll_window_end = round(1.5 * mrl.f_sample)
                chunk_len = ll_window_end - ll_window_start  # traces.shape[3]
                total_chunks = ll_med_values.shape[0]
                time = np.arange(chunk_len * total_chunks) / mrl.f_sample
                trace_mean = np.nanmean(
                    traces[:, :, i, ll_window_start:ll_window_end], axis=1
                )  # shape: (chunks, chunk_len)

                # Plot each chunk with color-coded significance
                for j in range(total_chunks):
                    # traces
                    start_idx = j * chunk_len
                    end_idx = (j + 1) * chunk_len
                    time_chunk = time[start_idx:end_idx]
                    trace_chunk = trace_mean[j]

                    for trace in traces[j, :, i, ll_window_start:ll_window_end]:
                        ax_upper.plot(
                            time_chunk, trace, color="black", alpha=0.1, linewidth=0.5
                        )
                    ax_upper.plot(
                        time_chunk,
                        trace_chunk,
                        color=("green" if significance[i] else "red"),
                        linewidth=0.75,
                    )

                    start = start_idx / mrl.f_sample
                    end = end_idx / mrl.f_sample
                    ax_upper.axvspan(
                        start,
                        end,
                        facecolor=("lightgray" if j % 2 else "white"),
                        alpha=0.3,
                        zorder=0,
                    )

                    # continous line-length
                    for c_ll in cont_ll[j, :, i, ll_window_start:ll_window_end]:
                        ax_middle.plot(
                            time_chunk, c_ll, color="black", alpha=0.5, linewidth=0.5
                        )
                        max_idx = (
                            c_ll[ll_max_window_offset:].argmax() + ll_max_window_offset
                        )
                        ax_middle.scatter(
                            time_chunk[max_idx], c_ll[max_idx], color="purple", s=10
                        )
                    ax_middle.axvspan(
                        start,
                        end,
                        facecolor=("lightgray" if j % 2 else "white"),
                        alpha=0.3,
                        zorder=0,
                    )

                filtered_ll_values = ll_med_values[:, i]

                # SPEAR
                perf_title = f", Spear {spearman_p_values[i]:.2f}/{spearman_p_values_fdr_corrected[i]:.2f}"
                if spearman_p_values_fdr_corrected[i] < 1 - SIGNIFICANCE_LEVEL_SPEAR:
                    perf_title += "✓"
                if (
                    spearman_p_values_fdr_corrected[i] < 1 - SIGNIFICANCE_LEVEL_SPEAR
                    and manual_label == -1
                ) or (
                    spearman_p_values_fdr_corrected[i] >= 1 - SIGNIFICANCE_LEVEL_SPEAR
                    and manual_label == 2
                ):
                    perf_title += "⚠️"

                ax_upper.margins(x=0, y=0)
                ax_upper.set_title(
                    f"{response_channel_name} ({destrieux_label})" + ", " + perf_title,
                    color="green" if significance[i] else "red",
                )
                ax_upper.set_ylabel("EEG [uV]")
                ax_upper.set_xlabel("Time [s]")
                ax_upper.set_xticks([])

                ax_middle.margins(x=0, y=0)
                ax_middle.set_ylabel("LL [uV/ms]")
                ax_middle.set_xlabel("Time [s]")
                ax_middle.set_xticks([])

                ax_lower.set_title(f"SNR: {snr[i]:.2f}, rho: {spearman_rho: .2f}")
                ax_lower.scatter(
                    norm_io_intensities,
                    normalized_ll_med_values[:, i],
                    c="black",
                    s=5,
                    label="Med. LL",
                )

                ax_lower.legend()
                ax_lower.set_xlabel("Normalized Intensity")
                ax_lower.set_ylabel("Normalized LL")
                ax_lower.set_ylim(-0.1, 1.2)

                label_color_mapping = {
                    -2: "#fcbaba",  # no response, drift
                    -1: "#fcbaba",  # no response
                    0: "#ffffff",  # no label
                    1: "#fff7aa",  # unsure
                    2: "#c0ffa2",  # response
                    3: "#c0ffa2",  # response with decrease
                }
                label_desc_mapping = {
                    -2: "No response/drift",  # no response, drift
                    -1: "No response",  # no response
                    0: "No label",  # no label
                    1: "Unsure",  # unsure
                    2: "Response",  # response
                    3: "Response/decrease",  # response with decrease
                }
                ax_lower.set_facecolor(label_color_mapping[int(manual_label)])

                channel_dict = {
                    "patient_id": patient_id,
                    "window_width_seconds": window_width,
                    "stim_channel_name": stim_channel_name,
                    "response_channel_name": response_channel_name,
                    "response_channel_path": response_channel_path,
                    "med_lls": list(ll_med_values[:, i]),
                    "norm_med_lls": list(normalized_ll_med_values[:, i]),
                    "is_significant": bool(significance[i]),
                    "snr": float(snr[i]),
                    "spearman_p_value": spearman_p_values[i],
                    "spearman_p_value_fdr_corrected": spearman_p_values_fdr_corrected[
                        i
                    ],
                    "spearman_rho": spearman_rhos[i],
                    "baseline_percentiles_med": {
                        "95": float(baseline_95_percentiles[i]),
                    },
                }

                result_rows.append(channel_dict)

            plt.suptitle(
                f"{base_path} - {PROTOCOL} - {stim_channel_name} - n_replications={N_REPLICATES} \n"
                + f"n_responses={len(channel_paths)}, n_significant={np.sum(significance)}, window={window_width}s"
            )
            plt.savefig(
                f"{out_path}/responses_{PROTOCOL}_{patient_id}_{stim_channel_name}_{window_width}s.png"
            )
            plt.close()

    json_path = f"{out_path}/window_width_lf.json"
    lock_path = json_path + ".lock"

    with FileLock(lock_path):
        if os.path.exists(json_path):
            df = pd.read_json(json_path, orient="records")
            # Remove old entries
            mask = df["patient_id"] == patient_id

            # Drop old entries
            df = df[~mask]

            # Create new DataFrame
            new_df = pd.DataFrame(result_rows)

            # Combine
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = pd.DataFrame(result_rows)

        df.to_json(json_path, orient="records", indent=4)
