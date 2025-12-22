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
    calculate_stimulation_response_curves,
    filter_logs,
    find_max_n_replications,
    fit_curve,
    normalize_ll_values,
    significant_exi_difference_testing,
)
from connectivity.curves import CURVES
from connectivity.load import MultipleHDFResponseLoader, get_h5_names_of_patient

base_path = "D:/data_paper"
out_path = "output/pharmaco"
n_replications = 3
SIGNIFICANCE_LEVEL = 0.05
CURVE = CURVES["5P"]
MAX_ITERATIONS = 1000
MAX_ITERATIONS_SURR = 1000

x_fit = np.linspace(0, 1, 1000)

# load

patient_ids = sys.argv[1:]  # ["EL010"]  # , "EL014"]
stim_blocks = {
    "EL008": {"off": 1, "on": 3},
    "EL010": {"off": 1, "on": 3},
    "EL012": {"off": 1, "on": 3},
    "EL014": {"off": 1, "on": 2},
}


for patient_id in patient_ids:
    result_rows = []

    print(f"{pd.Timestamp.now()}: Processing patient {patient_id}")
    names_h5 = get_h5_names_of_patient(base_path, patient_id, protocol="Ph")

    path_lookup = f"{base_path}/{patient_id}/Electrodes/Lookup.xlsx"
    paths_h5 = [f"{base_path}/{patient_id}/Electrophy/{name}.h5" for name in names_h5]
    paths_logs = [f"{base_path}/{patient_id}/out/{name}_logs.csv" for name in names_h5]
    path_excluded_responses = (
        f"{base_path}/{patient_id}/out/clean_Ph_IO/bad_responses_dict.json"
    )

    mrl = MultipleHDFResponseLoader(
        paths_h5=paths_h5,
        paths_logs=paths_logs,
        recording_names=names_h5,
        path_lookup=path_lookup,
        path_excluded_responses=path_excluded_responses,
    )

    logs = mrl.get_logs()

    io_stim_channels = logs[logs["type"] == "Ph_IO"][
        ["name_pos", "name_neg"]
    ].drop_duplicates()
    io_stim_channel_names = io_stim_channels.agg("-".join, axis=1).tolist()
    io_stim_channel_paths = mrl.get_channel_paths_from_names(io_stim_channel_names)
    for chosen_stim_channel, io_stim_channel in io_stim_channels.iterrows():
        stim_channel_name = (
            f"{io_stim_channel['name_pos']}-{io_stim_channel['name_neg']}"
        )

        intensities = (
            logs[logs["type"] == "Ph_IO"]["Int_prob"].drop_duplicates().tolist()
        )
        intensities.sort()
        intensities.insert(0, 0)
        norm_intensities = np.array(intensities) / max(intensities)

        channel_paths = mrl.get_channel_paths(
            exclude_stim_channels=True,
            exclude_noisy_channels=True,
            stim_channel_name_pos=io_stim_channel["name_pos"],
            stim_channel_name_neg=io_stim_channel["name_neg"],
            exclude_wm_only_channels=True,
            exclude_out_channels=True,
        )

        on_ll_values = None
        on_norm_ll_med_values = None
        on_cont_ll_values = None
        on_traces = None
        on_id_matrix = None
        off_ll_values = None
        off_norm_ll_med_values = None
        off_cont_ll_values = None
        off_traces = None
        off_id_matrix = None

        for condition in ["on", "off"]:
            # print(logs["stim_block"].value_counts())
            cond_logs = logs[logs["stim_block"] == stim_blocks[patient_id][condition]]

            _, max_n_replications_on = find_max_n_replications(
                complete_logs=cond_logs,
                selected_stim_channel_name_neg=io_stim_channel["name_neg"],
                selected_stim_channel_name_pos=io_stim_channel["name_pos"],
                stim_protocol="Ph_IO",
                sleep_states=None,
            )
            if n_replications > max_n_replications_on:
                print("Warning: Not enough replicates.")
                continue

            cond_stimlist = filter_logs(
                complete_logs=cond_logs,
                n_replications=n_replications,
                selected_stim_channel_name_pos=io_stim_channel["name_pos"],
                selected_stim_channel_name_neg=io_stim_channel["name_neg"],
                sleep_stages=[],
                triplet_protocol=None,
                stim_protocol="Ph_IO",
            )

            ## CALCULATE STIMULUS RESPONSE CURVES
            ll_values, traces, id_matrix = calculate_stimulation_response_curves(
                stimlist=cond_stimlist,
                response_loader=mrl,
                selected_stim_channel_name_pos=io_stim_channel["name_pos"],
                selected_stim_channel_name_neg=io_stim_channel["name_neg"],
                selected_channel_paths=channel_paths,
                selected_intensities=intensities,
                exclude_responses=True,
                protocol="Ph",
            )
            original_ll_values = ll_values.copy()
            baseline_ll = calculate_ll_baseline(
                data=traces, offset_stim_seconds=1, f_sample=mrl.f_sample
            )  # shape: (n_intensities, n_replications, n_response_channel_paths)
            ll_values = ll_values - baseline_ll

            cont_ll = calculate_continuous_line_length(
                data=traces,
                start_index=0,
                end_index=round(2 * mrl.f_sample),
                window_width_indices=round(0.25 * mrl.f_sample),
                f_sample=mrl.f_sample,
            )

            if condition == "on":
                on_ll_values = ll_values
                on_cont_ll_values = cont_ll
                on_traces = traces
                on_id_matrix = id_matrix
            else:
                off_ll_values = ll_values
                off_cont_ll_values = cont_ll
                off_traces = traces
                off_id_matrix = id_matrix
        combined_ll_med_values = np.nanmedian(
            np.concatenate([on_ll_values, off_ll_values], axis=1), 1
        )
        # shared_max = np.nanpercentile(
        #     combined_ll_med_values,
        #     q=95,
        #     axis=0,
        # )  # shape: (n_channels)

        on_med_ll_values = np.nanmedian(
            on_ll_values, axis=1
        )  # shape: (n_intensities, n_channels)
        off_med_ll_values = np.nanmedian(
            off_ll_values, axis=1
        )  # shape: (n_intensities, n_channels)

        shared_max = np.nanpercentile(
            np.concatenate([on_med_ll_values, off_med_ll_values], axis=0),
            q=95,
            axis=0,
        )  # shape: (n_channels)

        on_norm_ll_med_values = normalize_ll_values(
            ll_values=on_med_ll_values,
            min=0,
            max=shared_max,
            axis=0,  # no min normalization
        )
        off_norm_ll_med_values = normalize_ll_values(
            ll_values=off_med_ll_values,
            min=0,
            max=shared_max,
            axis=0,  # no min normalization
        )
        on_norm_ll_values = normalize_ll_values(
            on_ll_values,
            max=shared_max,
            min=0,
            axis=0,
        )
        off_norm_ll_values = normalize_ll_values(
            off_ll_values,
            max=shared_max,
            min=0,
            axis=0,
        )

        on_spearman_p_values = []
        on_spearman_rhos = []
        off_spearman_p_values = []
        off_spearman_rhos = []
        combined_spearman_p_values = []
        combined_spearman_rhos = []
        ideal_ranks = np.arange(len(intensities)) + 1
        for i, response_channel_path in enumerate(channel_paths):
            response_channel_name = response_channel_path.split("/")[-1]
            on_spearman_rho, on_spearman_p_value = scipy.stats.spearmanr(
                on_norm_ll_med_values[:, i], ideal_ranks
            )
            off_spearman_rho, off_spearman_p_value = scipy.stats.spearmanr(
                off_norm_ll_med_values[:, i], ideal_ranks
            )
            combined_spearman_rho, combined_spearman_p_value = scipy.stats.spearmanr(
                combined_ll_med_values[:, i], ideal_ranks
            )

            on_spearman_p_values.append(float(on_spearman_p_value))
            on_spearman_rhos.append(float(on_spearman_rho))
            off_spearman_p_values.append(float(off_spearman_p_value))
            off_spearman_rhos.append(float(off_spearman_rho))
            combined_spearman_p_values.append(float(combined_spearman_p_value))
            combined_spearman_rhos.append(float(combined_spearman_rho))

        # FDR CORRECTION
        on_spearman_p_values_fdr_corrected = scipy.stats.false_discovery_control(
            on_spearman_p_values, method="bh"
        )
        off_spearman_p_values_fdr_corrected = scipy.stats.false_discovery_control(
            off_spearman_p_values, method="bh"
        )
        combined_spearman_p_values_fdr_corrected = scipy.stats.false_discovery_control(
            combined_spearman_p_values, method="bh"
        )
        significance_on = [
            p < SIGNIFICANCE_LEVEL for p in on_spearman_p_values_fdr_corrected
        ]
        significance_off = [
            p < SIGNIFICANCE_LEVEL for p in off_spearman_p_values_fdr_corrected
        ]
        significance_combined = [
            p < SIGNIFICANCE_LEVEL for p in combined_spearman_p_values_fdr_corrected
        ]

        n_cols = 6
        n_plots = len(channel_paths)
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

            ll_window_start = round(
                1 * mrl.f_sample
            )  # we only want to have the [0, 0.5s] window to display, as it is used for LL calculation
            ll_max_window_offset = round(0.25 * mrl.f_sample)  # max in [0.25, 0.5s]
            ll_window_end = round(1.5 * mrl.f_sample)
            chunk_len = ll_window_end - ll_window_start  # traces.shape[3]
            total_chunks = len(intensities)
            time = np.arange(chunk_len * total_chunks) / mrl.f_sample
            on_trace_mean = np.nanmean(
                on_traces[:, :, i, ll_window_start:ll_window_end], axis=1
            )  # shape: (chunks, chunk_len)
            off_trace_mean = np.nanmean(
                off_traces[:, :, i, ll_window_start:ll_window_end], axis=1
            )  # shape: (chunks, chunk_len)

            # Plot each chunk with color-coded significance
            for j in range(total_chunks):
                # traces
                start_idx = j * chunk_len
                end_idx = (j + 1) * chunk_len
                time_chunk = time[start_idx:end_idx]

                for trace in traces[j, :, i, ll_window_start:ll_window_end]:
                    ax_upper.plot(
                        time_chunk, trace, color="black", alpha=0.1, linewidth=0.5
                    )
                ax_upper.plot(
                    time_chunk,
                    on_trace_mean[j],
                    color="green" if significance_on[i] else "red",
                    linewidth=0.75,
                    linestyle="solid",
                )
                ax_upper.plot(
                    time_chunk,
                    off_trace_mean[j],
                    color="green" if significance_off[i] else "red",
                    linewidth=0.75,
                    linestyle="dotted",
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
                for c_ll in on_cont_ll_values[j, :, i, ll_window_start:ll_window_end]:
                    ax_middle.plot(
                        time_chunk, c_ll, color="green", alpha=0.5, linewidth=0.5
                    )
                    max_idx = (
                        c_ll[ll_max_window_offset:].argmax() + ll_max_window_offset
                    )
                    ax_middle.scatter(
                        time_chunk[max_idx], c_ll[max_idx], color="green", s=10
                    )
                for c_ll in off_cont_ll_values[j, :, i, ll_window_start:ll_window_end]:
                    ax_middle.plot(
                        time_chunk, c_ll, color="red", alpha=0.5, linewidth=0.5
                    )
                    max_idx = (
                        c_ll[ll_max_window_offset:].argmax() + ll_max_window_offset
                    )
                    ax_middle.scatter(
                        time_chunk[max_idx], c_ll[max_idx], color="red", s=10
                    )

                ax_middle.axvspan(
                    start,
                    end,
                    facecolor=("lightgray" if j % 2 else "white"),
                    alpha=0.3,
                    zorder=0,
                )

            ax_upper.margins(x=0, y=0)
            ax_upper.set_title(f"{response_channel_name} ({destrieux_label})" + ", ")
            ax_upper.set_ylabel("EEG [uV]")
            ax_upper.set_xlabel("Time [s]")
            ax_upper.set_xticks([])

            ax_middle.margins(x=0, y=0)
            ax_middle.set_ylabel("LL [uV/ms]")
            ax_middle.set_xlabel("Time [s]")
            ax_middle.set_xticks([])

            ax_lower.set_title(
                f"p-values: ON: {on_spearman_p_values[i]:.3f}/{on_spearman_p_values_fdr_corrected[i]:.3f}, OFF: {off_spearman_p_values[i]:.3f}/{off_spearman_p_values_fdr_corrected[i]:.3f} / COMB: {combined_spearman_p_values[i]:.3f}/{combined_spearman_p_values_fdr_corrected[i]:.3f}",
            )
            ax_lower.plot(
                norm_intensities,
                on_norm_ll_med_values[:, i],
                c="green" if significance_on[i] else "red",
                label="ON Med. LL",
                linestyle="solid",
            )
            ax_lower.plot(
                norm_intensities,
                off_norm_ll_med_values[:, i],
                c="green" if significance_off[i] else "red",
                label="OFF Med. LL",
                linestyle="dotted",
            )
            if i == 0:
                ax_lower.legend()

            if significance_combined[i]:
                ax_lower.set_facecolor("#b5f3ba")
            else:
                ax_lower.set_facecolor("#f3aeae")

            res_on = {
                "patient_id": patient_id,
                "stim_channel_name": stim_channel_name,
                "response_channel_name": response_channel_name,
                "condition": "on",
                "ll_values": on_ll_values[:, :, i].tolist(),
                "id_matrix": on_id_matrix[:, :, i].tolist(),
                "norm_ll_values": on_norm_ll_values[:, :, i].tolist(),
                "norm_ll_med_values": on_norm_ll_med_values[:, i].tolist(),
                "spearman_rho": on_spearman_rhos[i],
                "spearman_p_value": on_spearman_p_values[i],
                "spearman_p_value_fdr_corrected": on_spearman_p_values_fdr_corrected[i],
                "significant": significance_on[i],
                "combined_spearman_p_value": combined_spearman_p_values[i],
                "combined_spearman_p_value_fdr_corrected": combined_spearman_p_values_fdr_corrected[
                    i
                ],
                "combined_significant": significance_combined[i],
            }
            result_rows.append(res_on)
            res_off = {
                "patient_id": patient_id,
                "stim_channel_name": stim_channel_name,
                "response_channel_name": response_channel_name,
                "condition": "off",
                "ll_values": off_ll_values[:, :, i].tolist(),
                "id_matrix": off_id_matrix[:, :, i].tolist(),
                "norm_ll_values": off_norm_ll_values[:, :, i].tolist(),
                "norm_ll_med_values": off_norm_ll_med_values[:, i].tolist(),
                "spearman_rho": off_spearman_rhos[i],
                "spearman_p_value": off_spearman_p_values[i],
                "spearman_p_value_fdr_corrected": off_spearman_p_values_fdr_corrected[
                    i
                ],
                "significant": significance_off[i],
                "combined_spearman_p_value": combined_spearman_p_values[i],
                "combined_spearman_p_value_fdr_corrected": combined_spearman_p_values_fdr_corrected[
                    i
                ],
                "combined_significant": significance_combined[i],
            }
            result_rows.append(res_off)

        plt.savefig(f"{out_path}/responses_{patient_id}_{stim_channel_name}.png")
        plt.close()

    results_df = pd.DataFrame(result_rows)

    results_file = f"{out_path}/response_channels_lf.json"
    lock_path = results_file + ".lock"
    with FileLock(lock_path):
        if os.path.exists(results_file):
            df = pd.read_json(results_file, orient="records")
            # Remove old entries
            mask = df["patient_id"] == patient_id

            # Drop old entries
            df = df[~mask]

            # Combine
            results_df = pd.concat([df, results_df], ignore_index=True)

        results_df.to_json(results_file, orient="records", indent=4)
