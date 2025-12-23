import json
import sys
import os
from dotenv import load_dotenv
import math

from matplotlib.pylab import Enum
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from filelock import FileLock
import scipy.stats


from connectivity.load import (
    MultipleHDFResponseLoader,
    get_h5_names_of_patient,
    parsed_list_to_numpy_array,
)
from connectivity.analyze import (
    calculate_continuous_line_length,
    calculate_ll_baseline,
    calculate_peak_latency,
    calculate_peak_latency_1,
    calculate_pointwise_line_length_max,
    calculate_upper_bounds_using_surrogates_auc,
    find_max_n_replications,
    filter_logs,
    calculate_stimulation_response_curves,
    fit_curve,
    calculate_model_performance,
    normalize_ll_values,
)
from connectivity.plot import plot_response_stimulation_curves
from connectivity.enums import SleepStage
from connectivity.curves import CURVES


if __name__ == "__main__":
    load_dotenv()

    BASE_PATH = os.getenv("BASE_PATH_PAPER", "/default/path")
    PATIENTS_ID = [arg for arg in sys.argv[1:]]
    print(f"Patients: {PATIENTS_ID}")
    OUT_PATH = "output/peak_and_onset"
    sleep_stage = [SleepStage.AWAKE, SleepStage.QWAKE]
    CLEAN_DATA = True
    CLEAN_DATA_FILE = (
        "out/clean_CR_IO/bad_responses_dict.json"  # "out/clean/bad_responses_dict.json"
    )
    CONNECTIONS_RESULT_FILE = f"output/significant_responses/response_channels_lf.json"

    with open(f"{OUT_PATH}/params_{'_'.join(PATIENTS_ID)}.json", "w") as f:
        json.dump(
            {
                "base_path": BASE_PATH,
                "patients_id": PATIENTS_ID,
                "sleep_stage": [stage.name for stage in sleep_stage],
                "clean_data": CLEAN_DATA,
                "clean_data_file": CLEAN_DATA_FILE,
                "connections_file": CONNECTIONS_RESULT_FILE,
            },
            f,
            indent=4,
        )

    results = []

    responses_df = pd.read_json(CONNECTIONS_RESULT_FILE, orient="records")

    for patient_id in PATIENTS_ID:
        print(f"{pd.Timestamp.now()}: Processing patient {patient_id}")
        names_h5 = get_h5_names_of_patient(
            BASE_PATH, patient_id, protocol="CR", new_overview_format=True
        )

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
        if sleep_stage is not None and len(sleep_stage) > 0:
            mrl.add_sleep_score_to_logs()

        logs = mrl.get_logs()
        intensities = (
            logs[logs["type"] == "CR_IO"]["Int_prob"].drop_duplicates().tolist()
        )
        intensities.sort()
        intensities.insert(0, 0)
        norm_intensities = np.array(intensities) / np.max(intensities)

        # 1. get patient df
        stim_channels = (
            logs[logs["type"] == "CR_IO"][["name_pos", "name_neg"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        stim_channel_names = stim_channels.agg("-".join, axis=1).tolist()
        stim_channel_paths = mrl.get_channel_paths_from_names(stim_channel_names)

        df_patient = responses_df[
            (responses_df["patient_id"] == patient_id)
            & (responses_df["is_significant"] == True)
        ]
        for stim_channel_index, stim_channel in stim_channels.iterrows():
            # 2. iterate stim channels
            result_rows = []

            stim_channel_name = stim_channel_names[stim_channel_index]
            df_subset = df_patient[df_patient["stim_channel_name"] == stim_channel_name]

            if len(df_subset) > 0:
                fig = plt.figure(
                    figsize=(50, len(df_subset) * 2)
                )  # Adjust figure size as needed

                gs = GridSpec(
                    len(df_subset),
                    4,
                    width_ratios=[7, 0.4, 1, 1],
                    figure=fig,
                )

                for i, (i_row, df_row) in enumerate(df_subset.iterrows()):
                    response_channel_name = df_row["response_channel_name"]
                    response_channel_path = df_row["response_channel_path"]

                    id_matrix = np.array(df_row["id_matrix"])

                    # ignore 0mA
                    ind_matrix = mrl.get_inds_from_stim_ids(
                        id_matrix[1:,]
                    )  # shape (n_replications)

                    traces = []
                    for row in ind_matrix:
                        data = mrl.get_responses(
                            stim_indices=row,
                            response_channel_paths=[response_channel_path],
                            t_start=-1,
                            t_stop=1,
                            overwrite_excluded_recordings=CLEAN_DATA,
                        ).squeeze(
                            1
                        )  # shape: (n_rep, n_time)
                        traces.append(data)
                    traces = np.array(traces)
                    cont_ll = calculate_continuous_line_length(
                        data=traces,
                        start_index=0,
                        end_index=round(mrl.f_sample * 2),
                        window_width_indices=int(mrl.f_sample * 0.04),
                        f_sample=mrl.f_sample,
                    )  # shape (n_intensities, n_replications, n_time)
                    cont_ll = (
                        cont_ll
                        - calculate_ll_baseline(
                            data=traces, f_sample=mrl.f_sample, offset_stim_seconds=1
                        )[:, :, np.newaxis]
                    )
                    med_cont_ll = np.nanmedian(
                        cont_ll[7:], axis=(0, 1)
                    )  # shape (n_time)

                    sos_bandpass = scipy.signal.butter(
                        4, [45], fs=mrl.f_sample, btype="lowpass", output="sos"
                    )
                    filtered_traces = scipy.signal.sosfiltfilt(
                        sos_bandpass, traces, axis=2
                    )

                    ax_left = fig.add_subplot(gs[i, 0])
                    ax_left.set_title(response_channel_name)
                    ax_middle = fig.add_subplot(gs[i, 1])
                    ax_right = fig.add_subplot(gs[i, 2])  # , sharey=ax_left)
                    ax_rightright = fig.add_subplot(gs[i, 3])

                    ll_window_start = round(
                        0.95 * mrl.f_sample
                    )  # we only want to have the [-0.05, 0.55s] window to display, as it is used for LL calculation
                    ll_window_end = round(1.55 * mrl.f_sample)
                    chunk_len = ll_window_end - ll_window_start  # traces.shape[3]
                    total_chunks = traces.shape[0]
                    time = np.arange(chunk_len * total_chunks) / mrl.f_sample
                    trace_mean = scipy.stats.trim_mean(
                        traces[:, :, ll_window_start:ll_window_end],
                        axis=1,
                        proportiontocut=0.1,
                    )  # shape: (chunks, chunk_len)
                    filtered_trace_mean = scipy.stats.trim_mean(
                        filtered_traces[:, :, ll_window_start:ll_window_end],
                        axis=1,
                        proportiontocut=0.1,
                    )  # shape: (chunks, chunk_len)

                    peak_time, peak_polarity, baseline_corrected_traces = (
                        calculate_peak_latency(
                            traces=traces, offset_stim_seconds=1, f_sample=mrl.f_sample
                        )
                    )

                    complete_time = np.linspace(-1, 1, traces.shape[-1])

                    # Plot each chunk with color-coded significance
                    for j in range(total_chunks):
                        # traces
                        start_idx = j * chunk_len
                        end_idx = (j + 1) * chunk_len
                        time_chunk = time[start_idx:end_idx]
                        trace_chunk = trace_mean[j]

                        ax_left.axvline(time[start_idx] + 0.05, color="yellow")
                        if peak_time > 0:
                            ax_left.axvline(
                                time[start_idx] + 0.05 + peak_time, color="red"
                            )
                        for trace in traces[j, :, ll_window_start:ll_window_end]:
                            ax_left.plot(
                                time_chunk,
                                trace,
                                color="black",
                                alpha=0.1,
                                linewidth=0.5,
                            )

                        ax_left.plot(
                            time_chunk,
                            trace_chunk,
                            color="black",
                            linewidth=0.75,
                            linestyle=":",
                        )
                        ax_left.plot(
                            time_chunk,
                            filtered_trace_mean[j],
                            color="black",
                            linewidth=0.75,
                        )

                        start = start_idx / mrl.f_sample
                        end = end_idx / mrl.f_sample
                        ax_left.axvspan(
                            start,
                            end,
                            facecolor=("lightgray" if j % 2 else "white"),
                            alpha=0.3,
                            zorder=0,
                        )
                        ax_left.set_xlabel("Time [s]")
                        ax_left.set_ylabel("EEG [µV]")

                    ax_right.axvline(0, color="yellow")

                    if peak_time > 0:
                        ax_right.axvline(peak_time, color="red")
                        ax_right.set_title(f"Peak latency: {peak_time * 1000:.1f}ms")
                    else:
                        ax_right.set_title(f"Peak latency: n/a")
                    ax_right.plot(
                        complete_time, baseline_corrected_traces, color="black"
                    )

                    ax_right.set_xlim([-0.1, 0.6])
                    ax_right.axhline(5, color="black", linestyle="--", linewidth=0.5)
                    ax_right.axhline(-5, color="black", linestyle="--", linewidth=0.5)
                    ax_right.set_xlabel("Time [s]")
                    ax_right.set_ylabel("z-scored amp. [a.u.]")

                    # mask = med_cont_ll > df_row["surrogates_percentiles_med"]["95"]
                    mask = (
                        np.nanmedian(cont_ll, axis=1)
                        > df_row["surrogates_percentiles_med"]["95"]
                    )  # shape (n_intensities, n_time)
                    # first crossing along time axis
                    first_idx = np.argmax(mask, axis=-1)  # shape (n_intensities,)

                    for ll_int in np.nanmedian(cont_ll, axis=1):
                        ax_rightright.plot(
                            complete_time, ll_int, color="black", alpha=0.3
                        )
                    ax_rightright.plot(
                        complete_time, np.nanmean(cont_ll, axis=(0, 1)), color="black"
                    )
                    ax_rightright.plot(
                        complete_time,
                        np.nanmedian(cont_ll[7:], axis=(0, 1)),
                        color="green",
                    )

                    ax_rightright.axhline(
                        df_row["surrogates_percentiles_med"]["95"],
                        color="red",
                        linestyle="--",
                    )
                    ax_rightright.axhline(
                        df_row["surrogates_percentiles_dp"]["95"],
                        color="black",
                        linestyle="--",
                    )

                    ax_rightright.axvline(0, color="yellow")
                    ax_rightright.axvline(peak_time, color="red")
                    ax_rightright.set_xlim([-0.1, 0.6])

                    # color coding:
                    if peak_time > 0:
                        if peak_time < 0.65:
                            ax_right.set_facecolor("#c0ffa2")
                        else:
                            ax_right.set_facecolor("#fff7aa")
                    else:
                        ax_right.set_facecolor("#ffaaaa")

                    ax_left.margins(x=0)
                    ax_right.margins(x=0)

                    # add SRC
                    ll_values = parsed_list_to_numpy_array(
                        df_row["ll_values"]
                    )  # shape: (n_intensities, n_replications)
                    med_lls = np.nanmedian(ll_values, axis=1)
                    norm_med_lls = normalize_ll_values(
                        med_lls, axis=0, min=0  # BL-corrected
                    )
                    shared_ll_max = np.nanpercentile(med_lls, 95, axis=0)
                    norm_lls = normalize_ll_values(
                        ll_values=ll_values,
                        max=shared_ll_max,
                        min=0,
                        axis=0,
                    )
                    ax_middle.plot(
                        norm_intensities,
                        norm_med_lls,
                        color="purple",
                        label="Norm. med. LL",
                        zorder=7,
                    )
                    for j in range(norm_lls.shape[1]):
                        ax_middle.scatter(
                            norm_intensities,
                            norm_lls[:, j],
                            color="black",
                            alpha=0.5,
                            s=2,
                            zorder=6,
                            label="Norm. LL" if j == 0 else "_nolegend_",
                        )

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
                    if not np.isnan(df_row["label"]):
                        ax_middle.set_facecolor(label_color_mapping[df_row["label"]])
                        ax_middle.set_title(
                            "SRC: manual label - " + label_desc_mapping[df_row["label"]]
                        )
                    ax_middle.set_xlabel("Norm. intensity [a.u.]")
                    ax_middle.set_ylabel("Norm. LL [a.u.]")

                    results.append(
                        {
                            "patient_id": patient_id,
                            "stim_channel_name": stim_channel_name,
                            "response_channel_name": response_channel_name,
                            "peak_latency_ms": (
                                float(peak_time) * 1000 if peak_time > 0 else -1
                            ),
                            "peak_polarity": int(peak_polarity),
                        }
                    )

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.suptitle(f"{BASE_PATH} - {stim_channel_name}")
                plt.savefig(
                    f"{OUT_PATH}/peak_and_onset_{patient_id}_{stim_channel_name}.png"
                )
                plt.close()

    results = pd.DataFrame(results)

    json_path = f"{OUT_PATH}/peak_and_onset_lf.json"
    lock_path = json_path + ".lock"

    with FileLock(lock_path):
        if os.path.exists(json_path):
            existing = pd.read_json(json_path, orient="records")
        else:
            existing = pd.DataFrame()

        key_cols = ["patient_id", "stim_channel_name", "response_channel_name"]

        if not existing.empty:
            combined = pd.concat([existing, results])
            # Drop duplicates, keeping the last (newest) version
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
        else:
            combined = results.copy()

        combined.to_json(json_path, orient="records", indent=4)
