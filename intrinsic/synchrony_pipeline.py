import json
import sys
import os
from dotenv import load_dotenv
import math

from matplotlib import gridspec
from matplotlib.pylab import Enum
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from filelock import FileLock
import scipy.stats


from connectivity.load import MultipleHDFResponseLoader, get_h5_names_of_patient
from connectivity.analyze import (
    calculate_continuous_line_length,
    calculate_ll_baseline,
    calculate_peak_latency,
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


load_dotenv()

n_surrogates = 1000
r_squared_theshold = 0.6
n_replications = 12

base_path = os.getenv("BASE_PATH_PAPER", "/default/path")

patients_id = [arg for arg in sys.argv[1:]]  # ["EL027", "EL019", "EL022", "EL026"]
print(f"Patients: {patients_id}")
out_path = "output/synchrony"

sleep_stage = [SleepStage.AWAKE, SleepStage.QWAKE]

curves = [
    CURVES["5P"],
    CURVES["4P"],
]
PROTOCOL = "CR_IO"  # Ph_IO
PROTOCOL_SHORT = "CR"  # Ph
CLEAN_DATA = True
RESPONSES_FILE = "output/significant_responses/response_channels_lf.json"
CLEAN_DATA_FILE = f"out/clean_{PROTOCOL}/bad_responses_dict.json"  # "out/clean/bad_responses_dict.json"

FREQUENCY_BANDS = [
    [50, 100],
    [100, 200],
    # [200, 400], not feasible with current data
]

PLOT = False

import multiprocessing as mp

mp.freeze_support()

with open(f"{out_path}/params_{'_'.join(patients_id)}.json", "w") as f:
    json.dump(
        {
            "n_surrogates": n_surrogates,
            "r_squared_theshold": r_squared_theshold,
            "n_replications": n_replications,
            "base_path": base_path,
            "patients_id": patients_id,
            "sleep_stage": [stage.name for stage in sleep_stage],
            "curves": [curve["name"] for curve in curves],
            "ll_first": True,
            "clean_data": CLEAN_DATA,
            "clean_data_file": CLEAN_DATA_FILE,
            "protocol": PROTOCOL,
            "protocol_short": PROTOCOL_SHORT,
            "frequency_bands": FREQUENCY_BANDS,
        },
        f,
        indent=4,
    )

results = {}
responses_df = pd.read_json(RESPONSES_FILE, orient="records")

for patient_id in patients_id:
    peak_synchrony_results = []

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

    logs = mrl.get_logs()
    logs = logs[logs["noise"] == 0]

    patient_responses_df = responses_df[responses_df["patient_id"] == patient_id]

    all_stim_ids = [
        stim_id
        for matrix in patient_responses_df["id_matrix"]
        for row in matrix[1:]
        for stim_id in row  # ignore 0mA pseudo-stim
    ]
    filtered_logs = logs[logs["stim_id"].isin(all_stim_ids)]

    for i_stim, stim_row in filtered_logs[0:].iterrows():  # FIXME for testing only
        stim_id = stim_row["stim_id"]
        stim_channel_name = stim_row["name_pos"] + "-" + stim_row["name_neg"]

        # get response channels
        response_channel_paths = mrl.get_channel_paths(
            exclude_noisy_channels=True,
            exclude_out_channels=True,
            exclude_stim_channels=True,
            exclude_wm_only_channels=True,
            stim_channel_name_pos=stim_row["name_pos"],
            stim_channel_name_neg=stim_row["name_neg"],
        )

        # get traces
        traces = mrl.get_responses(
            stim_indices=[i_stim],
            response_channel_paths=response_channel_paths,
            t_start=-1,
            t_stop=0,
            overwrite_excluded_recordings=True,
        ).squeeze()  # shape (n_channels, n_times)

        if PLOT:
            n_rows = 1 + 3 * len(FREQUENCY_BANDS)

            fig = plt.figure(figsize=(10, 3 * n_rows), constrained_layout=True)
            gs = gridspec.GridSpec(n_rows, 1, figure=fig)

            time = np.arange(traces.shape[1]) / mrl.f_sample - 1  # in seconds
            # 1. raw traces
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(time, traces.T, alpha=0.5, linewidth=0.5)
            ax.set_title("Raw traces")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("EEG [uV]")
            ax.margins(x=0)

        mean_synchrony_result = {}

        # filter
        for i_freq, band in enumerate(FREQUENCY_BANDS):
            sos_bandpass = scipy.signal.butter(
                4, [band[0], band[1]], fs=mrl.f_sample, btype="bandpass", output="sos"
            )
            filtered_traces = scipy.signal.sosfiltfilt(sos_bandpass, traces, axis=-1)

            analytic = scipy.signal.hilbert(filtered_traces, axis=-1)
            phase = np.angle(analytic)

            kuramoto_order_parameter = np.abs(np.mean(np.exp(1j * phase), axis=0))
            mean_synchrony = np.mean(kuramoto_order_parameter)

            if PLOT:
                ax_upper = fig.add_subplot(gs[1 + 3 * i_freq, 0])
                ax_upper.set_title(f"Filtered traces {band[0]}-{band[1]} Hz")
                ax_upper.set_xlabel("Time [s]")
                ax_upper.set_ylabel("EEG [uV]")
                ax_upper.plot(time, filtered_traces.T, alpha=0.5, linewidth=0.5)
                ax_upper.margins(x=0)

                ax_middle = fig.add_subplot(gs[2 + 3 * i_freq, 0])
                # ax_middle.set_title(f"Phase {band[0]}-{band[1]} Hz")
                # ax_middle.set_xlabel("Time [s]")
                # ax_middle.set_ylabel("Phase [rad]")
                # ax_middle.plot(time, phase.T, alpha=0.5, linewidth=0.5)
                # ax_middle.margins(x=0)
                channels = np.arange(phase.shape[0])
                ax_middle.pcolormesh(
                    time,
                    channels,
                    phase,
                    cmap="twilight_shifted",
                    vmin=-np.pi,
                    vmax=np.pi,
                    shading="auto",
                )
                ax_middle.set_title("Phase across channels")
                ax_middle.set_xlabel("Time [s]")
                ax_middle.set_ylabel("Channel")

                ax_lower = fig.add_subplot(gs[3 + 3 * i_freq, 0])
                ax_lower.set_title(
                    f"Kuramoto Order Parameter {band[0]}-{band[1]} Hz, Mean synchrony: {mean_synchrony:.4f}"
                )
                ax_lower.set_xlabel("Time [s]")
                ax_lower.plot(time, kuramoto_order_parameter)
                ax_lower.margins(x=0)
            print(mean_synchrony)
            mean_synchrony_result[f"{band[0]}-{band[1]}_Hz"] = mean_synchrony

        peak_synchrony_results.append(
            {
                "patient_id": patient_id,
                "stim_channel_name": stim_channel_name,
                "stim_id": stim_id,
                "intensity_mA": stim_row["Int_prob"],
                **mean_synchrony_result,
            }
        )

        if PLOT:
            fig.suptitle(
                f"Patient {patient_id}, stim channel name: {stim_channel_name}, n_resp: {len(response_channel_paths)}\nstim: {stim_id})",
            )
            plt.savefig(
                f"{out_path}/{patient_id}_{stim_id.replace('/', '_')}_synchrony.png",
            )
            plt.close(fig)

    peak_synchrony_df = pd.DataFrame(peak_synchrony_results)
    print("write")
    print(peak_synchrony_df.head())

    peak_synchrony_file = f"{out_path}/peak_synchrony_lf.json"
    lock_path = peak_synchrony_file + ".lock"
    with FileLock(lock_path):
        if os.path.exists(peak_synchrony_file):
            df = pd.read_json(peak_synchrony_file, orient="records")
            # Remove old entries
            mask = df["patient_id"] == patient_id

            # Drop old entries
            df = df[~mask]

            # Combine
            peak_synchrony_df = pd.concat([df, peak_synchrony_df], ignore_index=True)

        peak_synchrony_df.to_json(peak_synchrony_file, orient="records", indent=4)
