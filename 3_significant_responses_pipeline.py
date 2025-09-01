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


from connectivity.load import MultipleHDFResponseLoader, get_h5_names_of_patient
from connectivity.analyze import (
    calculate_continuous_line_length,
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


class SignificanceMethod(Enum):
    SURROGATES = 1
    SPEARMAN = 2
    WILCOXON = 3


if __name__ == "__main__":
    load_dotenv()

    DATA_PAPER = True

    n_surrogates = 1000
    r_squared_theshold = 0.6
    n_replications = 12
    if DATA_PAPER:
        base_path = os.getenv("BASE_PATH_PAPER", "/default/path")
    else:
        base_path = os.getenv("BASE_PATH", "/default/path")
    # base_path = "D:/data"
    patients_id = [arg for arg in sys.argv[1:]]  # ["EL027", "EL019", "EL022", "EL026"]
    print(f"Patients: {patients_id}")
    out_path = (
        "output/paper/significant_responses"
        if DATA_PAPER
        else "output/significant_responses"
    )
    sleep_stage = [SleepStage.AWAKE, SleepStage.QWAKE]

    curves = [
        CURVES["5P"],
        CURVES["4P"],
    ]
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
    CLEANING_SURROGATES = False  # True
    CLEANING_SCALE_FACTOR = -1  # 4
    # SIGNIFICANCE_LEVEL = 0.995
    SIGNIFICANCE_LEVEL_SURR = 0.995
    SIGNIFICANCE_LEVEL_WILX = 0.95
    SIGNIFICANCE_LEVEL_SPEAR = 0.995
    METHOD = SignificanceMethod.SPEARMAN

    USE_CACHE = False
    DO_NOT_UPDATE_JSON = False  # FIXME
    USE_MIN_NORMALIZATION = True

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
                "cleaning_surrogates": CLEANING_SURROGATES,
                "cleaning_scale_factor": CLEANING_SCALE_FACTOR,
                "metric": "AUC",
                "significance_level_surr": SIGNIFICANCE_LEVEL_SURR,
                "significance_level_spear": SIGNIFICANCE_LEVEL_SPEAR,
                "significance_level_wilx": SIGNIFICANCE_LEVEL_WILX,
                "method": METHOD.name,
                "use_cache": USE_CACHE,
                "do_not_update_json": DO_NOT_UPDATE_JSON,
                "use_min_normalization": USE_MIN_NORMALIZATION,
            },
            f,
            indent=4,
        )

    results = {}

    json_path = f"{out_path}/response_channels_lf.json"
    labels_available = False
    if os.path.exists(json_path):
        old_df = pd.read_json(json_path, orient="records")  # for labels
        labels_available = True

    for patient_id in patients_id:
        print(f"{pd.Timestamp.now()}: Processing patient {patient_id}")
        names_h5 = get_h5_names_of_patient(
            base_path,
            patient_id,
            protocol="CR",
            new_overview_format=True if DATA_PAPER else False,
        )
        results[patient_id] = {}

        path_lookup = f"{base_path}/{patient_id}/Electrodes/Lookup.xlsx"
        paths_h5 = [
            f"{base_path}/{patient_id}/Electrophy/{name}.h5" for name in names_h5
        ]
        paths_logs = [
            f"{base_path}/{patient_id}/out/{name}_logs.csv" for name in names_h5
        ]
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
        if sleep_stage is not None and len(sleep_stage) > 0:
            mrl.add_sleep_score_to_logs()

        logs = mrl.get_logs()

        io_stim_channels = logs[logs["type"] == "CR_IO"][
            ["name_pos", "name_neg"]
        ].drop_duplicates()
        io_stim_channel_names = io_stim_channels.agg("-".join, axis=1).tolist()
        io_stim_channel_paths = mrl.get_channel_paths_from_names(io_stim_channel_names)
        for chosen_stim_channel, io_stim_channel in io_stim_channels.iterrows():
            stim_channel_name = (
                f"{io_stim_channel['name_pos']}-{io_stim_channel['name_neg']}"
            )

            _, max_n_replications = find_max_n_replications(
                complete_logs=logs,
                selected_stim_channel_name_neg=io_stim_channel["name_neg"],
                selected_stim_channel_name_pos=io_stim_channel["name_pos"],
                stim_protocol="CR_IO",
                sleep_states=sleep_stage,
            )
            if max_n_replications < n_replications:
                print(f"Not enough replications for {patient_id} - {io_stim_channel}")
                continue

            results[patient_id][
                io_stim_channel["name_pos"] + "-" + io_stim_channel["name_neg"]
            ] = {}

            ## PREPARATION
            io_stimlist = filter_logs(
                complete_logs=logs,
                n_replications=n_replications,
                selected_stim_channel_name_pos=io_stim_channel["name_pos"],
                selected_stim_channel_name_neg=io_stim_channel["name_neg"],
                sleep_stages=sleep_stage,
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
                logs[logs["type"] == "CR_IO"]["Int_prob"].drop_duplicates().tolist()
            )
            io_intensities.sort()
            io_intensities.insert(0, 0)

            # CALCULATE UPPER BOUNDS
            fig = plt.figure(figsize=(20, 35))

            surrogate_result = calculate_upper_bounds_using_surrogates_auc(
                # cr_triplet_stims=cr_triplet_stims,
                response_loader=mrl,
                response_channel_paths=channel_paths,
                n_surrogates=n_surrogates,
                n_replications=n_replications,
                intensities=io_intensities,
                fig=fig,
                significance_level=SIGNIFICANCE_LEVEL_SURR,
                sleep_score_restriction=sleep_stage,
                cleaning_factor=CLEANING_SCALE_FACTOR,
                out_file_name=f"{out_path}/null_distribution_{patient_id}_{io_stim_channel['name_pos']}-{io_stim_channel['name_neg']}.npy",
                use_cache=USE_CACHE,
            )
            upper_bounds = surrogate_result["upper_bounds"]["significance_level"]
            upper_bounds_95 = surrogate_result["upper_bounds"]["upper_bounds_95"]
            upper_bounds_99 = surrogate_result["upper_bounds"]["upper_bounds_99"]
            surrogate_auc_matrix = surrogate_result[
                "auc_matrix"
            ]  # shape: (n_surrogates, n_response_channels)

            surrogates_percentiles_dp = surrogate_result["percentiles_dp"]
            surrogates_percentiles_med = surrogate_result["percentiles_agg"]

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(
                f"{out_path}/null_distribution_{patient_id}_{io_stim_channel['name_pos']}-{io_stim_channel['name_neg']}.png"
            )
            plt.suptitle(
                f"{base_path} - {io_stim_channel['name_pos']}-{io_stim_channel['name_neg']} - n_replications={n_replications}"
            )
            plt.close("all")

            ## CALCULATE STIMULUS RESPONSE CURVES

            ll_values, traces, id_matrix = calculate_stimulation_response_curves(
                stimlist=io_stimlist,
                response_loader=mrl,
                selected_stim_channel_name_pos=io_stim_channel["name_pos"],
                selected_stim_channel_name_neg=io_stim_channel["name_neg"],
                selected_channel_paths=channel_paths,
                selected_intensities=io_intensities,
                exclude_responses=CLEAN_DATA,
            )
            # (n_intensities, n_replications, n_response_channel_paths)

            # calculate peak latency and CCEP onset
            peak_latencies, peak_lat_traces = calculate_peak_latency(
                traces=traces, offset_stim_seconds=1, f_sample=mrl.f_sample
            )  # (n_intensities, n_response_channel_paths)

            # calculate baseline LL
            baseline_end_index = round(1 * mrl.f_sample)
            baseline_traces = traces[:, :, :, :baseline_end_index]
            ll_baseline = calculate_pointwise_line_length_max(
                data=baseline_traces, offset_stim_seconds=0.4, f_sample=mrl.f_sample
            )
            ll_med_values = np.nanmedian(
                ll_values, axis=1
            )  # shape (n_intensities, n_response_channel_paths)
            ll_baseline_med_values = np.nanmedian(
                ll_baseline, axis=1
            )  # shape (n_intensities, n_response_channel_paths)

            intensities_reshaped = np.array(io_intensities)[
                :, None
            ]  # shape (n_intensities, 1)

            real_aucs = np.trapezoid(
                y=ll_med_values, x=intensities_reshaped, axis=0
            )  # shape (n_channels)

            ## NORMALIZATION OF LL VALUES
            result_response_channels = []

            normalized_ll_med_values = normalize_ll_values(
                ll_values=ll_med_values,
                # surrogate_ll_percentile_5=surrogates_percentiles_med["5"],
                axis=0,
                use_min=USE_MIN_NORMALIZATION,
            )

            norm_io_intensities = io_intensities / np.max(io_intensities)

            # SIGNAL-TO-NOISE RATIO based on datapoints
            snr = (
                np.nanpercentile(ll_med_values, q=95, axis=(0))
                / surrogates_percentiles_med["95"]
            )

            # RANKS
            ideal_ranks = np.arange(len(io_intensities)) + 1

            spearman_p_values = []
            spearman_rhos = []
            wilcoxon_p_values = []
            surrogate_p_values = []

            for i, response_channel_path in enumerate(channel_paths):
                response_channel_name = response_channel_path.split("/")[-1]

                # SURROGATE P-VALUES
                surrogate_p_value = (
                    np.sum(surrogate_auc_matrix[:, i] >= real_aucs[i]) + 1
                ) / (n_surrogates + 1)
                surrogate_p_values.append(surrogate_p_value)

                spearman_rho, spearman_p_value = scipy.stats.spearmanr(
                    ll_med_values[:, i], ideal_ranks
                )
                spearman_p_values.append(float(spearman_p_value))
                spearman_rhos.append(float(spearman_rho))

                signs = np.sign(
                    ll_med_values[1:, i] - ll_baseline_med_values[1:, i]
                )  # ignore 0mA component
                # One-sided binomial test?
                statistic, wilcoxon_p_value = scipy.stats.wilcoxon(signs)
                wilcoxon_p_values.append(wilcoxon_p_value)

            # FDR CORRECTION
            spearman_p_values_fdr_corrected = scipy.stats.false_discovery_control(
                spearman_p_values, method="bh"
            )
            surrogate_p_values_fdr_corrected = scipy.stats.false_discovery_control(
                surrogate_p_values, method="bh"
            )
            wilcoxon_p_values_fdr_corrected = scipy.stats.false_discovery_control(
                wilcoxon_p_values, method="bh"
            )

            if METHOD == SignificanceMethod.SURROGATES:
                significance = (
                    surrogate_p_values_fdr_corrected < 1 - SIGNIFICANCE_LEVEL_SURR
                )
            elif METHOD == SignificanceMethod.SPEARMAN:
                significance = (
                    spearman_p_values_fdr_corrected < 1 - SIGNIFICANCE_LEVEL_SPEAR
                )
            else:
                assert METHOD == SignificanceMethod.WILCOXON
                significance = (
                    wilcoxon_p_values_fdr_corrected < 1 - SIGNIFICANCE_LEVEL_WILX
                )

            cont_ll = calculate_continuous_line_length(
                data=traces,
                start_index=0,
                end_index=round(2 * mrl.f_sample),
                window_width_indices=round(0.25 * mrl.f_sample),
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
                if labels_available:
                    mask = (
                        (old_df["patient_id"] == patient_id)
                        & (old_df["stim_channel_name"] == stim_channel_name)
                        & (old_df["response_channel_name"] == response_channel_name)
                    )

                    if "label" in old_df.columns:
                        s = old_df.loc[mask, "label"].dropna()
                        manual_label = s.iloc[0] if not s.empty else 0
                    else:
                        manual_label = 0

                ll_window_start = round(
                    1 * mrl.f_sample
                )  # we only want to have the [0, 0.5s] window to display, as it is used for LL calculation
                ll_max_window_offset = round(0.25 * mrl.f_sample)  # max in [0.25, 0.5s]
                ll_window_end = round(1.5 * mrl.f_sample)
                chunk_len = ll_window_end - ll_window_start  # traces.shape[3]
                total_chunks = ll_med_values.shape[0]
                time = np.arange(chunk_len * total_chunks) / mrl.f_sample
                trace_mean = np.nanmean(
                    traces[:, :, i, ll_window_start:ll_window_end], axis=1
                )  # shape: (chunks, chunk_len)
                trace_peak_lat = peak_lat_traces[
                    :, i, ll_window_start:ll_window_end
                ]  # shape: (chunks, chunk_len)

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

                # SURR
                perf_title = f"\nSurr {surrogate_p_values[i]:.2f}/{surrogate_p_values_fdr_corrected[i]:.2f}"
                if surrogate_p_values_fdr_corrected[i] < 1 - SIGNIFICANCE_LEVEL_SURR:
                    perf_title += "✓"
                if (
                    surrogate_p_values_fdr_corrected[i] < 1 - SIGNIFICANCE_LEVEL_SURR
                    and manual_label == -1
                ) or (
                    surrogate_p_values_fdr_corrected[i] >= 1 - SIGNIFICANCE_LEVEL_SURR
                    and manual_label == 2
                ):
                    perf_title += "⚠️"
                # SPEAR
                perf_title += f", Spear {spearman_p_values[i]:.2f}/{spearman_p_values_fdr_corrected[i]:.2f}"
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
                # WILX
                perf_title += f", Wilx {wilcoxon_p_values[i]:.2f}/{wilcoxon_p_values_fdr_corrected[i]:.2f}"
                if wilcoxon_p_values_fdr_corrected[i] < 1 - SIGNIFICANCE_LEVEL_WILX:
                    perf_title += "✓"
                if (
                    wilcoxon_p_values_fdr_corrected[i] < 1 - SIGNIFICANCE_LEVEL_WILX
                    and manual_label == -1
                ) or (
                    wilcoxon_p_values_fdr_corrected[i] >= 1 - SIGNIFICANCE_LEVEL_WILX
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

                if significance[i] == True:
                    n_significant += 1
                    colors = ["blue", "green", "orange", "red"]
                    for j, curve in enumerate(curves):
                        try:
                            params = fit_curve(
                                curve_function=curve["function"],
                                x=norm_io_intensities,
                                y=normalized_ll_med_values[:, i],
                                initial_values=curve["initial_values"],
                                bounds=curve["bounds"],
                                max_iterations=1000,
                            )
                            x_fit = np.linspace(0, 1, 1000)
                            y_fit = curve["function"](x_fit, *params)
                            y_pred = curve["function"](norm_io_intensities, *params)

                            # num params +1 for variance of errors: https://en.wikipedia.org/wiki/Akaike_information_criterion#Counting_parameters
                            performance_dict = calculate_model_performance(
                                y=normalized_ll_med_values[:, i],
                                y_pred=y_pred,
                                num_params=len(curve["initial_values"]) + 1,
                            )

                            if (
                                performance_dict["r_squared"] > r_squared_theshold
                                and curve["name"] == "RRHC"
                            ):
                                n_r_squared_significant += 1

                            ax_lower.plot(
                                x_fit,
                                y_fit,
                                label=f"{curve['name']}: {performance_dict['r_squared']: .2f}",
                                color=colors[j],
                            )

                        except RuntimeError as e:
                            print(f"{response_channel_name}: Optimization failed.")
                    # ax_lower.set_facecolor("#b6fc9d")
                    ax_lower.legend()
                    ax_lower.set_xlabel("Normalized Intensity")
                    ax_lower.set_ylabel("Normalized LL")
                    ax_lower.set_ylim(-0.1, 1.2)
                # else:
                # ax_lower.set_facecolor("#f2c2c2")

                label_color_mapping = {
                    -1: "#fcbaba",  # no response
                    0: "#ffffff",  # no label
                    1: "#fff7aa",  # unsure
                    2: "#c0ffa2",  # response
                }
                label_desc_mapping = {
                    -1: "No response",  # no response
                    0: "No label",  # no label
                    1: "Unsure",  # unsure
                    2: "Response",  # response
                }
                ax_lower.set_facecolor(label_color_mapping[manual_label])

                channel_dict = {
                    "patient_id": patient_id,
                    "stim_channel_name": stim_channel_name,
                    "response_channel_name": response_channel_name,
                    "response_channel_path": response_channel_path,
                    "ll_values": ll_values[:, :, i].tolist(),
                    "med_lls": list(ll_med_values[:, i]),
                    "norm_med_lls": list(
                        normalized_ll_med_values[:, i]
                    ),  # TODO to be removed?
                    # "upper_bound": float(upper_bounds[i]),
                    # "upper_bound_95": float(upper_bounds_95[i]),
                    # "upper_bound_99": float(upper_bounds_99[i]),
                    "is_significant": bool(significance[i]),
                    "snr": float(snr[i]),
                    "spearman_p_value": spearman_p_values[i],
                    "spearman_p_value_fdr_corrected": spearman_p_values_fdr_corrected[
                        i
                    ],
                    "spearman_rho": spearman_rhos[i],
                    "wilcoxon_p_value": wilcoxon_p_values[i],
                    "wilcoxon_p_value_fdr_corrected": wilcoxon_p_values_fdr_corrected[
                        i
                    ],
                    "surrogate_p_value": surrogate_p_values[i],
                    "surrogate_p_value_fdr_corrected": surrogate_p_values_fdr_corrected[
                        i
                    ],
                    "surrogates_percentiles_dp": {
                        "min": float(surrogates_percentiles_dp["min"][i]),
                        "5": float(surrogates_percentiles_dp["5"][i]),
                        "med": float(surrogates_percentiles_dp["med"][i]),
                        "95": float(surrogates_percentiles_dp["95"][i]),
                    },
                    "surrogates_percentiles_med": {
                        "min": float(surrogates_percentiles_med["min"][i]),
                        "5": float(surrogates_percentiles_med["5"][i]),
                        "med": float(surrogates_percentiles_med["med"][i]),
                        "95": float(surrogates_percentiles_med["95"][i]),
                    },
                    "id_matrix": id_matrix[:, :, i].tolist(),
                    # "curve_fittings": {},
                }

                result_response_channels.append(channel_dict)

            # plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.suptitle(
                f"{base_path} - {stim_channel_name} - n_replications={n_replications} \n"
                + f"n_responses={len(channel_paths)}, n_significant={np.sum(significance)}, method={METHOD.name}"
            )
            plt.savefig(f"{out_path}/responses_{patient_id}_{stim_channel_name}.png")
            plt.close()

            json_path = f"{out_path}/response_channels_lf.json"
            lock_path = json_path + ".lock"

            with FileLock(lock_path):
                if os.path.exists(json_path):
                    df = pd.read_json(json_path, orient="records")
                    # Remove old entries
                    mask = (df["patient_id"] == patient_id) & (
                        df["stim_channel_name"] == stim_channel_name
                    )

                    # Backup existing labels if they exist
                    if "label" in df.columns:
                        key_cols = [
                            "patient_id",
                            "stim_channel_name",
                            "response_channel_name",
                        ]
                        labels = df.loc[mask, key_cols + ["label"]]
                    else:
                        labels = None

                    # Drop old entries
                    df = df[~mask]

                    # Create new DataFrame
                    new_df = pd.DataFrame(result_response_channels)

                    # Restore labels
                    if labels is not None:
                        new_df = new_df.merge(labels, on=key_cols, how="left")

                    # Combine
                    df = pd.concat([df, new_df], ignore_index=True)
                else:
                    df = pd.DataFrame(result_response_channels)

                # Write back if allowed
                if not DO_NOT_UPDATE_JSON:
                    df.to_json(json_path, orient="records", indent=4)
