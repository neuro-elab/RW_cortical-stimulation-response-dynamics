import math
import os
import time as tm

import numpy as np
import pywt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy
import multiprocessing as mp


from connectivity.load import (
    MultipleHDFResponseLoader,
)
from connectivity.enums import SleepStage, TimeGrade


def calculate_cwt(data: np.ndarray, f_sample: int, freqs: list[float] = None):
    # Define the wavelet
    wavelet = "cmor1.5-2.0"

    # Set scales based on frequencies you are interested in
    if freqs is None:
        freqs = np.logspace(np.log10(4), np.log10(200), num=50)
    scales = pywt.frequency2scale(wavelet, freqs / f_sample)

    # Continuous Wavelet Transform
    coefficients, res_freqs = pywt.cwt(
        data, scales, wavelet, sampling_period=1 / f_sample
    )

    return coefficients, res_freqs


def calculate_plv():
    pass


def calculate_continuous_line_length(
    data: np.ndarray,
    start_index: int,
    end_index: int,
    window_width_indices: int,
    f_sample: int,
):
    """
    Calculate the continouus line-length measure for multidimensional data.
    Supports data of shape (..., n_time), where the last dimension is time.

    Parameters
    ----------
    data : np.ndarray
        Input array with shape (..., n_time).
    start_index : int
        Start index for the computation (along the time axis).
    end_index : int
        End index for the computation (along the time axis).
    window_width_indices : int
        Window size for the line-length computation.
    f_sample : int
        Sampling frequency in Hz.

    Returns
    -------
    ll_data : np.ndarray
        Line-length results with shape (..., line_length_range).
    """
    # Calculate the range of the output along the time axis
    line_length_range = end_index - start_index

    # Determine the shape of the output array
    output_shape = data.shape[:-1] + (line_length_range,)
    ll_data = np.zeros(output_shape)

    # Iterate over the time range and compute line-length for each slice
    for t in range(start_index, end_index):
        # Slice the last (time) axis with the rolling window
        time_slice = data[..., t - window_width_indices : t + 1]

        # Compute the differences along the time axis
        diff = np.abs(np.diff(time_slice, axis=-1))

        # Sum the differences along the time axis, leaving other dimensions intact
        summed_diff = np.sum(diff, axis=-1)

        # Normalize by the window width and scale by sampling frequency
        ll_data[..., t - start_index] = (
            summed_diff / window_width_indices * (f_sample / 1000)
        )

    return ll_data


def calculate_line_length(continuous_ll: np.ndarray, start_index: int, end_index: int):
    """
    continuous_ll in shape (..., ll_time)
    usually (n_replicates, n_channels, n_time)
    """

    return np.max(continuous_ll[..., start_index:end_index], axis=-1)


def calculate_pointwise_line_length_max(
    data: np.ndarray, offset_stim_seconds: float, f_sample: int
):
    """
    Calculate the line-length measure for multidimensional data using the max + continuous method.
    At offset_stim_seconds, we suppose the stimulation starts. We then calculate the LL from
    250ms up to 500ms with a 250ms window. We then take the max value of the LL in this window.

    Parameters
    ----------
    data : np.ndarray
        Input array with shape (..., n_time).
    offset_stim_seconds : float
        Time in seconds where the stimulation starts.
    f_sample : int
        Sampling frequency in Hz.

    Returns
    -------
    ll : float
        Line-length results with shape (...).
    """
    start_index_cll = round(offset_stim_seconds * f_sample)
    end_index_cll = round((offset_stim_seconds + 2) * f_sample)
    start_index_ll = round(0.25 * f_sample)
    end_index_ll = round(0.5 * f_sample)

    cont_ll = calculate_continuous_line_length(
        data=data,
        start_index=start_index_cll,
        end_index=end_index_cll,
        window_width_indices=round(0.25 * f_sample),
        f_sample=f_sample,
    )
    single_ll = calculate_line_length(
        continuous_ll=cont_ll,
        start_index=start_index_ll,
        end_index=end_index_ll,
    )

    return single_ll


def calculate_pointwise_line_length(
    data: np.ndarray,
    at_index: int,
    window_width_indices: int,
    f_sample: int,
):
    """
    Calculate the line-length measure for multidimensional data at a specific index.
    Note that the line-length is calculated starting from window_width_indices before the at_index and up to at_index.
    Supports data of shape (..., n_time), where the last dimension is time.

    Parameters
    ----------
    data : np.ndarray
        Input array with shape (..., n_time).
    at_index : int
        Index for the computation (along the time axis).
    window_width_indices : int
        Window size for the line-length computation.
    f_sample : int
        Sampling frequency in Hz.

    Returns
    -------
    ll : float
        Line-length results with shape (...).
    """
    # Slice the last (time) axis with the rolling window
    time_slice = data[..., at_index - window_width_indices : at_index + 1]

    # Compute the absolute differences along the time axis
    diff = np.abs(np.diff(time_slice, axis=-1))

    # Sum the differences along the time axis, leaving other dimensions intact
    summed_diff = np.sum(diff, axis=-1)

    # Normalize by the window width and scale by sampling frequency
    ll = summed_diff / window_width_indices * (f_sample / 1000)

    return ll


def calculate_stimulation_response_curves(
    stimlist: pd.DataFrame,
    response_loader: MultipleHDFResponseLoader,  # for legacy Python, change back to if Python updated on Ubelix: HDFResponseLoader | MultipleHDFResponseLoader,
    selected_stim_channel_name_pos: str,
    selected_stim_channel_name_neg: str,
    selected_channel_paths: list[str],
    selected_intensities: list[float],
    protocol: str = "CR",
    no_traces: bool = False,
    exclude_responses: bool = False,
):
    #### FIXME calculates avg LL with average first
    assert len(selected_channel_paths) > 0

    ll_values = []
    traces = []
    id_matrix = []
    io_stimlist = stimlist[
        (stimlist["name_pos"] == selected_stim_channel_name_pos)
        & (stimlist["name_neg"] == selected_stim_channel_name_neg)
    ]

    n_replications_list = []

    io_stimlist = io_stimlist[io_stimlist["type"] == f"{protocol}_IO"]

    # Intensities
    ordinary_intensities = (
        selected_intensities[1:]
        if selected_intensities[0] == 0
        else selected_intensities
    )

    for io_intensity in ordinary_intensities:
        # print(f"... working on intensity {io_intensity} mA")
        filtered_io_stimlist = io_stimlist[io_stimlist["Int_prob"] == io_intensity]

        stim_indices = filtered_io_stimlist.index.tolist()
        assert len(stim_indices) > 0, (
            f"No stimulations found for intensity {io_intensity} mA. "
            "Check if there are suitable stimulations in the given stimlist."
        )
        res, stim_ids = response_loader.get_responses(
            stim_indices=stim_indices,
            response_channel_paths=selected_channel_paths,
            overwrite_excluded_recordings=exclude_responses,
            t_start=-1,
            t_stop=1,
            return_stim_ids=True,
        )
        if np.isnan(np.min(res)):
            print("Warning: NaN values in traces for intensity", io_intensity)

        n_replications_list.append(res.shape[0])

        if not no_traces:
            traces.append(res)
        else:
            traces.append([])

        id_matrix.append(np.tile(stim_ids[:, np.newaxis], res.shape[1]))

        # calculate ll on single trial basis
        single_ll = calculate_pointwise_line_length_max(
            data=res, offset_stim_seconds=1, f_sample=response_loader.f_sample
        )
        ll_values.append(single_ll)

    max_replications = np.max(n_replications_list)

    # 0mA componen
    if selected_intensities[0] == 0:
        # pick max_replication stimulations
        subset_io_stimlist = io_stimlist.sample(
            n=max_replications, replace=False, random_state=42
        )  # random state for reproducibility

        # get traces from -1.6 up to 0.6
        stim_indices = subset_io_stimlist.index.tolist()
        assert len(stim_indices) > 0, "No stimulations found for intensity 0 mA. "
        zero_res, zero_stim_ids = response_loader.get_responses(
            stim_indices=stim_indices,
            response_channel_paths=selected_channel_paths,
            overwrite_excluded_recordings=exclude_responses,
            t_start=-1 - 0.6,  # -0.6 w.r.t. stimulation time
            t_stop=1 - 0.6,
            return_stim_ids=True,
        )
        zero_stim_ids = zero_stim_ids + "-0.6s"
        if np.isnan(np.min(zero_res)):
            print("Warning: NaN values in traces for intensity 0mA")

        if not no_traces:
            traces.insert(0, zero_res)
        else:
            traces.insert(0, [])

        id_matrix.insert(0, np.tile(zero_stim_ids[:, np.newaxis], zero_res.shape[1]))

        # calculate ll on single trial basis
        single_ll = calculate_pointwise_line_length_max(
            data=zero_res, offset_stim_seconds=1, f_sample=response_loader.f_sample
        )
        ll_values.insert(0, single_ll)

    consistent_ll_values = []
    consistent_traces = []
    consistent_id_matrix = []
    for ll_values_element, traces_element, id_matrix_element in zip(
        ll_values, traces, id_matrix
    ):
        missing_stimulations = max_replications - id_matrix_element.shape[0]
        if missing_stimulations > 0:
            print("Warning: Missing: ", missing_stimulations)

        if not no_traces:
            empty_traces = np.full(
                (
                    missing_stimulations,
                    traces_element.shape[1],
                    traces_element.shape[2],
                ),
                np.nan,
            )
            consistent_trace = np.concatenate([traces_element, empty_traces], axis=0)
            consistent_traces.append(consistent_trace)

        empty_ll = np.full((missing_stimulations, traces_element.shape[1]), np.nan)
        consistent_ll = np.concatenate([ll_values_element, empty_ll], axis=0)
        consistent_ll_values.append(consistent_ll)
        empty_ids = np.full((missing_stimulations, id_matrix_element.shape[1]), "")
        consistent_ids = np.concatenate([id_matrix_element, empty_ids])
        consistent_id_matrix.append(consistent_ids)

    consistent_ll_values = np.array(consistent_ll_values)
    consistent_id_matrix = np.array(consistent_id_matrix)
    consistent_traces = np.array(consistent_traces) if not no_traces else None

    return (
        consistent_ll_values,  # shape: (n_intensities, n_replications, n_response_channel_paths)
        consistent_traces,  # shape: (n_intensities, n_replications, n_response_channel_paths, n_times)
        consistent_id_matrix,  # shape: (n_intensities, n_replications, n_response_channel_paths)
    )


def calculate_stimulation_response_curves_from_ids(
    id_matrix: np.ndarray,  # assumes 0mA component in row 0
    mrl: MultipleHDFResponseLoader,  # for legacy Python, change back to if Python updated on Ubelix: HDFResponseLoader | MultipleHDFResponseLoader,
    response_channel_paths: list[str],
    exclude_responses: bool = False,
):
    id_matrix = np.array(id_matrix)
    # ignore 0mA
    ind_matrix = mrl.get_inds_from_stim_ids(id_matrix[1:])

    traces = []
    for row in ind_matrix:
        data = mrl.get_responses(
            stim_indices=row,
            response_channel_paths=response_channel_paths,
            t_start=-1,
            t_stop=1,
            overwrite_excluded_recordings=exclude_responses,
        ).squeeze(
            1
        )  # shape: (n_rep, n_time)
        traces.append(data)
    traces = np.array(traces)

    zero_trace = np.zeros_like(traces[0])[np.newaxis, ...]
    traces = np.concatenate((zero_trace, traces), axis=0)

    ll_values = calculate_pointwise_line_length_max(
        traces, offset_stim_seconds=1, f_sample=mrl.f_sample
    )
    ll_med_values = np.nanmedian(ll_values, axis=1)

    return (
        ll_values,  # shape: (n_intensities, n_replications, n_response_channel_paths)
        ll_med_values,  # shape: (n_intensities, n_response_channel_paths)
    )


def calculate_upper_bounds_using_surrogates_auc(
    response_loader: MultipleHDFResponseLoader,
    response_channel_paths: list[str],
    n_surrogates: int = 100,
    intensities: int = list[float],
    n_replications: int = 12,
    fig: plt.Figure = None,
    significance_level: float = 0.95,
    sleep_score_restriction: list[SleepStage] = None,
    cleaning_factor: float = -1,
    out_file_name: str = None,
    use_cache: bool = False,
    ll_first: bool = True,
):
    """
    Calculates lower bound for significance of responses.
    Uses stimulation-free periods before CR_triplet stimulations.
    Returns: lower_bounds with shape (n_response_channel_paths)
    """
    intensities = np.array(intensities)
    n_intensities = len(intensities)

    ax_overview = None
    if fig is not None:
        n_cols = 6
        n_plots = len(response_channel_paths)  # ll_matrix.shape[2]
        n_rows = math.ceil(n_plots / n_cols) + 1
        gs = GridSpec(n_rows, n_cols, figure=fig)

        ax_overview = fig.add_subplot(gs[0, :])

    recording_indices, times, durations = find_stimulation_free_periods(
        response_loader=response_loader,
        minimal_duration_sec=60,
        ax=ax_overview,
        sleep_score_restriction=sleep_score_restriction,
    )
    ll_matrix = []
    if not use_cache:
        print(
            f"Calculating surrogate LL matrix with {n_surrogates} surrogates, {mp.cpu_count()} CPUs"
        )
        pool = mp.Pool(mp.cpu_count())  # or specify a smaller number
        # Build argument list
        tasks = [
            (
                i,
                {
                    "paths_h5": response_loader.paths_h5,
                    "paths_logs": response_loader.paths_logs,
                    "recording_names": response_loader.recording_names,
                    "path_lookup": response_loader.path_lookup,
                    "path_excluded_responses": response_loader.path_excluded_responses,
                },
                recording_indices,
                times,
                durations,
                response_channel_paths,
                n_replications,
                n_intensities,
                ll_first,
            )
            for i in range(n_surrogates)
        ]
        try:
            results = pool.map(_upper_bounds_var_worker, tasks, chunksize=1)
        except KeyboardInterrupt:
            print("Ctrl+C received; terminating pool.")
            pool.terminate()
            raise
        finally:
            pool.close()
            pool.join()
        # results is a list of n_surrogates arrays of shape (n_intensities, n_replications, n_channels)
        # Stack along the first axis => shape (n_surrogates, n_intensities, n_replications, n_channels)
        # if avg_first: shape (n_surrogates, n_intensities, n_channels)
        ll_matrix = np.stack(results, axis=0)

        if out_file_name is not None:
            np.save(out_file_name, ll_matrix)
    else:
        print(f"Try to use cached file {out_file_name}.")
        ll_matrix = np.load(out_file_name, allow_pickle=True)

    if ll_first:
        if cleaning_factor > 0:
            median_values = np.median(
                ll_matrix, axis=(0, 1, 2)
            )  # shape: (n_channel_paths,)
            threshold = cleaning_factor * median_values  # shape: (n_channel_paths,)

            mask = (
                ll_matrix > threshold[None, None, None, :]
            )  # shape broadcasted to (n_surrogates, n_intensities, n_replications, n_channel_paths)

            ll_matrix[mask] = np.nan
            print(f"Cleaning factor applied. {np.sum(mask)} traces ignored.")

        reduced_ll_matrix = np.nanmedian(
            ll_matrix, axis=2
        )  # shape: (n_surrogates, n_intensities, n_channel_paths)

        ll_values_minimum_dp = np.nanmin(
            ll_matrix, axis=(0, 1, 2)
        )  # shape: (n_channel_paths)
        ll_values_percentiles_dp = np.nanpercentile(
            ll_matrix, q=[5, 50, 95], axis=(0, 1, 2)
        )  # based on datapoints!
    else:
        if cleaning_factor > 0:
            print("Warning: cleaning does not work with avg first.")
        reduced_ll_matrix = ll_matrix

        ll_values_minimum_dp = None
        ll_values_percentiles_dp = [None, None, None]

    ll_values_minimum = np.nanmin(
        reduced_ll_matrix, axis=(0, 1)
    )  # shape: (n_channel_paths)
    ll_values_percentiles = np.nanpercentile(
        reduced_ll_matrix, q=[5, 50, 95], axis=(0, 1)
    )

    intensities_reshaped = intensities[None, :, None]  # shape (1, n_intensities, 1)

    auc_matrix = np.trapezoid(
        y=reduced_ll_matrix, x=intensities_reshaped, axis=1
    )  # shape (n_surrogates, n_channels)
    upper_bounds = np.nanpercentile(auc_matrix, significance_level * 100, axis=0)
    upper_bounds_95 = np.nanpercentile(auc_matrix, 0.95 * 100, axis=0)
    upper_bounds_99 = np.nanpercentile(auc_matrix, 0.99 * 100, axis=0)

    if fig is not None:
        for i, channel in enumerate(response_channel_paths):
            all_data = auc_matrix[:, i]
            all_data = all_data[~np.isnan(all_data)]
            bins = np.histogram_bin_edges(all_data, bins=40)
            row, col = divmod(i, n_cols)  # Determine row and column
            ax = fig.add_subplot(gs[row + 1, col])

            ax.set_ylabel("Frequency")
            ax.set_xlabel("AUC [a.u.]")
            ax.set_title(f"{channel.split('/')[-1]}")
            ax.axvline(upper_bounds[i], label=significance_level, color="red")
            ax.axvline(upper_bounds_95[i], label="0.95", color="orange")
            ax.axvline(upper_bounds_99[i], label="0.99", color="green")
            ax.hist(auc_matrix[:, i], bins=bins)

            if i == 0:
                ax.legend()

    return {
        "auc_matrix": auc_matrix,
        "upper_bounds": {
            "significance_level": upper_bounds,
            "upper_bounds_95": upper_bounds_95,
            "upper_bounds_99": upper_bounds_99,
        },
        "percentiles_dp": {
            "min": ll_values_minimum_dp,
            "5": ll_values_percentiles_dp[0],
            "med": ll_values_percentiles_dp[1],
            "95": ll_values_percentiles_dp[2],
        },
        "percentiles_agg": {
            "min": ll_values_minimum,
            "5": ll_values_percentiles[0],
            "med": ll_values_percentiles[1],
            "95": ll_values_percentiles[2],
        },
    }


def _upper_bounds_var_worker(args):
    (
        i,  # index of this surrogate
        response_loader_args,
        recording_indices,
        times,
        durations,
        response_channel_paths,
        n_replications,
        n_intensities,
        ll_first,
    ) = args
    mrl = MultipleHDFResponseLoader(**response_loader_args)

    # e.g., use i plus current time or PID for seed
    seed = int(tm.time()) + os.getpid() + i
    np.random.seed(seed)

    ll_intensities = []
    for j in range(n_intensities):
        # pick random stimulation free period
        random_idx = np.random.randint(len(recording_indices))

        recording_index = recording_indices[random_idx]
        time = times[random_idx]
        duration = durations[random_idx]
        random_offsets = np.random.uniform(0, duration, size=n_replications)
        random_times = random_offsets + time
        # TODO check annotations/time_grades

        data = mrl.get_traces(
            recording_indices=np.full(n_replications, recording_index),
            times_sec=random_times,
            response_channel_paths=response_channel_paths,
            t_start=-1,
            t_stop=1,
        )  # shape: (n_replications, n_response_channel_paths, n_times)

        if ll_first:
            single_ll = calculate_pointwise_line_length_max(
                data=data,
                offset_stim_seconds=1,
                f_sample=mrl.f_sample,
            )
            ll_intensities.append(single_ll)
        else:
            avg_data = np.nanmean(
                data, axis=0
            )  # shape: (n_response_channel_paths, n_times)
            single_ll = calculate_pointwise_line_length_max(
                data=avg_data,
                offset_stim_seconds=1,
                f_sample=mrl.f_sample,
            )  # (n_channel_paths)
            ll_intensities.append(single_ll)
    return ll_intensities


def calculate_AUC(
    ll_values: np.ndarray,
    intensities: np.ndarray,
):
    """
    Calculate the area under the curve (AUC) for each channel in the line length matrix.

    Parameters
    ----------
    ll_matrix : np.ndarray
        Line length matrix with shape (n_intensities, n_replications, n_channel_paths).
    intensities : np.ndarray
        Intensities corresponding to the line length matrix.

    Returns
    -------
    aucs : np.ndarray
        AUC values for each channel with shape (n_channel_paths).
    """
    ll_med_values = np.nanmedian(
        ll_values, axis=1
    )  # shape: (n_intensities, n_channel_paths)
    # Reshape intensities to match the dimensions of ll_matrix
    intensities_reshaped = np.array(intensities)[:, None]  # shape (n_intensities, 1)

    # Calculate AUC using trapezoidal rule
    aucs = np.trapezoid(y=ll_med_values, x=intensities_reshaped, axis=0)

    return aucs


def get_sleep_score_matrix(id_matrix: np.ndarray, logs: pd.DataFrame):
    stim_dict = dict(zip(logs["stim_id"], logs["sleep_score"]))

    sleep_score_matrix = np.vectorize(stim_dict.get)(id_matrix)
    return sleep_score_matrix


def calculate_excitability_index():
    pass  # TODO


def fit_curve(
    curve_function: callable,
    x: np.ndarray,
    y: np.ndarray,
    initial_values: list,
    bounds: tuple = None,
    loss: str = None,
    full_output: bool = False,
    max_iterations: int = 5000,
):
    options = {
        "p0": initial_values,
        "maxfev": max_iterations,
        "full_output": full_output,
    }
    if bounds is not None:
        options["bounds"] = bounds
    if loss is not None:
        options["loss"] = loss
        options["method"] = "trf"
    # return params

    if full_output:
        params, covariance, infodict, mesg, ier = scipy.optimize.curve_fit(
            curve_function, x, y, **options
        )

        return params, infodict["nfev"]
    else:
        params, covariance = scipy.optimize.curve_fit(curve_function, x, y, **options)

        return params


def bootstrap_curve_fitting(
    curve_function: callable,
    x: np.ndarray,
    y: np.ndarray,
    initial_values: list,
    bounds: tuple = None,
    n_bootstrap: int = 100,
    max_optimizer_iterations: int = 100,
    loss: str = "linear",
    parallelize: bool = False,
    normalize: bool = False,
    shared_normalization_min: np.ndarray = None,
    shared_normalization_max: np.ndarray = None,
):
    rng_seed = np.random.randint(0, 1_000_000)

    if parallelize:
        args_list = [
            (
                i,
                curve_function,
                x,
                y,
                initial_values,
                bounds,
                max_optimizer_iterations,
                loss,
                rng_seed,
                normalize,
                shared_normalization_min,
                shared_normalization_max,
            )
            for i in range(n_bootstrap)
        ]

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(_bootstrap_curve_fitting_worker, args_list)
        params_list = [res for res in results if res is not None]
    else:
        params_list = []
        for i in range(n_bootstrap):
            args = (
                i,
                curve_function,
                x,
                y,
                initial_values,
                bounds,
                max_optimizer_iterations,
                loss,
                rng_seed,
                normalize,
                shared_normalization_min,
                shared_normalization_max,
            )
            res = _bootstrap_curve_fitting_worker(args)
            if res is not None:
                params_list.append(res)

    return params_list


def _bootstrap_curve_fitting_worker(args):
    (
        i,
        curve_function,
        x,
        y,
        initial_values,
        bounds,
        max_optimizer_iterations,
        loss,
        rng_seed,
        normalize,
        shared_normalization_min,
        shared_normalization_max,
    ) = args

    np.random.seed(rng_seed + i)
    num_samples_per_intensity = y.shape[1]
    y_boot_list = []

    for intensity_index in range(x.shape[0]):
        valid_indices = np.where(np.isfinite(y[intensity_index]))[
            0
        ]  # remove -inf, +inf and NaN

        if len(valid_indices) == 0:
            print("Warning: No non-nan values for intensity index", intensity_index)
            y_boot_list.append(np.full(num_samples_per_intensity, 0))
            continue

        selected_indices = np.random.choice(
            valid_indices, size=num_samples_per_intensity, replace=True
        )

        y_boot_list.append(y[intensity_index, selected_indices])
    y_boot = np.array(y_boot_list)  # shape: (n_intensities, n_replications)
    y_boot = np.nanmedian(y_boot, axis=1)  # shape: (n_intensities,)

    if normalize:
        y_boot = normalize_ll_values(
            ll_values=y_boot,
            min=shared_normalization_min,
            max=shared_normalization_max,
            axis=0,
        )

    try:
        params = fit_curve(
            curve_function=curve_function,
            x=x,
            y=y_boot,
            bounds=bounds,
            initial_values=initial_values,
            loss=loss,
            max_iterations=max_optimizer_iterations,
        )
        return params
    except RuntimeError as e:
        # print(f"Warning in bootstrap iteration {i}: {e}")
        return None


def subset_bootstrap_curve_fitting(
    curve_function: callable,
    x: np.ndarray,  # shap
    y: np.ndarray,
    n_subset_replications: int,
    initial_values: list,
    bounds: tuple = None,
    n_bootstrap: int = 100,
    max_optimizer_iterations: int = 100,
    loss: str = "linear",
    parallelize: bool = False,
    normalize: bool = False,
    shared_normalization_min: float = None,
    shared_normalization_max: float = None,
):
    rng_seed = np.random.randint(0, 1_000_000)

    if parallelize:
        args_list = [
            (
                i,
                curve_function,
                x,
                y,
                initial_values,
                bounds,
                max_optimizer_iterations,
                loss,
                rng_seed,
                n_subset_replications,
                normalize,
                shared_normalization_min,
                shared_normalization_max,
            )
            for i in range(n_bootstrap)
        ]

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(_subset_bootstrap_curve_fitting_worker, args_list)
        params_list = [res for res in results if res is not None]
    else:
        params_list = []
        for i in range(n_bootstrap):
            args = (
                i,
                curve_function,
                x,
                y,
                initial_values,
                bounds,
                max_optimizer_iterations,
                loss,
                rng_seed,
                n_subset_replications,
                normalize,
                shared_normalization_min,
                shared_normalization_max,
            )
            res = _subset_bootstrap_curve_fitting_worker(args)
            if res is not None:
                params_list.append(res)

    return params_list


def _subset_bootstrap_curve_fitting_worker(args):
    (
        i,
        curve_function,
        x,
        y,
        initial_values,
        bounds,
        max_optimizer_iterations,
        loss,
        rng_seed,
        n_subset_replications,
        normalize,
        shared_normalization_min,
        shared_normalization_max,
    ) = args

    np.random.seed(rng_seed + i)
    y_boot_list = []

    for intensity_index in range(x.shape[0]):
        valid_indices = np.where(np.isfinite(y[intensity_index]))[
            0
        ]  # remove -inf, +inf and NaN

        if len(valid_indices) == 0:
            print("Warning: No non-nan values for intensity index", intensity_index)
            y_boot_list.append(np.full(n_subset_replications, 0))
            continue

        selected_indices = np.random.choice(  # FIXME
            valid_indices, size=n_subset_replications, replace=True
        )  # replacement is False, we do "Subsampling" here

        y_boot_list.append(y[intensity_index, selected_indices])

    # Concatenate the subsets
    y_boot = np.array(y_boot_list)  # shape: (n_intensities, n_subset_replications)
    y_boot = np.nanmedian(y_boot, axis=1)  # Median across replications
    if normalize:
        y_boot = normalize_ll_values(
            ll_values=y_boot,
            min=shared_normalization_min,
            max=shared_normalization_max,
            axis=0,
        )
    try:
        params = fit_curve(
            curve_function=curve_function,
            x=x,
            y=y_boot,
            bounds=bounds,
            initial_values=initial_values,
            loss=loss,
            max_iterations=max_optimizer_iterations,
        )
        return params
    except RuntimeError as e:
        # print(f"Warning in bootstrap iteration {i}: {e}")
        return None


def evaluate_bootstrap_result(
    x_fit: np.ndarray,
    curve_function: callable,
    params_list: list[list],
    percentiles: list[float] = [2.5, 97.5],
    r_squared_threshold: float = -1,
    x_dp: np.ndarray = None,
    y_dp_true: np.ndarray = None,
):
    y_fits = []
    aucs = []

    excluded_r_squared = 0
    if r_squared_threshold > 0:
        y_dp_true_avg = np.mean(y_dp_true)

    for i, params in enumerate(params_list):
        if r_squared_threshold > 0:
            y_dp_pred = curve_function(x_dp, *params)

            ss_res = np.sum((y_dp_true - y_dp_pred) ** 2)

            ss_tot = np.sum((y_dp_true - y_dp_true_avg) ** 2)

            r_squared = 1 - ss_res / ss_tot

            if r_squared < r_squared_threshold:
                excluded_r_squared += 1
                continue

        y_fit = curve_function(x_fit, *params)

        auc = np.trapezoid(y=y_fit, x=x_fit)
        aucs.append(auc)
        y_fits.append(y_fit)

    # calculate empirical CI
    y_fits = np.array(y_fits)
    aucs = np.array(aucs)
    ci = np.nanpercentile(y_fits, percentiles, axis=0)
    auc_ci = np.nanpercentile(aucs, percentiles, axis=0)

    return {
        "percentiles": ci,
        "aucs": aucs,
        "auc_percentiles": auc_ci,
        "excluded_r_squared": excluded_r_squared,
    }


def normalize_ll_values(
    ll_values: np.ndarray,
    min: float = None,  # or np.ndarray,
    max: float = None,  # or np.ndarray,
    axis: tuple = (0, 1),
    use_min=False,
):
    # TODO adapt for more dimensions
    """
    Parameters
    ----------
    ll_values : np.ndarray
        Line length values with shape (n_replications, n_intensities, [n_channels]).
    surrogate_ll_percentile_5 : float | np.ndarray shape float or (n_channels,)
        5th percentile of surrogate LL values.
    """
    if min is None:
        if use_min:
            min = np.min(ll_values, axis=axis)
        else:
            min = np.nanpercentile(ll_values, q=5, axis=axis)

    if max is None:
        max = np.nanpercentile(
            ll_values, q=95, axis=axis
        )  # shape: float or (n_channels)

    normalized_ll_values = (ll_values - min) / (max - min)

    return normalized_ll_values


def calculate_misfit(  # deprecated
    curve_function: callable,
    x: np.ndarray,
    params_to_compare: list,
    params_list: list[list],
    ax: plt.Axes,
    add_legend: bool = True,
    bootstrap_percentiles: tuple[float] = (2.5, 97.5),
):
    x_fit = np.linspace(min(x), max(x), 1000)
    y_to_compare_fit = curve_function(x_fit, *params_to_compare)

    y_fits = []
    for i, params in enumerate(params_list):
        y_fit = curve_function(x_fit, *params)
        y_fits.append(y_fit)
    # calculate empirical CI
    y_fits = np.array(y_fits)
    ci = np.percentile(y_fits, bootstrap_percentiles, axis=0)

    misfit_mask = (y_to_compare_fit < ci[0]) | (y_to_compare_fit > ci[1])
    misfit = misfit_mask.mean()

    ax.plot(
        x_fit,
        # masked_where masks if True
        np.ma.masked_where(misfit_mask, y_to_compare_fit),
        color="green",
        label="Inside CI",
    )
    ax.plot(
        x_fit,
        np.ma.masked_where(~misfit_mask, y_to_compare_fit),
        color="red",
        label="Outside CI",
    )
    ax.fill_between(
        x_fit,
        ci[0],
        ci[1],
        color="blue",
        alpha=0.2,
        label=f"{round(bootstrap_percentiles[1] - bootstrap_percentiles[0])}% Bootstrap CI",
    )

    if add_legend:
        ax.legend()
    ax.set_ylim([0, 1.1])
    ax.set_xlabel("Normalized intensities [a.u.]")
    ax.set_ylabel("Normalized LL [a.u.]")

    return misfit


def calculate_auc_misfit(
    curve_function: callable,
    x: np.ndarray,
    params_to_compare: list,
    params_list: list[list],
    ax: plt.Axes = None,
    add_legend: bool = True,
    bootstrap_percentiles: tuple[float, float] = (2.5, 97.5),
):
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = curve_function(x_fit, *params_to_compare)
    area_to_compare = np.trapezoid(y=y_fit, x=x_fit)

    areas = []
    for i, params in enumerate(params_list):
        y_fit = curve_function(x_fit, *params)
        area = np.trapezoid(y=y_fit, x=x_fit)
        areas.append(area)
    # calculate empirical CI
    areas = np.array(areas)
    ci = np.percentile(areas, bootstrap_percentiles, axis=0)

    median = np.median(areas)

    binary_misfit = (area_to_compare < ci[0]) | (area_to_compare > ci[1])
    delta_misfit = np.abs(area_to_compare - median)

    if ax is not None:
        ax.hist(areas, bins=20, alpha=0.5, label="Bootstrap AUCs")
        ax.axvline(area_to_compare, color="black", label="Subset AUC")
        ax.axvline(median, color="red", label="Median")

        if add_legend:
            ax.legend()
        ax.set_xlabel("AUC [a.u.]")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{delta_misfit:.3f}")

        if binary_misfit:
            ax.set_facecolor("#FFADB0")
        else:
            ax.set_facecolor("#90EE90")

    return binary_misfit, delta_misfit, area_to_compare, median


def find_stimulation_free_periods(
    response_loader: MultipleHDFResponseLoader,
    minimal_duration_sec: int = 60,
    sleep_score_restriction: list[SleepStage] = [],
    ax: plt.Axes = None,
):
    logs = response_loader.get_logs()

    before_gap = logs[logs["time_diff"] > minimal_duration_sec]

    recording_indices = before_gap["temp_recording_index"].to_numpy()
    durations = (
        before_gap["time_diff"].to_numpy() - 20
    )  # at least 20 seconds after stim before
    times = before_gap["TTL_DS"].to_numpy() / response_loader.f_sample - durations
    durations -= 20  # at least 20 seconds before next stim

    # Ensure times is non-negative
    # If negative, subtract the amount from the duration and set time to 0
    negative_mask = times < 0
    durations[negative_mask] += times[
        negative_mask
    ]  # Subtract negative part from duration
    times[negative_mask] = 0  # Set negative times to zero

    # remove durations < minimal duration
    negative_mask = durations < minimal_duration_sec
    recording_indices = recording_indices[~negative_mask]
    times = times[~negative_mask]
    durations = durations[~negative_mask]

    if sleep_score_restriction is not None and len(sleep_score_restriction) > 0:
        sleep_scores = response_loader.get_all_sleep_scores()
        recording_indices, times, durations = _restrict_sleep_stages_in_gaps(
            recording_indices=recording_indices,
            times=times,
            durations=durations,
            sleep_scores=sleep_scores,
            sleep_score_restriction=sleep_score_restriction,
            f_sample=response_loader.f_sample,
            minimal_duration_sec=minimal_duration_sec,
        )

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if ax is not None:
        rec_lengths = response_loader.get_recording_lengths()
        offsets = np.cumsum(rec_lengths)
        offsets = np.insert(offsets, 0, 0)  # add 0 at beginning

        if sleep_score_restriction is not None and len(sleep_score_restriction) > 0:
            sleep_scores = response_loader.get_all_sleep_scores()
            time_grades = response_loader.get_all_time_grades()

        for i, rec_length in enumerate(rec_lengths):
            color = color_cycle[i % len(color_cycle)]  # Cycle through colors

            sec_to_h = 60 * 60
            ind_to_h = response_loader.f_sample * sec_to_h

            ax.axvspan(
                offsets[i] / ind_to_h,
                (offsets[i] + rec_length) / ind_to_h,
                label=f"Recording {i}",
                color=color,
            )

            mask = recording_indices == i

            if sleep_score_restriction is not None and len(sleep_score_restriction) > 0:
                x = np.arange(len(sleep_scores[i]))
                ax.plot(
                    (offsets[i] + x) / ind_to_h,
                    sleep_scores[i],
                    color="black",
                    linewidth=0.5,
                )
                ax.plot(
                    (offsets[i] + x) / ind_to_h,
                    -time_grades[i],
                    color="black",
                    linewidth=0.5,
                )
                # Get sleep stage ticks
                sleep_ticks = [s.value for s in SleepStage]
                sleep_labels = [SleepStage(y).name for y in sleep_ticks]

                # Get time grade ticks (negated to match the reversed values in the plot)
                time_ticks = [-t.value for t in TimeGrade][
                    1:
                ]  # otherwise we have 0 twice
                time_labels = [TimeGrade(t.value).name for t in TimeGrade][1:]

                # Combine both into yticks and ylabels
                yticks = sleep_ticks + time_ticks  # Merge both lists
                ylabels = sleep_labels + time_labels  # Merge corresponding labels

                # Set the combined y-axis ticks
                ax.set_yticks(yticks)
                ax.set_yticklabels(ylabels)

            ax.hlines(
                y=np.zeros_like(times[mask]),
                xmin=(offsets[i] / ind_to_h) + (times[mask] / sec_to_h),
                xmax=(offsets[i] / ind_to_h)
                + ((times[mask] + durations[mask]) / sec_to_h),
                linewidth=4,
                color="white",
            )

        # ax.legend()
        ax.set_xlabel("Time [h]")
        ax.set_title("Recording overview")

    return recording_indices, times, durations


def _restrict_sleep_stages_in_gaps(
    recording_indices: np.ndarray,
    times: np.ndarray,
    durations: np.ndarray,
    sleep_scores: list[np.ndarray],
    sleep_score_restriction: list[SleepStage],
    f_sample: int,
    minimal_duration_sec: int,
):
    new_recs, new_starts, new_durs = [], [], []

    for i, time in enumerate(times):
        rec_idx = recording_indices[i]
        gap_start_time = time
        gap_end_time = time + durations[i]

        # Convert to integer indices in sleep_stages array
        start_idx = round(gap_start_time * f_sample)
        end_idx = round(gap_end_time * f_sample)

        # Bound checks (if end_idx goes beyond the length of sleep_stages)
        max_index = len(sleep_scores[rec_idx])
        if start_idx < 0:
            start_idx = 0
        if end_idx > max_index:
            end_idx = max_index

        assert start_idx < end_idx

        # Create a mask for allowed stages within [start_idx, end_idx)
        stage_segment = sleep_scores[rec_idx][start_idx:end_idx]  # slice of the array
        sleep_score_restriction_values = [s.value for s in sleep_score_restriction]
        mask = np.isin(stage_segment, sleep_score_restriction_values)  # bool array

        # Get contiguous True segments
        segments = _contiguous_true_segments(mask)
        for seg_start, seg_end in segments:
            # Convert segment indices back to time (in seconds)
            # seg_start, seg_end are relative to 'start_idx'
            absolute_seg_start = start_idx + seg_start
            absolute_seg_end = start_idx + seg_end

            # Convert back to seconds
            segment_start_time = absolute_seg_start / f_sample
            segment_end_time = absolute_seg_end / f_sample

            # Clip to original gap if needed
            # (Not strictly necessary, but to be safe)
            if segment_start_time < gap_start_time:
                segment_start_time = gap_start_time
            if segment_end_time > gap_end_time:
                segment_end_time = gap_end_time

            segment_duration = segment_end_time - segment_start_time
            if segment_duration <= minimal_duration_sec:
                continue

            # Store the new smaller gap
            new_recs.append(rec_idx)
            new_starts.append(segment_start_time)
            new_durs.append(segment_duration)

    return np.array(new_recs), np.array(new_starts), np.array(new_durs)


def _contiguous_true_segments(bool_array):
    """
    Returns a list of (start, end) index pairs for each
    contiguous True segment in bool_array.
    """
    segments = []
    in_segment = False
    seg_start = None

    for i, val in enumerate(bool_array):
        if val and not in_segment:
            # Beginning of a True segment
            in_segment = True
            seg_start = i
        elif not val and in_segment:
            # End of a True segment
            in_segment = False
            segments.append((seg_start, i))
            seg_start = None

    # If the last segment goes till the end
    if in_segment:
        segments.append((seg_start, len(bool_array)))

    return segments


#  return param
#    pass


def filter_logs(
    complete_logs: pd.DataFrame,
    selected_stim_channel_name_pos: str,
    selected_stim_channel_name_neg: str,
    n_replications: int = -1,
    sleep_stages: list[SleepStage] = [SleepStage.AWAKE, SleepStage.QWAKE],
    stim_protocol: str = "CR_IO",
    triplet_protocol: str = "CR_triplet",
):
    # Filter logs and apply filtering
    if sleep_stages is not None and len(sleep_stages) > 0:
        sleep_states = [s.name for s in [SleepStage.AWAKE, SleepStage.QWAKE]]
        protocol_logs = complete_logs[
            (complete_logs["sleep_score"].isin(sleep_states))
            & (complete_logs["name_pos"] == selected_stim_channel_name_pos)
            & (complete_logs["name_neg"] == selected_stim_channel_name_neg)
            & (complete_logs["type"] == f"{stim_protocol}")
        ]
    else:
        protocol_logs = complete_logs[
            (complete_logs["name_pos"] == selected_stim_channel_name_pos)
            & (complete_logs["name_neg"] == selected_stim_channel_name_neg)
            & (complete_logs["type"] == f"{stim_protocol}")
        ]

    # remove noise
    protocol_logs = protocol_logs[protocol_logs["noise"] == 0]

    if n_replications == -1:
        filtered_protocol_logs = protocol_logs
    elif stim_protocol == "CR_PP" or stim_protocol == "PP_PP":
        filtered_protocol_logs = protocol_logs.groupby(
            [
                "Int_prob",
                "Int_cond",
                "IPI_ms",
                "name_pos",
                "name_neg",
            ]
        ).head(n_replications)
    else:
        # Count the size of each group
        group_sizes = protocol_logs.groupby(["Int_prob", "name_pos", "name_neg"]).size()

        # Identify groups with less than n_replications
        too_small_groups = group_sizes[group_sizes < n_replications]
        if not too_small_groups.empty:
            print(
                f"Warning: Some groups have less than {n_replications} replications: {too_small_groups}"
            )

        filtered_protocol_logs = protocol_logs.groupby(
            [
                "Int_prob",
                "name_pos",
                "name_neg",
            ]
        ).head(n_replications)

    # Combine both datasets and sort by original index to preserve order
    if triplet_protocol is not None:
        if n_replications:
            triplet_logs = complete_logs[
                complete_logs["type"] == f"{triplet_protocol}"
            ].head(n_replications)
        else:
            triplet_logs = complete_logs[complete_logs["type"] == f"{triplet_protocol}"]
        filtered_logs = pd.concat([filtered_protocol_logs, triplet_logs]).sort_index()
    else:
        filtered_logs = filtered_protocol_logs.sort_index()

    if len(filtered_logs) == 0:
        print("Warning: No logs found for the given filter criteria.")
    return filtered_logs


def find_max_n_replications(
    complete_logs: pd.DataFrame,
    selected_stim_channel_name_pos: str,
    selected_stim_channel_name_neg: str,
    stim_protocol: str = "CR_IO",
    sleep_states: list[SleepStage] = [SleepStage.AWAKE, SleepStage.QWAKE],
):
    if sleep_states is not None and len(sleep_states) > 0:
        sleep_states = [s.name for s in sleep_states]
        protocol_logs = complete_logs[
            (complete_logs["sleep_score"].isin(sleep_states))
            & (complete_logs["name_pos"] == selected_stim_channel_name_pos)
            & (complete_logs["name_neg"] == selected_stim_channel_name_neg)
            & (complete_logs["type"] == f"{stim_protocol}")
        ]
    else:
        protocol_logs = complete_logs[
            (complete_logs["type"] == f"{stim_protocol}")
            & (complete_logs["name_pos"] == selected_stim_channel_name_pos)
            & (complete_logs["name_neg"] == selected_stim_channel_name_neg)
        ]

    if stim_protocol == "CR_IO":
        group_by = ["Int_prob", "name_pos", "name_neg"]
    else:
        print("Warning: Unknown stimulation protocol. Using default grouping.")
        group_by = ["name_pos", "name_neg"]

    group_sizes = protocol_logs.groupby(group_by).size()
    max_n_replications = group_sizes.min()
    return group_sizes, max_n_replications


def calculate_model_performance(y, y_pred, num_params):
    total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)

    residuals = y - y_pred

    sum_squared_residuals = np.sum(residuals**2)
    r_squared = 1 - (sum_squared_residuals / total_sum_of_squares)

    n_dp = len(y)

    # source: https://en.wikipedia.org/wiki/Akaike_information_criterion#Comparison_with_least_squares
    delta_aic = 2 * num_params + n_dp * np.log(sum_squared_residuals / n_dp)

    mae = np.mean(np.abs(y - y_pred))

    mape = 100 * np.mean(np.abs((y - y_pred) / y))

    smape = 100 * np.mean((np.abs(y_pred - y) / ((np.abs(y) + np.abs(y_pred)) * 0.5)))

    return {
        "r_squared": r_squared,
        "MAE": mae,
        "MAPE": mape,
        "sMAPE": smape,
        "dAIC": delta_aic,
    }


def max_consecutive_trues(arr: np.ndarray):
    """
    Calculates the maximum consecutive streak of true values in an array.
    Works with 1D and 2D arrays.
    1d: returns scalar
    2d: (n_1, n_2) input, output (n_2)
    """
    arr = np.asarray(arr).astype(int)

    if arr.ndim == 1:
        arr = arr[:, None]  # shape (N,) -> (N,1)
        squeeze_output = True
    elif arr.ndim == 2:
        squeeze_output = False
    else:
        raise ValueError("Only 1D or 2D arrays are supported")

    results = []
    for col in arr.T:
        padded = np.pad(col, (1, 1))
        diff = np.diff(padded)
        run_starts = np.where(diff == 1)[0]
        run_ends = np.where(diff == -1)[0]
        run_lengths = run_ends - run_starts
        results.append(np.max(run_lengths) if run_lengths.size > 0 else 0)

    result = np.array(results)
    return result[0] if squeeze_output else result


def calculate_woi_start_indices(
    data: np.ndarray, offset_stim_seconds: float, f_sample: int
):
    """
    Calculates the WOI start index using the max + continuous LLmethod.
    At offset_stim_seconds, we suppose the stimulation starts. We then calculate the LL from
    250ms up to 500ms with a 250ms window. We then take the arg max value of the LL in this window.

    Parameters
    ----------
    data : np.ndarray
        Input array with shape (..., n_time).
    offset_stim_seconds : float
        Time in seconds where the stimulation starts.
    f_sample : int
        Sampling frequency in Hz.

    Returns
    -------
    woi_start_indices : np.ndarray
        Line-length results with shape (...).
    """
    start_index_cll = round(offset_stim_seconds * f_sample)
    end_index_cll = round((offset_stim_seconds + 2) * f_sample)
    start_index_ll = round(0.25 * f_sample)
    end_index_ll = round(0.5 * f_sample)

    cont_ll = calculate_continuous_line_length(
        data=data,
        start_index=start_index_cll,
        end_index=end_index_cll,
        window_width_indices=round(0.25 * f_sample),
        f_sample=f_sample,
    )
    woi_start_indices = np.argmax(cont_ll[..., start_index_ll:end_index_ll], axis=-1)

    return woi_start_indices


def calculate_peak_latency_1(
    traces: np.ndarray, offset_stim_seconds: int, f_sample: int
):
    sos_bandpass = scipy.signal.butter(
        4, [45], fs=f_sample, btype="lowpass", output="sos"
    )
    traces = scipy.signal.sosfiltfilt(sos_bandpass, traces, axis=-1)
    # TODO is WOI necessary? YES

    # 1. baseline correct single trials
    baseline_period = [
        round((offset_stim_seconds - 0.5) * f_sample),
        int((offset_stim_seconds - 0.00) * f_sample),
    ]
    baseline_median = np.median(
        traces[..., baseline_period[0] : baseline_period[1]],
        axis=-1,
        keepdims=True,
    )  # shape (n_intensities, n_replications, [n_channels], 1)
    baseline_corrected_traces = traces - baseline_median

    # 2. average signal
    avg_traces = np.mean(
        baseline_corrected_traces, axis=1
    )  # shape (n_intensities, [n_channels], n_time)

    # 3. baseline correct mean signal
    baseline_avg_median = np.median(
        avg_traces[..., baseline_period[0] : baseline_period[1]],
        axis=-1,
        keepdims=True,
    )  # shape (n_intensities, [n_channels], 1)
    baseline_corrected_avg_traces = avg_traces - baseline_avg_median

    if baseline_corrected_avg_traces.ndim == 2:
        baseline_corrected_avg_traces = baseline_corrected_avg_traces[
            :, np.newaxis
        ]  # shape: (n_intensities, n_channels or 1)

    # 4. calculate baseline std
    baseline_std = np.std(
        baseline_corrected_avg_traces[..., baseline_period[0] : baseline_period[1]],
        axis=-1,
    )  # shape (n_intensities, n_channels or 1) # FIXME is this correct? from avg traces?

    # 5. set everything before stim to a constant value

    # TODO EvM used stim time + 10ms
    baseline_corrected_avg_traces[
        :, :, : int((offset_stim_seconds - 0.00) * f_sample)
    ] = (
        baseline_corrected_avg_traces[:, :, int((offset_stim_seconds - 0.0) * f_sample)]
    )[
        ..., np.newaxis
    ]

    n_intensities = baseline_corrected_avg_traces.shape[0]
    n_channels = baseline_corrected_avg_traces.shape[1]
    peaks_positive = [
        [
            scipy.signal.find_peaks(
                baseline_corrected_avg_traces[a, b, :],
                prominence=baseline_std[a, b],
            )[0]
            for b in range(n_channels)
        ]
        for a in range(n_intensities)
    ]
    peaks_positive = np.array(peaks_positive, dtype=object)

    peaks_negative = [
        [
            scipy.signal.find_peaks(
                -baseline_corrected_avg_traces[a, b, :],
                prominence=baseline_std[a, b],
            )[0]
            for b in range(n_channels)
        ]
        for a in range(n_intensities)
    ]
    peaks_negative = np.array(peaks_negative, dtype=object)

    peaks = np.vectorize(lambda p, n: np.concatenate([p, n]), otypes=[object])(
        peaks_positive,
        peaks_negative,
    )

    polarity_atlas_positive = np.vectorize(
        lambda p: np.full(len(p), 1, dtype=int), otypes=[object]
    )(peaks_positive)
    polarity_atlas_negative = np.vectorize(
        lambda n: np.full(len(n), -1, dtype=int), otypes=[object]
    )(peaks_negative)
    polarity_atlas = np.vectorize(lambda p, n: np.concatenate([p, n]), otypes=[object])(
        polarity_atlas_positive, polarity_atlas_negative
    )

    # function to reduce one array of peaks to the closest value
    target = round(offset_stim_seconds * f_sample)

    def closest_peak(arr, arr_to_pick):
        return arr_to_pick[np.abs(arr - target).argmin()] if arr.size > 0 else -1

    # apply across all cells
    closest_peaks = np.vectorize(closest_peak)(peaks, peaks)
    closest_polarities = np.vectorize(closest_peak)(peaks, polarity_atlas)

    peak_latencies = closest_peaks / f_sample - offset_stim_seconds
    if baseline_corrected_avg_traces.shape[1] == 1:
        baseline_corrected_avg_traces = np.squeeze(
            baseline_corrected_avg_traces, axis=1
        )
        peak_latencies = np.squeeze(peak_latencies, axis=1)

    return peak_latencies, baseline_corrected_avg_traces


def calculate_peak_latency(traces: np.ndarray, offset_stim_seconds: int, f_sample: int):
    # traces shape: (n_intensities, n_replications, n_channels, n_time)

    sos_bandpass = scipy.signal.butter(
        4, [45], fs=f_sample, btype="lowpass", output="sos"
    )
    traces = scipy.signal.sosfiltfilt(sos_bandpass, traces, axis=3)
    # TODO is WOI necessary? YES

    # 1. baseline correct single trials
    baseline_period = [
        round((offset_stim_seconds - 0.5) * f_sample),
        int((offset_stim_seconds - 0.00) * f_sample),
    ]
    baseline_median = np.median(
        traces[:, :, :, baseline_period[0] : baseline_period[1]], axis=3
    )[
        ..., np.newaxis
    ]  # shape (n_intensities, n_replications, n_channels, 1)
    baseline_corrected_traces = traces - baseline_median

    # 2. average signal
    avg_traces = np.mean(
        baseline_corrected_traces, axis=1
    )  # shape (n_intensities, n_channels, n_time)

    # 3. baseline correct mean signal
    baseline_avg_median = np.median(
        avg_traces[:, :, baseline_period[0] : baseline_period[1]], axis=2
    )[
        ..., np.newaxis
    ]  # shape (n_intensities, n_channels, 1)
    baseline_corrected_avg_traces = avg_traces - baseline_avg_median

    # 4. calculate baseline std
    baseline_std = np.std(
        baseline_corrected_avg_traces[:, :, baseline_period[0] : baseline_period[1]],
        axis=2,
    )  # shape (n_intensities, n_channels)

    # 5. set everything before stim to a constant value
    # TODO EvM used stim time + 10ms
    baseline_corrected_avg_traces[
        :, :, : int((offset_stim_seconds - 0.00) * f_sample)
    ] = (
        baseline_corrected_avg_traces[:, :, int((offset_stim_seconds - 0.0) * f_sample)]
    )[
        ..., np.newaxis
    ]

    n_intensities = baseline_corrected_avg_traces.shape[0]
    n_channels = baseline_corrected_avg_traces.shape[1]
    peaks_positive = [
        [
            scipy.signal.find_peaks(
                baseline_corrected_avg_traces[a, b, :],
                prominence=baseline_std[a, b],
            )[0]
            for b in range(n_channels)
        ]
        for a in range(n_intensities)
    ]
    peaks_positive = np.array(peaks_positive, dtype=object)

    peaks_negative = [
        [
            scipy.signal.find_peaks(
                -baseline_corrected_avg_traces[a, b, :],
                prominence=baseline_std[a, b],
            )[0]
            for b in range(n_channels)
        ]
        for a in range(n_intensities)
    ]
    peaks_negative = np.array(peaks_negative, dtype=object)

    peaks = np.vectorize(lambda p, n: np.concatenate([p, n]), otypes=[object])(
        peaks_positive,
        peaks_negative,
    )

    polarity_atlas_positive = np.vectorize(
        lambda p: np.full(len(p), 1, dtype=int), otypes=[object]
    )(peaks_positive)
    polarity_atlas_negative = np.vectorize(
        lambda n: np.full(len(n), -1, dtype=int), otypes=[object]
    )(peaks_negative)
    polarity_atlas = np.vectorize(lambda p, n: np.concatenate([p, n]), otypes=[object])(
        polarity_atlas_positive, polarity_atlas_negative
    )

    # function to reduce one array of peaks to the closest value
    target = round(offset_stim_seconds * f_sample)

    def closest_peak(arr, arr_to_pick):
        return arr_to_pick[np.abs(arr - target).argmin()] if arr.size > 0 else -1

    # apply across all cells
    closest_peaks = np.vectorize(closest_peak)(peaks, peaks)
    closest_polarities = np.vectorize(closest_peak)(peaks, polarity_atlas)

    peak_latencies = closest_peaks / f_sample - offset_stim_seconds
    print(baseline_corrected_avg_traces.shape)
    return peak_latencies, baseline_corrected_avg_traces

    # time = np.arange(baseline_corrected_avg_traces.shape[2])
    # for i in range(4):
    #     plt.plot(time, baseline_corrected_avg_traces[i + 10, 8])
    #     plt.scatter(
    #         time[closest_peaks[i + 10, 8]],
    #         baseline_corrected_avg_traces[i + 10, 8][closest_peaks[i + 10, 8]],
    #     )
    #     plt.title(closest_polarities[i + 10, 8])

    # woi_start_indices = calculate_woi_start_indices(

    # )

    # From EvM
    # BL_period = [int((t0 - 0.5) * Fs), int((t0 - 0.00) * Fs)]
    # bl_median = np.median(trials[:, BL_period[0]:BL_period[1]], axis=1)
    # trials = ff.lp_filter(trials, 45, Fs)
    # trials = trials - bl_median[:, None]

    # # 2. Average signal
    # mean_signal = np.mean(trials, axis=0)

    # # 3. Subtract BL median
    # mean_signal = mean_signal - np.median(mean_signal[BL_period[0]:BL_period[1]])

    # # 4. Calculate standard deviation of baseline period
    # std = np.std(mean_signal[BL_period[0]:BL_period[1]])
    # # 5. Threshold: +/- 3.4 std, find first peak crossing this threshold
    # get_peak_check = 1
    # factor = 1
    # mean_signal[:int((t0 + 0.010) * Fs)] = mean_signal[int((t0 + 0.010) * Fs)]
    # # mean_signal[int((t0 + WOI + 2 * w_LL / 3) * Fs)] = 0

    # first_peak = None

    # threshold = 1 * std
    # if polarity == 1:
    #     peaks, _ = find_peaks(mean_signal, prominence=threshold)
    # else:
    #     peaks, _ = find_peaks(-mean_signal, prominence=threshold)

    # if peaks.size > 0:
    #     # find closest peak to peak_lat_general
    #     peak_lat_general_datapoint = (t0 + peak_lat_general) * Fs
    #     first_peak = peaks[np.argmin(np.abs(peaks - peak_lat_general_datapoint))]
    #     peak_detected = 1
    # else:
    #     min_thr = np.max([peak_lat_general - 0.03, 0.005])
    #     mean_signal[:int((t0 + min_thr) * Fs)] = 0
    #     mean_signal[int((t0 + peak_lat_general + 0.03) * Fs):] = 0
    #     first_peak = np.argmax(polarity*mean_signal)
    #     peak_detected = 0

    # return first_peak / Fs - t0, peak_detected


def calcualte_stimulation_onsets():
    pass


def significant_exi_difference_testing(
    norm_ll_values_1,
    norm_ll_values_2,
    n_surrogates,
    intensities,
    curve,
    ax: plt.Axes = None,
    parallelize=False,
    max_iterations=1000,
):
    assert norm_ll_values_1.shape == norm_ll_values_2.shape
    n_replications = norm_ll_values_1.shape[1]
    n_intensities = norm_ll_values_2.shape[0]
    pooled_ll_values = np.concatenate([norm_ll_values_1, norm_ll_values_2], axis=1)

    delta_exis_null = []
    delta_empirical_exis_null = []

    x_fit = np.linspace(0, 1, 1000)
    if parallelize:
        rng_seed = np.random.randint(0, 1_000_000)
        args_list = [
            (
                i,
                norm_ll_values_1,
                norm_ll_values_2,
                intensities,
                curve,
                max_iterations,
                rng_seed,
            )
            for i in range(n_surrogates)
        ]

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(_significant_exi_difference_testing, args_list)
        delta_exis_null = [res[0] for res in results]
        delta_empirical_exis_null = [res[1] for res in results]
    else:

        for j in range(n_surrogates):
            values_1 = np.empty_like(norm_ll_values_1)
            values_2 = np.empty_like(norm_ll_values_1)
            for i in range(n_intensities):
                idx = np.random.permutation(2 * n_replications)
                values_1[i] = pooled_ll_values[i, idx[:n_replications]]
                values_2[i] = pooled_ll_values[i, idx[n_replications:]]

            med_values_1 = np.nanmedian(values_1, axis=1)
            med_values_2 = np.nanmedian(values_2, axis=1)

            empirical_exi_1 = np.trapezoid(med_values_1, intensities)
            empirical_exi_2 = np.trapezoid(med_values_2, intensities)
            try:
                params_1 = fit_curve(
                    curve_function=curve["function"],
                    x=intensities,
                    y=med_values_1,
                    initial_values=curve["initial_values"],
                    bounds=curve["bounds"],
                    max_iterations=max_iterations,
                )
                params_2 = fit_curve(
                    curve_function=curve["function"],
                    x=intensities,
                    y=med_values_2,
                    initial_values=curve["initial_values"],
                    bounds=curve["bounds"],
                    max_iterations=max_iterations,
                )

                y_fit_1 = curve["function"](x_fit, *params_1)
                y_fit_2 = curve["function"](x_fit, *params_2)

                exi_1 = np.trapezoid(y_fit_1, x_fit)
                exi_2 = np.trapezoid(y_fit_2, x_fit)
                # TODO add r_squared requirement

                delta_exis_null.append(np.abs(exi_1 - exi_2))
            except RuntimeError as e:
                pass

            delta_empirical_exis_null.append(np.abs(empirical_exi_1 - empirical_exi_2))

    delta_exis_null = np.array(delta_exis_null)
    delta_exis_null = delta_exis_null[~np.isnan(delta_exis_null)]
    delta_empirical_exis_null = np.array(delta_empirical_exis_null)

    # calculate actual delta exi
    empirical_actual_exi_1 = np.trapezoid(
        np.nanmedian(norm_ll_values_1, axis=1), intensities
    )
    empirical_actual_exi_2 = np.trapezoid(
        np.nanmedian(norm_ll_values_2, axis=1), intensities
    )

    actual_delta_empirical_exi = np.abs(empirical_actual_exi_1 - empirical_actual_exi_2)

    surrogate_p_value_empirical = (
        np.sum(delta_empirical_exis_null >= actual_delta_empirical_exi) + 1
    ) / (n_surrogates + 1)

    try:
        params_1 = fit_curve(
            curve_function=curve["function"],
            x=intensities,
            y=np.nanmedian(norm_ll_values_1, axis=1),
            initial_values=curve["initial_values"],
            bounds=curve["bounds"],
            max_iterations=max_iterations,
        )
        params_2 = fit_curve(
            curve_function=curve["function"],
            x=intensities,
            y=np.nanmedian(norm_ll_values_2, axis=1),
            initial_values=curve["initial_values"],
            bounds=curve["bounds"],
            max_iterations=max_iterations,
        )

        y_fit_1 = curve["function"](x_fit, *params_1)
        y_fit_2 = curve["function"](x_fit, *params_2)

        exi_1 = np.trapezoid(y_fit_1, x_fit)
        exi_2 = np.trapezoid(y_fit_2, x_fit)

        actual_delta_exi = np.abs(exi_1 - exi_2)

        surrogate_p_value = (np.sum(delta_exis_null >= actual_delta_exi) + 1) / (
            n_surrogates + 1
        )

        if ax is not None:
            xmin, xmax = ax.get_xlim()
            bins = np.linspace(xmin, xmax, 41)
            ax.hist(
                delta_exis_null,
                histtype="stepfilled",
                facecolor="#088F8F",
                alpha=0.4,
                edgecolor="#088F8F",
                bins=bins,
                label="$\\Delta$ExI null distr.",
            )
            ax.axvline(actual_delta_exi, color="black", label="$\\Delta$ExI")

    except Exception as e:
        surrogate_p_value = np.nan

    return surrogate_p_value_empirical, surrogate_p_value


def _significant_exi_difference_testing(args):
    (
        i,
        norm_ll_values_1,
        norm_ll_values_2,
        intensities,
        curve,
        max_iterations,
        rng_seed,
    ) = args

    np.random.seed(rng_seed + i)
    x_fit = np.linspace(0, 1, 1000)

    n_replications = norm_ll_values_1.shape[1]
    n_intensities = norm_ll_values_2.shape[0]
    pooled_ll_values = np.concatenate([norm_ll_values_1, norm_ll_values_2], axis=1)

    values_1 = np.empty_like(norm_ll_values_1)
    values_2 = np.empty_like(norm_ll_values_1)
    for i in range(n_intensities):
        idx = np.random.permutation(2 * n_replications)
        values_1[i] = pooled_ll_values[i, idx[:n_replications]]
        values_2[i] = pooled_ll_values[i, idx[n_replications:]]

    med_values_1 = np.nanmedian(values_1, axis=1)
    med_values_2 = np.nanmedian(values_2, axis=1)

    empirical_exi_1 = np.trapezoid(med_values_1, intensities)
    empirical_exi_2 = np.trapezoid(med_values_2, intensities)

    try:
        params_1 = fit_curve(
            curve_function=curve["function"],
            x=intensities,
            y=med_values_1,
            initial_values=curve["initial_values"],
            bounds=curve["bounds"],
            max_iterations=max_iterations,
        )
        params_2 = fit_curve(
            curve_function=curve["function"],
            x=intensities,
            y=med_values_2,
            initial_values=curve["initial_values"],
            bounds=curve["bounds"],
            max_iterations=max_iterations,
        )

        y_fit_1 = curve["function"](x_fit, *params_1)
        y_fit_2 = curve["function"](x_fit, *params_2)

        exi_1 = np.trapezoid(y_fit_1, x_fit)
        exi_2 = np.trapezoid(y_fit_2, x_fit)
        # TODO add r_squared requirement
    except RuntimeError as e:
        exi_1 = np.nan
        exi_2 = np.nan

    return np.abs(exi_1 - exi_2), np.abs(empirical_exi_1 - empirical_exi_2)


def find_params_for_given_effect_size(
    curve,
    orig_params,
    selected_param_index,
    target_exi,
    x,
    precision,
    max_iterations=10000,
):
    # selected param should be monotonic param to make bisection algorithm work

    # TODO add corner cases
    lo, hi = -40.0, 40.0
    current_exi = 100000
    iter = 0
    while np.abs(target_exi - current_exi) > precision:
        mid = 0.5 * (lo + hi)
        adj_params = orig_params.copy()
        adj_params[selected_param_index] += mid

        current_y_fit = curve["function"](x, *adj_params)
        current_exi = np.trapezoid(current_y_fit, x)
        if current_exi > target_exi:
            lo = mid
        else:
            hi = mid

        iter += 1
        if iter > max_iterations:
            raise Exception

    return adj_params


def pick_random_replications(x, n_replications_to_select):
    x_subset = []
    for intensity_index in range(x.shape[0]):
        valid_indices = np.where(np.isfinite(x[intensity_index]))[
            0
        ]  # remove -inf, +inf and NaN

        if len(valid_indices) == 0:
            print("Warning: No non-nan values for intensity index", intensity_index)
            x_subset.append(np.full(n_replications_to_select, 0))
            continue

        replace = n_replications_to_select > len(valid_indices)
        selected_indices = np.random.choice(
            valid_indices, size=n_replications_to_select, replace=replace
        )  # replacement is False, we do "Subsampling" here, but if not enough, fallback to sampling with replacement

        x_subset.append(x[intensity_index, selected_indices])

    # Concatenate the subsets
    x_subset = np.array(x_subset)  # shape: (n_intensities, n_replications_to_select)
    return x_subset
