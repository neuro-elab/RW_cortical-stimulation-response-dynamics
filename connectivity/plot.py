import math
import matplotlib.pyplot as plt
from connectivity.load import MultipleHDFResponseLoader
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import pandas as pd


def plot_responses(
    ax: plt.Axes, data: np.ndarray, f_sample: int, t_offset_seconds: float
):
    data = np.asarray(data)  # Ensure input is a numpy array
    if data.ndim == 1:
        data = data[np.newaxis, :]  # Convert to 2D with one row

    time = (np.arange(data.shape[1]) * 1 / f_sample) + t_offset_seconds
    for row in data:
        ax.plot(time, row, linewidth=1, color="black", alpha=0.1)

    ax.margins(x=0)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("EEG [uV]")


def plot_responses_grid(
    data: np.ndarray,
    f_sample: int,
    t_offset_seconds: float,
    n_cols: int = 6,
    figsize: tuple = (20, 35),
    plot_average: bool = False,
):
    """
    data shape: (n_replications, n_channels, n_time)
    """
    fig = plt.figure(figsize=figsize)
    n_plots = data.shape[1]
    n_rows = math.ceil(n_plots / n_cols)
    gs = GridSpec(n_rows, n_cols, figure=fig)

    time = np.arange(data.shape[2]) * 1 / f_sample + t_offset_seconds

    axes = []
    for i in range(n_plots):
        row, col = divmod(i, n_cols)  # Determine row and column
        ax = fig.add_subplot(gs[row, col])

        for trace in data[:, i, :]:
            ax.plot(time, trace, linewidth=1, color="black", alpha=0.5)
        if plot_average:
            ax.plot(
                time, np.mean(data[:, i, :], axis=0), linewidth=2, color="red", alpha=1
            )
        axes.append(ax)

    return fig, axes


def plot_spectrogram(
    ax: plt.Axes,
    ax_colorbar: plt.Axes,
    freq: np.ndarray,
    power: np.ndarray,
    f_sample: int,
    t_offset_seconds: float,
    vmax: float = None,
):
    time = (np.arange(power.shape[1]) * 1 / f_sample) + t_offset_seconds

    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [sec]")

    im = ax.pcolormesh(time, freq, power, shading="gouraud", vmax=vmax)

    fig = ax.figure
    cbar1 = fig.colorbar(im, cax=ax_colorbar, label="Power [a.u.]")


def plot_response_stimulation_curves(
    fig: plt.Figure,
    selected_intensities: np.ndarray,
    selected_channel_paths: list[str],
    ll_values: np.ndarray,
    sleep_score_matrix: np.ndarray = None,
    upper_bounds: np.ndarray = None,
    scatter_size: int = 1,
    normalize: bool = True,
):
    """
    upper_bounds: shape (n_channels)

    TODO: remove upper_bounds, remove normalize option
    """
    if normalize:
        norm_io_intensities = selected_intensities / np.max(selected_intensities)
    else:
        norm_io_intensities = np.array(selected_intensities)

    # Define the number of columns
    n_cols = 6
    n_plots = ll_values.shape[2]
    n_rows = math.ceil(n_plots / n_cols)
    gs = GridSpec(n_rows, n_cols, figure=fig)

    # Keep track of the first axis for sharing
    first_ax = None
    axes = []  # Store all axes to apply label_outer later

    # Define manual grouping and colors
    if sleep_score_matrix is not None:
        # TODO move to enums.py
        state_grouping = {
            "UNKNOWN": "other",
            "N3": "asleep",
            "N2": "asleep",
            "N1": "asleep",
            "REM": "asleep",
            "QWAKE": "awake",
            "AWAKE": "awake",
            "ICTAL": "ictal",
            "OTHER": "other",
            "ARTEFACT": "other",
        }
        # Assign manual colors to each group
        group_colors = {
            "awake": "#1f77b4",  # Blue
            "asleep": "#ff7f0e",  # Orange
            "ictal": "red",
            "other": "black",
            "None": "grey",
        }
        # Transform `state_matrix` to use grouped categories
        grouped_state_matrix = np.vectorize(lambda s: state_grouping.get(s, s))(
            sleep_score_matrix
        )
        legend_handles = [
            Patch(color=color, label=group) for group, color in group_colors.items()
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            title="States",
            bbox_to_anchor=(1.15, 1),
        )
    for i, channel in enumerate(selected_channel_paths):
        row, col = divmod(i, n_cols)  # Determine row and column
        ax = fig.add_subplot(gs[row, col], sharex=first_ax, sharey=first_ax)
        if first_ax is None:
            first_ax = ax  # Set the first axis for sharing
        axes.append(ax)

        # line length
        if normalize:
            max_ll_value = np.nanmax(ll_values[:, :, i])
            filtered_ll_values = ll_values[:, :, i] / max_ll_value
        else:
            filtered_ll_values = ll_values[:, :, i]

        # Expand `norm_io_intensities` to match `filtered_ll_values`
        x_values = np.tile(
            norm_io_intensities[:, np.newaxis], filtered_ll_values.shape[1]
        ).ravel()
        y_values = filtered_ll_values.ravel()  # Flatten

        if sleep_score_matrix is not None:
            states = grouped_state_matrix[:, :, i].ravel()
            colors = np.array(
                [group_colors[state] for state in states]
            )  # Map states to colors
        elif upper_bounds is not None:
            significant_mask = y_values > (upper_bounds[i] / max_ll_value)
            colors = np.where(
                significant_mask, "green", "red"
            )  # Green if above bound, red otherwise
            ax.axhline(
                y=upper_bounds[i] / max_ll_value,
                color="black",
                linestyle="-",
                label="Upper bound.",
            )
        else:
            colors = "black"  # Single color for all points

        ax.scatter(x_values, y_values, c=colors, s=scatter_size)
        ax.set_ylim([0, 1.1])

        n_datapoints = np.count_nonzero(~np.isnan(y_values))  # calculate non-nan values
        ax.set_title(f"{channel.split('/')[-1]} (n={n_datapoints})")
        ax.set_ylabel("Response [a.u.]")
        ax.set_xlabel("Stimulation intensities [a.u.]")

    # Apply label_outer to hide duplicate ticks and labels
    for ax in axes:
        ax.label_outer()

    return axes


def plot_stim_responses(
    ax: plt.Axes,
    mrl: MultipleHDFResponseLoader,
    id_matrix: np.ndarray,
    response_channel_path: str,
    avg_trace_color: str = "black",
    ignore_first_intensity: bool = False,
):
    """
    Plot the responses for a specific stimulation channel.

    Parameters:
    - ax: The matplotlib Axes object to plot on.
    - mrl: MultipleHDFResponseLoader instance to load the data.
    - id_matrix: A matrix containing the stimulation IDs in shape (n_intensities, n_replications)
    - response_channel_path: Selected response channel path to plot.
    """
    ind_matrix = mrl.get_inds_from_stim_ids(id_matrix)

    ll_window_start = round(
        1 * mrl.f_sample
    )  # we only want to have the [0, 0.5s] window to display, as it is used for LL calculation
    ll_window_end = round(1.5 * mrl.f_sample)
    chunk_len = ll_window_end - ll_window_start  # traces.shape[3]
    total_chunks = id_matrix.shape[0]
    time = np.arange(chunk_len * total_chunks) / mrl.f_sample

    # Plot each chunk with color-coded significance
    for j in range(total_chunks):
        if ignore_first_intensity and j == 0:
            continue
        try:
            traces = mrl.get_responses(
                stim_indices=ind_matrix[j],
                response_channel_paths=[response_channel_path],
                # overwrite_excluded_recordings=exclude_responses,
                t_start=-1,
                t_stop=1,
                overwrite_excluded_recordings=True,  # TODO do we want this?
            ).squeeze(
                1
            )  # shape: n_replications, 1, n_time, remove middle dim
        except:
            continue
        traces_chunk = traces[:, ll_window_start:ll_window_end]
        avg_trace_chunk = np.nanmean(traces_chunk, axis=0)  # shape: n_time

        start_idx = j * chunk_len
        end_idx = (j + 1) * chunk_len
        time_chunk = time[start_idx:end_idx]

        for trace in traces_chunk:
            ax.plot(time_chunk, trace, color="black", alpha=0.1, linewidth=0.5)
        ax.plot(
            time_chunk,
            avg_trace_chunk,
            color=avg_trace_color,
            linewidth=0.75,
        )

        start = start_idx / mrl.f_sample
        end = end_idx / mrl.f_sample
        ax.axvspan(
            start,
            end,
            facecolor=("lightgray" if j % 2 else "white"),
            alpha=0.3,
            zorder=0,
        )
    ax.set_ylabel("EEG [uV]")
    ax.margins(x=0, y=0)


# def plot_responses_grid(
#     fig: plt.Figure,
#     data: np.ndarray,
#     single_ax_plot_functon: callable,
#     n_cols: int = 6,
#     figsize: tuple = (20, 35),
# ):
#     """
#     params:
#     data: shape (n_channels, ...)
#     single_ax_plot_functon: callable, function to plot single ax, receives ax, slice of data, index
#     n_cols: int, number of columns
#     figsize: tuple, figure size
#     """
#     fig = plt.figure(figsize=figsize)
#     n_plots = data.shape[0]
#     n_rows = math.ceil(n_plots / n_cols)
#     gs = GridSpec(n_rows, n_cols, figure=fig)

#     for i in range(n_plots):
#         row, col = divmod(i, n_cols)  # Determine row and column
#         ax = fig.add_subplot(gs[row, col])
#         single_ax_plot_functon(ax, data[i], i)


def plot_response_stimulation_curve(
    ax: plt.Axes,
    side_ax: plt.Axes,
    fig: plt.Figure,
    response_channel_path: str,
    selected_channel_paths: list[str],
    selected_intensities: np.ndarray,
    ll_values: np.ndarray,
    id_matrix: np.ndarray,
    offset_seconds: float = -1,
    response_loader: MultipleHDFResponseLoader = None,
    sleep_score_matrix: np.ndarray = None,
    averaged_ll: bool = False,
    upper_bound: float = None,  # not normalized
):
    norm_io_intensities = selected_intensities / np.max(selected_intensities)
    selected_channel_index = selected_channel_paths.index(response_channel_path)
    max_ll_value = np.nanmax(ll_values[:, :, selected_channel_index])
    filtered_ll_values = ll_values[:, :, selected_channel_index] / max_ll_value
    for repetition in range(filtered_ll_values.shape[1]):
        time = np.arange(filtered_ll_values.shape[0])
        ax.scatter(
            norm_io_intensities,
            filtered_ll_values[:, repetition],
            gid=repetition,
            c="black",
            s=3,
        )
    ax.set_title(
        f"{response_channel_path.split('/')[-1]} (n={filtered_ll_values.shape[1]})"
    )
    ax.title.set_zorder(0)
    ax.set_ylabel("Response [a.u.]")
    ax.set_xlabel("Stimulation intensities [a.u.]")
    if upper_bound is not None:
        ax.axhline(
            upper_bound / max_ll_value,
            color="black",
            linestyle="--",
            label="Upper bound",
        )
    annotation = ax.annotate(
        text="",
        xy=(0, 0),
        size=5,
        xytext=(-100, -50),
        textcoords="offset points",
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", zorder=20
        ),
        arrowprops=dict(arrowstyle="->", lw=0.5),
    )
    annotation.set_visible(False)

    side_ax.set_zorder(0)

    # Inspired by https://stackoverflow.com/a/38377630/8287424
    def on_plot_hover(event):
        # Check for hover events on scatter points
        annotation.set_visible(False)  # Hide annotation by default
        for j, collection in enumerate(ax.collections):  # Iterate over repetitions
            contains, indices = collection.contains(event)
            if contains:
                # Get the indices of the hovered points
                hovered_index = indices["ind"][
                    0
                ]  # First matching point in this collection
                io_intensity = selected_intensities[hovered_index]
                response_value = filtered_ll_values[hovered_index, j]
                if averaged_ll:
                    stim = ", ".join(
                        s.split("/")[-1]
                        for s in id_matrix[hovered_index, :, selected_channel_index]
                    )
                else:
                    stim = id_matrix[hovered_index, j, selected_channel_index]
                sleep_score = (
                    sleep_score_matrix[hovered_index, j, selected_channel_index]
                    if sleep_score_matrix is not None
                    else ""
                )
                # Update the annotation
                annot_text = f"Intensity={io_intensity:.2f}mA\nLL={response_value:.2f}\nStim={stim}"
                if sleep_score_matrix is not None:
                    annot_text += f"\nSleep score: {sleep_score}"
                annotation.set_text(annot_text)
                annotation.xy = (
                    norm_io_intensities[hovered_index],
                    response_value,
                )  # Position of the annotation
                annotation.set_visible(True)
                break  # Stop after finding the first match
        ax.set_zorder(side_ax.get_zorder() + 1)
        fig.canvas.draw_idle()  # Redraw the figure to show updates

    logs = response_loader.get_logs()

    def on_plot_click(event):
        for j, collection in enumerate(ax.collections):  # Iterate over repetitions
            contains, indices = collection.contains(event)
            if contains:
                # Get the indices of the hovered points
                hovered_index = indices["ind"][0]

                side_ax.cla()
                side_ax.set_ylabel("EEG [uV]")
                side_ax.set_xlabel("Time [sec]")
                side_ax.set_xlim([-0.5, 1])

                if averaged_ll:
                    stim_ids = id_matrix[hovered_index, :, selected_channel_index]
                    stim_indices = logs[logs["stim_id"].isin(stim_ids)].index

                    traces = response_loader.get_responses(
                        stim_indices=stim_indices,
                        response_channel_paths=[response_channel_path],
                        t_start=-1,
                        t_stop=1,
                    ).squeeze(1)
                    time = (
                        np.arange(traces.shape[1]) / response_loader.f_sample
                        + offset_seconds
                    )
                    avg_trace = np.mean(traces, axis=0)
                    for repetition in range(traces.shape[0]):
                        side_ax.plot(
                            time,
                            traces[repetition],
                            label=f"{stim_ids[repetition].split('/')[-1]}",
                            alpha=0.5,
                        )
                    side_ax.plot(time, avg_trace, label=f"Avg.", color="black")
                    side_ax.legend()
                else:
                    stim_id = id_matrix[hovered_index, j, selected_channel_index]
                    stim_index = logs[logs["stim_id"] == stim_id].index

                    trace = response_loader.get_responses(
                        stim_indices=stim_index,
                        response_channel_paths=[response_channel_path],
                        t_start=-1,
                        t_stop=1,
                    ).reshape(
                        -1
                    )  # from (1, 1, 1024) to (1024)
                    time = (
                        np.arange(trace.shape[0]) / response_loader.f_sample
                        + offset_seconds
                    )
                    side_ax.plot(time, trace)
                    side_ax.set_title(f"{stim_id}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_plot_hover)
    if response_loader is not None:
        fig.canvas.mpl_connect("button_press_event", on_plot_click)


def plot_curve_fittings(
    curve_function: callable,
    x: np.ndarray,
    y: np.ndarray,
    params_list: list[list],
    names: list[str],
    fig: plt.Figure,
):
    # Generate fitted curve
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fits = []
    y_preds = []
    residuals_list = []
    sum_squared_residuals_list = []
    total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)
    r_squared_list = []

    gs = GridSpec(2, 1, figure=fig, height_ratios=(3, 1))
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    ax1.scatter(
        x,
        y,
        label=f"Original data",
        color="red",
        s=5,
        alpha=0.5,
    )

    for i, params in enumerate(params_list):
        y_fit = curve_function(x_fit, *params)
        y_pred = curve_function(x, *params)
        residuals = y - y_pred
        y_fits.append(y_fit)
        y_preds.append(y_pred)
        residuals_list.append(residuals)

        sum_squared_residuals = np.sum(residuals**2)
        r_squared = 1 - (sum_squared_residuals / total_sum_of_squares)
        r_squared_list.append(r_squared)

        ax1.plot(
            x_fit,
            y_fit,
            label=f"{names[i]}, R^2={r_squared:.3f}",
        )
        ax2.scatter(
            x, residuals, label=f"{names[i]}, R^2={r_squared:.3f}", s=5, alpha=0.5
        )

    ax1.legend()
    ax1.set_xlabel("Normalized intensities [a.u.]")
    ax1.set_ylabel("Normalized LL [a.u.]")
    ax1.set_title(f"Sigmoid Curve Fit")
    ax2.set_ylim(bottom=-0.1)

    ax2.legend()
    ax2.set_title("Residuals")

    return r_squared_list


def plot_bootstrap_curve_fittings(
    curve_function: callable,
    x: np.ndarray,
    y: np.ndarray,
    params_original: list,
    params_list: list[list],
    fig: plt.Figure,
):
    # Generate fitted curve
    x_fit = np.linspace(min(x), max(x), 1000)
    sum_squared_residuals_list = []
    total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)
    r_squared_list = []

    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0])

    ax1.scatter(
        x,
        y,
        label=f"Original data",
        color="red",
        s=5,
        alpha=0.5,
    )

    # original
    y_fit = curve_function(x_fit, *params_original)
    y_pred = curve_function(x, *params_original)
    residuals = y - y_pred

    sum_squared_residuals = np.sum(residuals**2)
    r_squared = 1 - (sum_squared_residuals / total_sum_of_squares)
    r_squared_list.append(r_squared)

    ax1.plot(
        x_fit,
        y_fit,
        label=f"Original, R^2={r_squared:.3f}",
    )

    y_fits = []
    y_preds = []
    residuals_list = []
    for i, params in enumerate(params_list):
        y_fit = curve_function(x_fit, *params)
        y_pred = curve_function(x, *params)
        residuals = y - y_pred
        y_fits.append(y_fit)
        y_preds.append(y_pred)
        residuals_list.append(residuals)

        sum_squared_residuals = np.sum(residuals**2)
        r_squared = 1 - (sum_squared_residuals / total_sum_of_squares)
        r_squared_list.append(r_squared)
    # calculate empirical CI
    y_fits = np.array(y_fits)
    ci = np.percentile(y_fits, [2.5, 97.5], axis=0)
    ax1.plot(x_fit, np.mean(y_fits, axis=0), label="Mean")
    ax1.fill_between(
        x_fit, ci[0], ci[1], color="blue", alpha=0.2, label="95% Bootstrap CI"
    )

    ax1.legend()
    ax1.set_xlabel("Normalized intensities [a.u.]")
    ax1.set_ylabel("Normalized LL [a.u.]")
    ax1.set_title(f"Sigmoid Curve Fit")
