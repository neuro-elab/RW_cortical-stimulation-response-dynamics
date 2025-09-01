import json
import warnings
import time

import numpy as np
import pandas as pd

import h5py
from connectivity.enums import SleepStage, TimeGrade, TraceGrade


class HDFDataLoader:
    def __init__(self, file_path: str) -> None:
        """
        Initialize the loader with the path to the .h5 file.
        """
        self.file_path = file_path
        self._file = h5py.File(file_path, "r")  # Open the file in read-only mode
        # print(self._file.keys())

    def _fetch_datasets(self, group, current_path=""):
        """
        Recursively fetch the path of all datasets starting from the given group.
        """
        dataset_paths = []

        for name, item in group.items():
            if isinstance(item, h5py.Dataset):
                dataset_paths.append(f"{current_path}/{name}")  # Add dataset path
            elif isinstance(item, h5py.Group):
                # Recurse into the subgroup
                dataset_paths.extend(
                    self._fetch_datasets(item, f"{current_path}/{name}")
                )
        return dataset_paths

    def get_all_dataset_paths(self, group_path: str) -> list[str]:
        """
        Get a list of all dataset paths in the file, starting from the given group.
        """
        return self._fetch_datasets(self._file[group_path], current_path=group_path)

    def get_data(
        self,
        dataset_paths: list[str],
        start_row_index: int,
        end_row_index: int,
        return_np: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieve a slice of data from the specified datasets within the .h5 file.
        The slice goes from start_row_index to end_row_index.
        """
        # Note: The dataframe reconstruction makes the implementation slower
        # For faster computation time, only rely on numpy array
        columns = None

        data_slices = []
        for dataset_name in dataset_paths:
            if dataset_name in self._file:
                # Access the dataset directly
                dataset = self._file[dataset_name]
                expected_length = end_row_index - start_row_index
                # Load only the slice of data needed
                data_slice = dataset[start_row_index:end_row_index]
                actual_length = len(data_slice)
                if actual_length < expected_length:
                    # Pad with NaN
                    padding = np.full(expected_length - actual_length, np.nan)
                    data_slice = np.concatenate([data_slice, padding])
                data_slices.append(data_slice)
            else:
                raise ValueError(
                    f"Dataset {dataset_name} not found in the file {self.file_path}."
                )
        if return_np:
            return np.array(data_slices)
        data_slices = np.array(data_slices).T
        channel_names = [path.split("/")[-1] for path in dataset_paths]
        return pd.DataFrame(data_slices, columns=channel_names)

    def get_dataset(self, dataset_path: str) -> h5py.Dataset:
        return self._file[dataset_path]

    def does_dataset_exist(self, dataset_path: str) -> bool:
        return dataset_path in self._file.keys()

    def get_attrs(self, dataset_path: str):
        return self._file[dataset_path].attrs

    def get_trace_length(self, dataset_path: str) -> int:
        return len(self._file[dataset_path])

    def close(self):
        """
        Close the .h5 file.
        """
        self._file.close()


class HDFResponseLoader(HDFDataLoader):
    def __init__(
        self,
        path_h5: str,
        path_logs: str,
        path_lookup: str = None,
        path_excluded_recordings: str = None,
    ):
        super().__init__(file_path=path_h5)
        traces_path = "traces/bipolar/lead"

        self.logs = pd.read_csv(path_logs)

        if path_lookup is not None:
            self.lookup_channels = pd.read_excel(path_lookup, sheet_name="channels")
            self.lookup_bipoles = pd.read_excel(path_lookup, sheet_name="bipoles")

            bipoles = self.lookup_bipoles[
                (self.lookup_bipoles["sm_pos"] >= 0)
                & (self.lookup_bipoles["sm_neg"] >= 0)
            ]

            sm_to_name_df = self.lookup_channels[self.lookup_channels["sm"] >= 0][
                ["sm", "name"]
            ]
            sm_to_name_mapping = dict(zip(sm_to_name_df["sm"], sm_to_name_df["name"]))

            self.bipole_mapping = {
                row["name"]: {
                    "name_pos": sm_to_name_mapping[row["sm_pos"]],
                    "name_neg": sm_to_name_mapping[row["sm_neg"]],
                }
                for _, row in bipoles.iterrows()
            }
        else:
            self.lookup_bipoles = None
            self.lookup_channels = None
            self.bipole_mapping = None

        self.channel_paths = self.get_all_dataset_paths(traces_path)
        # here we assume that sfreq is the same for all bipolar channels
        self.f_sample = self.get_attrs(self.channel_paths[0])["sfreq"]

        # lazy-loaded
        self._time_grades = None
        self._annotations = None
        self._sleep_scores = None

        self.excluded_recordings = None
        if path_excluded_recordings:
            self._load_excluded_recordings(
                path_excluded_recordings=path_excluded_recordings
            )

    def _load_excluded_recordings(self, path_excluded_recordings: str):
        with open(path_excluded_recordings, "r") as f:
            excluded_recordings_dict = json.load(f)

        inverted_resp_dict = {}
        for key, values in excluded_recordings_dict.items():
            for value in values:
                if value not in inverted_resp_dict:
                    inverted_resp_dict[value] = []
                inverted_resp_dict[value].append(key)

        self.excluded_recordings = inverted_resp_dict

    def get_channel_paths(
        self,
        exclude_stim_channels: bool = False,
        stim_channel_name_pos: str = None,
        stim_channel_name_neg: str = None,
        exclude_noisy_channels: bool = False,
        exclude_wm_only_channels: bool = False,
        exclude_out_channels: bool = False,
    ) -> list[str]:
        channel_paths = []
        if exclude_stim_channels:
            assert self.bipole_mapping is not None
            assert stim_channel_name_pos is not None
            assert stim_channel_name_neg is not None
            for channel_path in self.channel_paths:
                bipole_name = channel_path.split("/")[-1]
                if bipole_name in self.bipole_mapping:
                    name_pos = self.bipole_mapping[bipole_name]["name_pos"]
                    name_neg = self.bipole_mapping[bipole_name]["name_neg"]

                    if not (
                        stim_channel_name_pos in [name_pos, name_neg]
                        or stim_channel_name_neg in [name_pos, name_neg]
                    ):
                        # response
                        channel_paths.append(channel_path)
                else:
                    # TODO find better solution
                    # it assumes that stimulating channels are in bipole mapping
                    print("Warning: Channel not found in bipole mapping", channel_path)
                    channel_paths.append(channel_path)
        else:
            channel_paths = self.channel_paths

        if exclude_noisy_channels:
            filtered_channel_paths = []
            num_noisy_channels = 0
            for channel_path in channel_paths:
                if not self.is_channel_path_noisy(channel_path=channel_path):
                    filtered_channel_paths.append(channel_path)
                else:
                    num_noisy_channels += 1
            if num_noisy_channels > 0:
                print(f"Excluded {num_noisy_channels} noisy channels.")
        else:
            filtered_channel_paths = channel_paths

        if exclude_wm_only_channels or exclude_wm_only_channels:
            n_out_channels = 0
            n_wm_only_channels = 0
            include_channel_paths = []
            assert self.bipole_mapping is not None
            for channel_path in filtered_channel_paths:
                bipole_name = channel_path.split("/")[-1]
                if bipole_name in self.bipole_mapping:
                    name_pos = self.bipole_mapping[bipole_name]["name_pos"]
                    name_neg = self.bipole_mapping[bipole_name]["name_neg"]

                    candidates_pos = self.lookup_channels.loc[
                        self.lookup_channels["name"] == name_pos, ["isWM", "isOut"]
                    ].values
                    assert len(candidates_pos) == 1
                    is_wm_pos = candidates_pos[0][0]
                    is_out_pos = candidates_pos[0][1]
                    candidates_neg = self.lookup_channels.loc[
                        self.lookup_channels["name"] == name_neg, ["isWM", "isOut"]
                    ].values
                    assert len(candidates_neg) == 1
                    is_wm_neg = candidates_pos[0][0]
                    is_out_neg = candidates_pos[0][1]

                    include_channel = True
                    if exclude_wm_only_channels and (is_wm_pos and is_wm_neg):
                        include_channel = False
                        n_wm_only_channels += 1
                    if exclude_out_channels and (is_out_pos or is_out_neg):
                        include_channel = False
                        n_out_channels += 1

                    if include_channel:
                        include_channel_paths.append(channel_path)

            if n_wm_only_channels > 0:
                print(f"Excluded {n_wm_only_channels} WM-only channels.")
            if n_out_channels > 0:
                print(f"Excluded {n_out_channels} out/putament/ventricle channels.")

            filtered_channel_paths = include_channel_paths
        return filtered_channel_paths

    def get_bipole_mapping(self):
        assert self.bipole_mapping is not None, "Bipole mapping is not available."

        return self.bipole_mapping

    def is_channel_path_noisy(self, channel_path: str):
        return self._file[channel_path].attrs["grade"] == TraceGrade.NOISY.name

    def get_stimulating_channel_names(self, protocol: str = None, join: bool = True):
        relevant_logs = self.logs
        if protocol is not None:
            relevant_logs = self.logs[self.logs["type"] == protocol]

        stim_channels = relevant_logs[["name_pos", "name_neg"]].drop_duplicates()
        if join:
            return stim_channels.agg("-".join, axis=1).tolist()
        else:
            return stim_channels

    def is_stimulating_channel_path(self, channel_path: str, protocol: str = None):
        assert self.bipole_mapping is not None
        bipole_name = channel_path.split("/")[-1]

        stim_channels = self.get_stimulating_channel_names(
            protocol=protocol, join=False
        )
        affected_contacts = (
            stim_channels["name_pos"].tolist() + stim_channels["name_neg"].tolist()
        )

        if bipole_name in self.bipole_mapping:
            name_pos = self.bipole_mapping[bipole_name]["name_pos"]
            name_neg = self.bipole_mapping[bipole_name]["name_neg"]

            return name_neg in affected_contacts or name_pos in affected_contacts
        else:
            print("Warning: Channel not found in bipole mapping", channel_path)
            return False

    def get_traces(
        self,
        times_sec: list[float],
        response_channel_paths: list[str],
        t_start: int = -4,
        t_stop: int = 2,
    ):
        res = []
        for time in times_sec:
            start_index = round((time + t_start) * self.f_sample)
            stop_index = round((time + t_stop) * self.f_sample)

            channels = super().get_data(
                dataset_paths=response_channel_paths,
                start_row_index=start_index,
                end_row_index=stop_index,
                return_np=True,
            )
            res.append(channels)
        res = np.array(res)

        return res

    def get_lookup_channels(self):
        return self.lookup_channels

    def get_channel_paths_from_names(
        self, channel_names: list[str], ignore_errors: bool = False
    ):
        channel_paths = []
        for name in channel_names:
            # Find the path that ends with the current name
            matches = [path for path in self.channel_paths if path.endswith(name)]
            if not ignore_errors:
                assert len(matches) == 1
                channel_paths.append(matches[0])
            else:
                if len(matches) == 1:
                    channel_paths.append(matches[0])
                else:
                    channel_paths.append(
                        "Unknown channel path"
                    )  # TODO find better solution

        return channel_paths

    def get_destrieux_labels_from_names(
        self, channel_names: list[str], short_form=False
    ):
        assert self.bipole_mapping is not None

        destrieux_labels = []
        for channel_name in channel_names:
            if channel_name in self.bipole_mapping:
                name_pos = self.bipole_mapping[channel_name]["name_pos"]
                name_neg = self.bipole_mapping[channel_name]["name_neg"]

                # Take label from positive contact
                candidates_pos = self.lookup_channels.loc[
                    self.lookup_channels["name"] == name_pos, "destrieux_corrected"
                ].values
                assert len(candidates_pos) == 1
                label = candidates_pos[0]
                if label == "WM":
                    # if positive in WM, take negative
                    candidates_neg = self.lookup_channels.loc[
                        self.lookup_channels["name"] == name_neg, "destrieux_corrected"
                    ].values
                    assert len(candidates_neg) == 1
                    label = candidates_neg[0]

                if short_form:
                    label = (
                        label.replace("ctx_rh_", "")
                        .replace("ctx_lh_", "")
                        .replace("Right-", "")
                        .replace("Left-", "")
                    )
                destrieux_labels.append(label)
            else:
                destrieux_labels.append(
                    "Unknown destrieux label"
                )  # TODO find better solution
        return destrieux_labels

    def get_labels_from_names(self, channel_names: list[str]):
        assert self.bipole_mapping is not None
        labels = []
        for channel_name in channel_names:
            name_pos = self.bipole_mapping[channel_name]["name_pos"]
            name_neg = self.bipole_mapping[channel_name]["name_neg"]

            # Take label from positive contact
            candidates_pos = self.lookup_channels.loc[
                self.lookup_channels["name"] == name_pos, "label"
            ].values
            assert len(candidates_pos) == 1
            label = candidates_pos[0]
            if label == "WM":
                # if positive in WM, take negative
                candidates_neg = self.lookup_channels.loc[
                    self.lookup_channels["name"] == name_neg, "label"
                ].values
                assert len(candidates_neg) == 1
                label = candidates_neg[0]

            labels.append(label)
        return labels

    def get_regions_from_names(self, channel_names: list[str]):
        assert self.bipole_mapping is not None
        regions = []
        for channel_name in channel_names:
            if channel_name in self.bipole_mapping:
                name_pos = self.bipole_mapping[channel_name]["name_pos"]
                name_neg = self.bipole_mapping[channel_name]["name_neg"]

                # Take region from positive contact
                candidates_pos = self.lookup_channels.loc[
                    self.lookup_channels["name"] == name_pos, "region"
                ].values
                assert len(candidates_pos) == 1
                region = candidates_pos[0]
                if region == "WM":
                    # if positive in WM, take negative
                    candidates_neg = self.lookup_channels.loc[
                        self.lookup_channels["name"] == name_neg, "region"
                    ].values
                    assert len(candidates_neg) == 1
                    region = candidates_neg[0]

                regions.append(region)
            else:
                regions.append("Unknown region")  # TODO find better solution
        return regions

    def get_responses(
        self,
        stim_indices: list[int],
        response_channel_paths: list[str],
        t_start: int = -4,
        t_stop: int = 2,
        return_stim_ids: bool = False,
        return_paired_pulse_probing_stims: bool = False,
        overwrite_excluded_recordings: bool = False,
    ):
        """
        Returns a np.ndarray with shape (n_stim, n_response_channel, n_time)
        """
        res = []
        for stim_index in stim_indices:
            if return_paired_pulse_probing_stims:
                ttl_index = self.logs.iloc[stim_index]["TTL_PP_DS"]
            else:
                ttl_index = self.logs.iloc[stim_index]["TTL_DS"]

            start_index = round(ttl_index + t_start * self.f_sample)
            stop_index = round(ttl_index + t_stop * self.f_sample)

            # assert start_index >= 0, stop_index >= 0

            channels = self.get_data(
                dataset_paths=response_channel_paths,
                start_row_index=start_index,
                end_row_index=stop_index,
                return_np=True,
            )
            if overwrite_excluded_recordings:
                assert self.excluded_recordings is not None
                if stim_index in self.excluded_recordings:
                    overwrite_channel_indices = [
                        i
                        for i, val in enumerate(response_channel_paths)
                        if val in self.excluded_recordings[stim_index]
                    ]
                    channels[overwrite_channel_indices] = np.nan
            elif np.isnan(np.min(channels)):
                print("Warning: Channel contains NaN values.", start_index, stop_index)
            res.append(channels)
        res = np.array(res)

        if return_stim_ids:
            stim_ids = self.logs.iloc[stim_indices]["stim_id"].to_numpy()
            return res, stim_ids

        return res

    def _overwrite_excluded_recordings(self, stim_index: int, traces: np.ndarray):
        pass

    def get_logs(self):
        return self.logs

    def add_sleep_score_to_logs(self):
        # TODO add similar lazy loading as for time grades
        scores = self.get_dataset("sleep_grades/text")
        durations = self.get_dataset("sleep_grades/duration")
        times = self.get_dataset("sleep_grades/time")
        assert len(scores) == len(durations) == len(times)

        first_raw_channel_path = self.get_all_dataset_paths("traces/raw")[0]
        dim_raw_trace = self.get_trace_length(dataset_path=first_raw_channel_path)

        raw_f_sample = self.get_attrs(first_raw_channel_path)["sfreq"]

        sleep_score = np.zeros(dim_raw_trace)
        for i, score in enumerate(scores):
            score_str = score.decode("utf-8")
            start_index = round(times[i] * raw_f_sample)
            stop_index = round(start_index + durations[i] * raw_f_sample)
            sleep_score[start_index:stop_index] = SleepStage[score_str].value

        int_to_str = np.vectorize(lambda x: SleepStage(x).name)
        self.logs["sleep_score"] = int_to_str(sleep_score[self.logs["TTL"].values])

    def _lazy_load_time_grades(self):
        if self.does_dataset_exist("time_grades") == True:
            grades = self.get_dataset("time_grades/text")
            durations = self.get_dataset("time_grades/duration")
            times = self.get_dataset("time_grades/time")
        else:
            grades = []
            durations = []
            times = []

        assert len(grades) == len(durations) == len(times)

        dim_trace = self.get_recording_length()

        time_grades = np.zeros(dim_trace)
        for i, grade in enumerate(grades):
            grade_str = grade.decode("utf-8")
            start_index = round(times[i] * self.f_sample)
            stop_index = round(start_index + durations[i] * self.f_sample)
            time_grades[start_index:stop_index] = TimeGrade[grade_str].value

        self._time_grades = time_grades

    def _lazy_load_annotations(self):
        annotations = self.get_dataset("annotations/text")
        times = self.get_dataset("annotations/time")
        assert len(annotations) == len(times)

        dim_trace = self.get_recording_length()
        annot = [""] * dim_trace

        for i, annotation in enumerate(annotations):
            annotation_str = annotation.decode("utf-8")
            annot[round(times[i] * self.f_sample)] = annotation_str

        self._annotations = annot

    def _lazy_load_sleep_scores(self):
        scores = self.get_dataset("sleep_grades/text")
        durations = self.get_dataset("sleep_grades/duration")
        times = self.get_dataset("sleep_grades/time")
        assert len(scores) == len(durations) == len(times)

        dim_trace = self.get_recording_length()

        sleep_score = np.zeros(dim_trace)
        for i, score in enumerate(scores):
            score_str = score.decode("utf-8")
            start_index = round(times[i] * self.f_sample)
            stop_index = round(start_index + durations[i] * self.f_sample)
            sleep_score[start_index:stop_index] = SleepStage[score_str].value

        self._sleep_scores = sleep_score

    def get_time_grades(
        self,
        stim_indices: list[int],
        t_start: int = -4,
        t_stop: int = 2,
    ):
        """
        Returns all specified time grades from t_start to t_stop during stimulation.

        returns a list with length n_stim_indices
        """
        if self._time_grades is None:
            self._lazy_load_time_grades()

        stim_time_indices = self.logs.iloc[stim_indices][
            "TTL_DS"
        ].to_numpy()  # TODO support for paired pulse
        res = []
        for stim_time_index in stim_time_indices:
            start_index = round(stim_time_index + t_start * self.f_sample)
            end_index = round(stim_time_index + t_stop * self.f_sample)
            unique_values = np.unique(self._time_grades[start_index:end_index])
            unique_values = unique_values[unique_values > 0].tolist()
            res.append([TimeGrade(x).name for x in unique_values])

        return res

    def get_annotations(
        self,
        stim_indices: list[int],
        t_start: int = -4,
        t_stop: int = 2,
    ):
        """
        Returns all specified annotations from t_start to t_stop during stimulation.

        returns a list with length n_stim_indices
        """
        if self._annotations is None:
            self._lazy_load_annotations()

        stim_time_indices = self.logs.iloc[stim_indices][
            "TTL_DS"
        ].to_numpy()  # TODO support for paired pulse
        res = set()
        for stim_time_index in stim_time_indices:
            start_index = round(stim_time_index + t_start * self.f_sample)
            end_index = round(stim_time_index + t_stop * self.f_sample)

            unique_values = set(self._annotations[start_index:end_index])

            res.update(unique_values)  # Add elements to the result set
        res.discard("")

        return list(res)

    def get_recording_length(self):
        first_channel_path = self.get_channel_paths()[0]
        length = self.get_trace_length(first_channel_path)

        return length

    def get_all_sleep_scores(self):
        if self._sleep_scores is None:
            self._lazy_load_sleep_scores()

        return self._sleep_scores

    def get_all_time_grades(self):
        if self._time_grades is None:
            self._lazy_load_time_grades()

        return self._time_grades

    def get_properties_of_channel(self, channel_name: str, properties: list[str]):
        row = self.lookup_bipoles[self.lookup_bipoles["name"] == channel_name]

        properties_dict = {}
        if len(row) == 1:
            for property in properties:
                properties_dict[property] = row[property].values[0]
        else:
            print("Warning: Channel not found in lookup.", channel_name)
            for property in properties:
                properties_dict[property] = None
        return properties_dict


class MultipleHDFResponseLoader:
    def __init__(
        self,
        paths_h5: list[str],
        paths_logs: list[str],
        recording_names: list[str],
        path_lookup: str = None,
        path_excluded_responses: str = None,
    ):
        # create array with response loaders
        # check that all channel paths are equal
        # open all stimlists, concat and preset stim id with recording
        assert len(paths_h5) == len(paths_logs) == len(recording_names)

        self._response_loaders: list[HDFResponseLoader] = []
        logs_list: list[pd.DataFrame] = []

        channel_paths_list = []
        self.intersection_of_channel_paths = None
        for i, (path_h5, path_logs) in enumerate(zip(paths_h5, paths_logs)):
            response_loader = HDFResponseLoader(
                path_h5=path_h5,
                path_logs=path_logs,
                path_lookup=path_lookup,
                path_excluded_recordings=(
                    path_excluded_responses
                    if path_excluded_responses is not None
                    else None
                ),
            )
            channel_paths = response_loader.get_channel_paths()
            if not self.intersection_of_channel_paths:
                self.intersection_of_channel_paths = set(channel_paths)
            else:
                self.intersection_of_channel_paths &= set(channel_paths)
            channel_paths_list.append(channel_paths)

            self._response_loaders.append(response_loader)
            logs_list.append(response_loader.get_logs())

        # check that all have the same channels
        channel_paths_set_list = [set(paths) for paths in channel_paths_list]
        reference_channels = channel_paths_set_list[0]  # use first as a reference
        for i, channel_set in enumerate(channel_paths_set_list):
            if channel_set != reference_channels:
                missing = reference_channels - channel_set
                extra = channel_set - reference_channels

                if missing:
                    warnings.warn(
                        f"Not all recordings have the same channel names. We use the first recording as a "
                        f"reference and keep only channels that are available in all recording. Missing channels in recording {recording_names[i]}: {sorted(missing)}"
                    )
                if extra:
                    warnings.warn(
                        f"Not all recordings have the same channel names. We use the first recording as a "
                        f"reference and keep only channels that are available in all recording. Extra channels in recording {recording_names[i]}: {sorted(extra)}"
                    )

        # concat logs, append rec id to stimlist
        for i, (logs, recording_name) in enumerate(zip(logs_list, recording_names)):
            logs["stim_id"] = recording_name + "/" + logs["stim_id"].astype(str)
            logs["temp_recording_index"] = i
            logs["temp_within_file_index"] = logs.index

        # take sampling frequency from first response loader
        self.f_sample = self._response_loaders[0].f_sample
        self.logs = pd.concat(logs_list, ignore_index=True)

        self.paths_h5 = paths_h5
        self.paths_logs = paths_logs
        self.recording_names = recording_names
        self.path_lookup = path_lookup
        self.path_excluded_responses = path_excluded_responses

    def get_logs(self) -> pd.DataFrame:
        return self.logs

    def get_channel_paths(
        self,
        exclude_stim_channels: bool = False,
        stim_channel_name_pos: str = None,
        stim_channel_name_neg: str = None,
        exclude_noisy_channels: bool = False,
        exclude_wm_only_channels: bool = False,
        exclude_out_channels: bool = False,
    ):
        channel_paths = []
        if exclude_stim_channels:
            assert stim_channel_name_pos is not None
            assert stim_channel_name_neg is not None

            bipole_mapping_dict = self._response_loaders[0].get_bipole_mapping()

            for channel_path in self.intersection_of_channel_paths:
                bipole_name = channel_path.split("/")[-1]
                if bipole_name in bipole_mapping_dict:
                    name_pos = bipole_mapping_dict[bipole_name]["name_pos"]
                    name_neg = bipole_mapping_dict[bipole_name]["name_neg"]

                    if not (
                        stim_channel_name_pos in [name_pos, name_neg]
                        or stim_channel_name_neg in [name_pos, name_neg]
                    ):
                        # response
                        channel_paths.append(channel_path)
                else:
                    # TODO find better solution
                    # it assumes that stimulating channels are in bipole mapping
                    print("Warning: Channel not found in bipole mapping", channel_path)
                    channel_paths.append(channel_path)
        else:
            channel_paths = self.intersection_of_channel_paths

        if exclude_noisy_channels:
            filtered_channel_paths = []
            num_noisy_channels = 0
            for channel_path in channel_paths:
                is_noisy = False
                for response_loader in self._response_loaders:
                    if response_loader.is_channel_path_noisy(channel_path=channel_path):
                        num_noisy_channels += 1
                        is_noisy = True
                        break

                if not is_noisy:
                    filtered_channel_paths.append(channel_path)

            if num_noisy_channels > 0:
                print(f"Excluded {num_noisy_channels} noisy channels.")
        else:
            filtered_channel_paths = channel_paths

        if exclude_wm_only_channels or exclude_wm_only_channels:
            bipole_mapping_dict = self._response_loaders[0].get_bipole_mapping()
            lookup_channels = self._response_loaders[0].get_lookup_channels()

            n_out_channels = 0
            n_wm_only_channels = 0
            include_channel_paths = []
            assert bipole_mapping_dict is not None
            for channel_path in filtered_channel_paths:
                bipole_name = channel_path.split("/")[-1]
                if bipole_name in bipole_mapping_dict:
                    name_pos = bipole_mapping_dict[bipole_name]["name_pos"]
                    name_neg = bipole_mapping_dict[bipole_name]["name_neg"]

                    candidates_pos = lookup_channels.loc[
                        lookup_channels["name"] == name_pos, ["isWM", "isOut"]
                    ].values
                    assert len(candidates_pos) == 1
                    is_wm_pos = candidates_pos[0][0]
                    is_out_pos = candidates_pos[0][1]
                    candidates_neg = lookup_channels.loc[
                        lookup_channels["name"] == name_neg, ["isWM", "isOut"]
                    ].values
                    assert len(candidates_neg) == 1
                    is_wm_neg = candidates_pos[0][0]
                    is_out_neg = candidates_pos[0][1]

                    include_channel = True
                    if exclude_wm_only_channels and (is_wm_pos and is_wm_neg):
                        include_channel = False
                        n_wm_only_channels += 1
                    if exclude_out_channels and (is_out_pos or is_out_neg):
                        include_channel = False
                        n_out_channels += 1

                    if include_channel:
                        include_channel_paths.append(channel_path)

            if n_wm_only_channels > 0:
                print(f"Excluded {n_wm_only_channels} WM-only channels.")
            if n_out_channels > 0:
                print(f"Excluded {n_out_channels} out/putament/ventricle channels.")

            filtered_channel_paths = include_channel_paths

        return sorted(filtered_channel_paths)

    def is_stimulating_channel_path(self, channel_path: str, protocol: str = None):
        return self._response_loaders[0].is_stimulating_channel_path(
            channel_path, protocol=protocol
        )

    def get_stimulating_channel_names(
        self,
        protocol: str = None,
        join: bool = True,
    ):
        return self._response_loaders[0].get_stimulating_channel_names(
            protocol=protocol, join=join
        )

    def add_sleep_score_to_logs(self):
        logs_list: list[pd.DataFrame] = []
        for rl in self._response_loaders:
            rl.add_sleep_score_to_logs()
            logs_list.append(rl.get_logs())

        self.logs = pd.concat(logs_list, ignore_index=True)

    def get_time_grades(
        self,
        stim_indices: list[int],
        t_start: int = -4,
        t_stop: int = 2,
    ):
        index_groups = {}

        # Group indices by recording_index
        for i in stim_indices:
            recording_index = self.logs.iloc[i]["temp_recording_index"]

            if recording_index not in index_groups:
                index_groups[recording_index] = []
            index_groups[recording_index].append(i)

        grades_list = []
        for recording_index, indices in sorted(index_groups.items()):
            within_indices = self.logs.iloc[indices]["temp_within_file_index"]
            grades = self._response_loaders[recording_index].get_time_grades(
                stim_indices=within_indices,
                t_start=t_start,
                t_stop=t_stop,
            )
            grades_list.extend(grades)
        return grades_list

    def get_annotations(
        self,
        stim_indices: list[int],
        t_start: int = -4,
        t_stop: int = 2,
    ):
        index_groups = {}

        # Group indices by recording_index
        for i in stim_indices:
            recording_index = self.logs.iloc[i]["temp_recording_index"]

            if recording_index not in index_groups:
                index_groups[recording_index] = []
            index_groups[recording_index].append(i)

        annotations_list = []
        for recording_index, indices in sorted(index_groups.items()):
            within_indices = self.logs.iloc[indices]["temp_within_file_index"]
            grades = self._response_loaders[recording_index].get_annotations(
                stim_indices=within_indices,
                t_start=t_start,
                t_stop=t_stop,
            )
            annotations_list.extend(grades)
        return annotations_list

    def get_responses(
        self,
        stim_indices: list[int],
        response_channel_paths: list[str],
        t_start: int = -4,
        t_stop: int = 2,
        return_stim_ids: bool = False,
        return_paired_pulse_probing_stims: bool = False,
        overwrite_excluded_recordings: bool = False,
    ):
        index_groups = {}

        # Group indices by recording_index
        for i in stim_indices:
            recording_index = self.logs.iloc[i]["temp_recording_index"]

            if recording_index not in index_groups:
                index_groups[recording_index] = []
            index_groups[recording_index].append(i)

        res_list = []
        for recording_index, indices in sorted(index_groups.items()):
            within_indices = self.logs.iloc[indices]["temp_within_file_index"]
            res = self._response_loaders[recording_index].get_responses(
                stim_indices=within_indices,
                response_channel_paths=response_channel_paths,
                t_start=t_start,
                t_stop=t_stop,
                return_stim_ids=False,
                return_paired_pulse_probing_stims=return_paired_pulse_probing_stims,
                overwrite_excluded_recordings=overwrite_excluded_recordings,
            )
            res_list.append(res)
        res = np.concatenate(res_list, axis=0)

        if return_stim_ids:
            stim_ids = self.logs.iloc[stim_indices]["stim_id"].to_numpy()
            return res, stim_ids

        return res

    def get_traces(
        self,
        recording_indices: list[int],
        times_sec: list[float],
        response_channel_paths: list[str],
        t_start: int = -4,
        t_stop: int = 2,
    ):
        assert len(recording_indices) == len(times_sec)
        index_groups = {}

        # Group indices by recording_index
        for i, recording_index in enumerate(recording_indices):
            if recording_index not in index_groups:
                index_groups[recording_index] = []
            index_groups[recording_index].append(times_sec[i])

        res_list = []
        for recording_index, times in sorted(index_groups.items()):
            res = self._response_loaders[recording_index].get_traces(
                times_sec=times,
                response_channel_paths=response_channel_paths,
                t_start=t_start,
                t_stop=t_stop,
            )
            res_list.append(res)

        res = np.concatenate(res_list, axis=0)
        return res

    def get_channel_paths_from_names(
        self, channel_names: list[str], ignore_errors: bool = False
    ):
        return self._response_loaders[0].get_channel_paths_from_names(
            channel_names=channel_names, ignore_errors=ignore_errors
        )

    def get_destrieux_labels_from_names(
        self, channel_names: list[str], short_form=False
    ):
        return self._response_loaders[0].get_destrieux_labels_from_names(
            channel_names=channel_names, short_form=short_form
        )

    def get_labels_from_names(self, channel_names: list[str]):
        return self._response_loaders[0].get_labels_from_names(
            channel_names=channel_names
        )

    def get_regions_from_names(self, channel_names: list[str]):
        return self._response_loaders[0].get_regions_from_names(
            channel_names=channel_names
        )

    def get_recording_lengths(self):
        recording_lengths = []
        for response_loader in self._response_loaders:
            recording_lengths.append(response_loader.get_recording_length())
        return recording_lengths

    def get_all_sleep_scores(self):
        sleep_scores = []
        for response_loader in self._response_loaders:
            sleep_scores.append(response_loader.get_all_sleep_scores())

        return sleep_scores

    def get_all_time_grades(self):
        time_grades = []
        for response_loader in self._response_loaders:
            time_grades.append(response_loader.get_all_time_grades())

        return time_grades

    def get_properties_of_channel(self, channel_name: str, properties: list[str]):
        return self._response_loaders[0].get_properties_of_channel(
            channel_name=channel_name, properties=properties
        )

    def get_inds_from_stim_ids(self, stim_ids: list[str]):
        """
        Returns the indices of the stim_ids in the logs.
        """
        stim_ids_array = np.asarray(stim_ids)
        flat_ids = stim_ids_array.flatten()

        # Map stim_id to row position (integer-based)
        stim_id_to_pos = pd.Series(
            np.arange(len(self.logs)), index=self.logs["stim_id"]
        )

        # Vectorized lookup
        result = stim_id_to_pos.reindex(flat_ids, fill_value=-1).to_numpy().astype(int)
        return result.reshape(stim_ids_array.shape)

    def close(self):
        for rl in self._response_loaders:
            rl.close()


class Atlas:
    def __init__(
        self,
        path_to_atlas: str,
    ):
        self._atlas = pd.read_csv(path_to_atlas)

    def get_regions(self, destrieux_labels: list[str]):
        regions = []
        for label in destrieux_labels:
            region = self._atlas.loc[self._atlas["destrieux"] == label, "region"].values
            if len(region) == 0:
                print(f"Label {label} not found in atlas.")
                regions.append("Unknown region")
            else:
                regions.append(region[0])
        return regions

    def get_global_order(self, destrieux_labels: list[str]):
        order = []
        for label in destrieux_labels:
            plot_order = self._atlas.loc[
                self._atlas["destrieux"] == label, "anatomical_order_in_region"
            ].values
            if len(plot_order) == 0:
                print(f"Label {label} not found in atlas.")
                order.append(-1)
            else:
                order.append(plot_order[0])
        return order


def get_h5_names_of_patient(
    base_path: str,
    patient_id: str,
    protocol: str = None,
    new_overview_format: bool = False,  # deprecated
    only_sleep_graded_and_ok: bool = True,  # only for new format
):
    with open(f"{base_path}/overview.json", "r") as json_file:
        overview = json.load(json_file)

    if protocol is not None:
        names_h5 = [
            recording["file"]
            for recording in sorted(
                [
                    r
                    for r in (
                        overview[patient_id]["electrophy"]
                        if new_overview_format
                        else overview[patient_id]
                    )
                    if r["protocol"] == protocol
                    and (
                        not new_overview_format
                        or not only_sleep_graded_and_ok
                        or (r["sleep"] and r["isOk"])
                    )
                ],
                key=lambda r: r["block"],
            )
        ]
    else:
        names_h5 = [recording["file"] for recording in overview[patient_id]]
    return names_h5


def _none_to_nan(obj):
    if obj is None:
        return np.nan
    if isinstance(obj, (list, tuple)):
        return [_none_to_nan(x) for x in obj]
    return obj


def parsed_list_to_numpy_array(obj):
    return np.array(_none_to_nan(obj), dtype=float)
