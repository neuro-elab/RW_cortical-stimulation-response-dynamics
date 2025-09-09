# Assessing the excitability of cortico-cortical connections

## Structure

- `connectivity`: contains code to load and analyze data
- `publish`: contains code to generate the figures from the publication
- `*_pipeline.py`: contain pipeline code to run a specific part of the analysis

### Pipelines

- **Significant responses:** Determines effective connections and calculates SNR.
- **Curve fitting:** Calculates original fits for 2P, 3P, 4P, and 5P models. `interactive_curves.ipynb` provides an interactive way to inspect the curves.
- **Bootstrap analysis:** Calculates the bootstrapped ground-truth (GT) and subsampled confidence intervals (CIs) for the accuracy analysis.
- **Power analysis:** Calculates ad hoc SRCs and analyzes the delta axis for a given effect size.
- **Average-first:** Compares the average-first and magnitude-first approaches described in the supplementary material.

The pipelines creates a long-form JSON file as output. Each row corresponds to a single combination of `patient_id`, `stim_channel_name`, and `response_channel_name`. In most files, these three fields form a unique key for the row and can be used to join tables.


### Data availability

In the folder `data`, the .json output files of all pipelines are available. It contains also all SRCs of our dataset (in `response_channels_lf.json`), including fitted parameters of the curves (in `curve_fitting_lf.json`).

### Stimulation-response curve tutorial

#### 1. Calculate line-length values

For this step, response epochs are required. In our analysis, we used [-1, 1] sec epochs with the stimulation happening in the middle. We run the analysis per patient and stimulation channel and hence had the following shape of the epochs numpy array: (`n_intensities`, `n_replications`, `n_response_channels`, `n_time`). One can then use the method `calculate_pointwise_line_length_max` provided in `analyze.py` to get the $LL_{max}$ values:

```
ll_max = calculate_pointwise_line_length_max(
    data=epochs, offset_stim_seconds=1, f_sample=f_sample
)
```

To get a single value per intensity, we used the median to be more robust against outliers:

```
ll_med = np.nanmedian(ll_max, axis=1)
```

To normalize the SRCs, we normalized the intensities using `intensities/np.max(intensities)` and the line-length values using `normalize_ll_values(..., use_min=True)` method provided in `analyze.py`.

#### 2. Connection significance

Only for effective connections it is reasonable to do curve fitting and calculate the excitability index (otherwise, the ExI should be defined as 0). We made good experiences using the Spearman correlation to determine effective connections:

```
ideal_ranks = np.arange(len(io_intensities)) + 1
spearman_p_values = []
for i, response_channel in enumerate(response_channels):
    spearman_rho, spearman_p_value = scipy.stats.spearmanr(ll_med[:, i], ideal_ranks)
    spearman_p_values.append(float(spearman_p_value))
```

Since this test leads to many tests per participant, we corrected the p-values for multiple comparisons using the Benjamini-Hochberg procedure:

```
spearman_p_values_fdr_corrected = scipy.stats.false_discovery_control(spearman_p_values, method="bh")
```

#### 3. Curve fitting

The proposed 5P asymmetrical growth sigmoid can be found in `curves.py`. Besides the function, also initial values and bounds are provided. 

```
curve = CURVES["5P"]

curve["name"] # access the name
curve["function"] # access the curve function
curves["param_names"] # access the parameter order and names
curves["initial_values"] # access initial values for params
curves["bounds"] # access bounds for constrained optimization
```

To curve fit, one can use the `fit_curve` method provided in `analyze.py`. It uses scipy's `curve_fit`.

Example usage:

```
params = fit_curve(
    curve_function=curve["function"],
    x=norm_intensities,
    y=norm_med_ll,
    initial_values=curve["initial_values"],
    bounds=curve["bounds"],
    max_iterations=1000,
)
```

`params` contains the fitted parameters. Since it is not guaranteed to get convergence, we suggest to wrap it in a `try ... except`.

To calculate the area under the curve, one can use numpy's `trapezoid` method:

```
empirical_exi = np.trapezoid(norm_med_ll, norm_intensities)

x_fit = np.linspace(0, 1, 1000)
y_fit = curve["function"](x_fit, *params)
exi = np.trapezoid(y_fit, x_fit)
```
