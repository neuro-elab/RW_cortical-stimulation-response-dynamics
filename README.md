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

The pipelines create a long-form JSON file as output. Each row corresponds to a single combination of `patient_id`, `stim_channel_name`, and `response_channel_name`. In most files, these three fields form a unique key for the row and can be used to join tables.