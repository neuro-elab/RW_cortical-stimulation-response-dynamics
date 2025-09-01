# Assessing the excitability of cortico-cortical connections

## Structure

- `connectivity`: contains code to load and anlalyze data
- `publish`: contains code to generate the figures from the publication
- `*_pipeline.py`: contain pipeline code to run a specific part of the analysis

### Pipelines

- Significant responses: Determines effective connections, calculates SNR.
- Curve fitting: Calculates original fits for 2P, 3P, 4P, and 5P models. `interactive_curves.ipynb` provides an interactive way to check out the curves.
- Boostrap analysis: Calculates the boostrapped GT/subsampled CIs for the accuracy analysis
- Power analysis: Calculates the ad-hoc SRCs and analysis of the Delta Exis for a given effect size
- Average first: Compares average-first and magnitude-first approaches described in supplementary

The pipelines create a longform JSON file as a result. Every row corresponds to a single `patient_id`, `stim_channel_name`, and `response_channel_name` pair. In most files, those three can be used as a unique key to the row and can be used to join different tables.