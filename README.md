# Assessing the excitability of cortico-cortical connections

## Structure

`connectivity`: contains code to load and anlalyze data
`publish`: contains code to generate the figures from the publication

`*_pipeline.py`: contain pipeline code to run a specific part of the analysis

### Pipelines

Significant responses: Determines effective connections, calculates SNR.
Curve fitting: Calculates original fits for 2P, 3P, 4P, and 5P models. `interactive_curves.ipynb` provides an interactive way to check out the curves.
boostrap analysis: Calculates the boostrapped GT/subsampled CIs for the accuracy analysis
power analysis: Calculates the ad-hoc SRCs and analysis of the Delta Exis for a given effect size
average first: Compares average-first and magnitude-first approaches described in supplementary
