# HiRAC predecessor result summaries

These CSVs reproduce the exact values in the inherited HiRAC experiment
summaries used by manuscript Tables 2, 3 and 4. The author confirmed on
17 November 2026 that they are earlier real experiments, and that RaHUM
evolved from HiRAC. Values have not been jittered, smoothed or newly
assigned to the current native PointLLM checkpoint.

- `table2_hirac_sensing.csv`: two auxiliary sensing tasks, eight method/configuration
  rows and five conditions, preserved as 80 long-form rows. Rate is percent.
  F/P is sensing-forward PFLOPs / total parameters in billions, not false positives.
- `table3_hirac_response_errors.csv`: four non-exclusive response error types and
  their arithmetic sum, preserving the six source rows.
- `table4_hirac_sunny_ablation.csv`: five predecessor sensing-head ablations.

The predecessor reliable-group K and the native adapter local-reference K
are different controls. The supplied two-update native engineering
checkpoints, recorded generation, strict restoration, and real-input
diagnostics remain separate in `backbone/` and `diagnostics/`. These CSVs
do not imply a new held-out evaluation of those native checkpoints.

`SOURCE_PROVENANCE.json` retains source identities, metric units and scope.
The original HiRAC five-class confusion matrix and its separate attribution
are in `matlab/actual_plot_data_v9.json`.
