# Article-ordered MATLAB figures v10

Run `RaHUM_Figures_v10('figures','all')` in MATLAB. The package has one `.m` source, including its local recorded-input rendering function. The export order follows the article: Figures 2 and 3, six radar-input/annotation panels for Figure 10, then the entropy, CPU-time and matrix charts for Figures 11–13. The recorded-input panels use three named, separately identified real scenes; the legacy export asset stems stay stable. Horizontal tick labels remain level and point-cloud backgrounds are white.

The native utility, memory and timing plots use the strictly restored fixed native checkpoint and actual real feature window. K=8,16,24,32,48,64,96,128 changes inference reference memory; 42 geometrically supported groups are observed at every budget. Mean/P10/P90 summarize these groups, not independent test scenes or a confidence interval. Adapter-only CPU timings exclude encoders, projector and language decoder.

The five-class confusion matrix reproduces the author-provided earlier-configuration array exactly and retains its identity in `actual_plot_data_v10.json`; it is not a new native PointLLM checkpoint evaluation. The additional scene CSVs and box footprints are measured inputs and independent publisher annotations, not SOTA/RaHUM predictions. The MATLAB renderer does not use them as training labels.

`online_outputs/` contains the actual MATLAB Online R2026b PDF/FIG exports and execution log. `ONLINE_EXPORT_RECEIPT.json` binds those exports to the exact source hash. `PLOT_SOURCE_VALIDATION.json` and `actual_plot_data_v10.json` preserve full-precision arrays and distinct populations. No output from an earlier version is relabelled as a v10 export.
