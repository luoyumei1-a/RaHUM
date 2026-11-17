# Article-ordered MATLAB figures (v9)

Run `RaHUM_Figures_v9('figures','all')` in MATLAB Online.
The single source exports Figures 2, 3, 10, 11, and 12 in article order.
Figures 2/3/10/11 use the actual native checkpoint diagnostics. Figure 12
reproduces the original five-class HiRAC predecessor matrix, separately scoped
from native PointLLM generation. The local-memory sweep uses fixed weights.
Entropy P10/P90 are descriptive group quantiles, not confidence intervals.
Adapter-only CPU timing excludes the encoders, projector, and 7B decoder.
