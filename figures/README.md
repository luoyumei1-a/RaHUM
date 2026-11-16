# Article plotting source

`RaHUM_Figures_v7.m` is one self-contained MATLAB file, ordered as article Figure 2 (four nonexclusive response-error types) then Figure 3 (outdoor auxiliary sensing rates). It was executed in MATLAB Online R2026b Update 2 (`26.2.0.3437455`); final source SHA-256: `4ee526cf2787850681358ec604235d39ce6b3e9b1428dc04e97fb1bf0094809a`.

The source measurements are the existing author-reported HiRAC full-adaptation values, displayed as RaHUM (full adaptation). They are not newly measured performance of the separately recorded two-update frozen-prefix checkpoint in `pointllm_native_v6`. No samples or uncertainty intervals are invented from rounded rates.

Run `RaHUM_Figures_v7(outputDirectory,'all')` in MATLAB Online. The file exports editable FIG, vector PDF, PNG and provenance JSON in article order.
