# Physical input units and transformations

This note clarifies field semantics without changing any cached values or frozen run receipts.

| Quantity | Recorded meaning and unit | Transformation in the native input |
|---|---|---|
| Radar XYZ | Source measurement coordinates in metres | Calibration and ego-pose transforms establish common metric coordinates; the PointBERT view is separately centred and radius-normalized. Geometry context retains its documented metric/pixel convention. |
| Compensated velocity | nuScenes `vx_comp`, `vy_comp`, m/s | The separate physical side channel projects the ego-motion-compensated horizontal velocity onto the measured ray direction. This signed radial component is derived; it is not an independently observed full 3D target velocity. |
| RCS | Radar cross section in **dBsm** | Source `rcs` values are retained and pooled using the actual group indices. No amplitude logarithm, intensity rescaling or conversion to square metres is performed. A mean of dBsm values is a pooled descriptor, not mean linear reflected power. |
| Projection and spatial sigma | Native camera-image pixels for this RGB-patch input | The recorded sigma is 49.224761962890625 pixels, derived from the actual patch-grid spacing after the recorded encoder resizing/cropping. It is not metres or a universal calibration tolerance. |
| Sensor timestamps | Source integer microseconds | Pairing comparisons retain source timestamps and report offset in microseconds; any elapsed-time feature records its own conversion. |
| Utility / attention | Dimensionless learned utility probabilities / normalized attention weights | The current two-way utility softmax does not multiply by attention or a 1/N factor. These are model diagnostics, not measured sensor confidence or independently calibrated correctness probabilities. |

RCS in dBsm means 10 log10(sigma / 1 m²), where sigma is radar cross section. It should not be called log-amplitude, RGB intensity, SNR or a noise probability. The inherited generic validation message `[vr,log-amplitude]` does not rename the source RCS channel. Frozen fields such as `source_rcs_native_unconverted` remain unchanged for provenance.

## Primary documentation

- nuScenes devkit `RadarPointCloud.from_file` documents its 18 fields and distinguishes compensated from uncompensated velocities: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py . The inspected source blob identity was `fcc757b839079895865a788a077fd7f733535544`.
- NVIDIA NCore's official nuScenes conversion documentation explicitly maps radar cross section to `rcs_dBsm` and the projection of compensated velocity to `radial_velocity_m_s`: https://nvidia.github.io/ncore/conversions/nuscenes/nuscenes.html .
- Official dataset format and sensor descriptions: https://www.nuscenes.org/data-format .

The dBsm mapping is explicitly supported by the official NVIDIA converter documentation; the inspected nuScenes SDK field comments themselves name `rcs` without spelling out its unit. This distinction is retained instead of attributing an unstated unit annotation to that source.
