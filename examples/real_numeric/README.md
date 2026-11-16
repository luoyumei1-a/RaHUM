# Small real numeric samples

These twelve CSVs contain the complete `pc_xyziv_ti` and `pc_xyz_key` numeric fields from six source MAT records: two adjacent numeric frame IDs in each of three author-supplied radar groups. The exact source MAT files are distributed with the full code/example-data archive. The dense `pc_xyzb_kinect` field is omitted from this small text release.

Every value is printed with 17 significant decimal digits. Reloading each CSV using its original field dtype was verified to recover the exact source array bytes. There is no coordinate transformation, unit conversion, point generation, or attribute-to-color mapping. Headers `column_1` ... preserve the original source column order without adding unverified attribute semantics. `provenance.json` gives every source MAT SHA-256, archive member, field, shape, dtype, CSV SHA-256, and round-trip verification.

The source MAT metadata does not certify physical units, calibration transforms, capture timestamps, independent point labels, or scene-category labels. In particular, columns four and five of the radar array are not RGB channels. No separate road image is paired with these records. Source frame numbers indicate stored ordering, not a measured temporal interval.

Run `python verify_csv.py` to check all CSV file and restored-array hashes. This requires only NumPy. Dataset rights remain with the contributing authors; a software license for the surrounding repository does not assign a new data license. The authors' full collection access table is in `DATA_ACCESS.md`.
