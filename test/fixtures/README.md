# `test/fixtures/`

Frozen reference data and intermediate JSON used by reproducibility /
regression tests. Files here are NOT used by the production code path —
they exist purely to compare a freshly computed result against a
known-good baseline.

### `feature_parity_fixture.json`

Snapshot of the V2 feature pipeline's intermediate tensors (raw input
columns + engineered features after `AppendDerivedFeatures`). Used by
`test/test_js_feature_parity.mjs` to verify that the JS port of the
feature pipeline agrees numerically with the Python implementation.
Regenerate with `python experiments/regenerate_feature_parity_fixture.py`.
