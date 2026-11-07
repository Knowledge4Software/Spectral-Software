# Stage 4: focused transfer, after comparator selection

The checked-in `spectra/` and `baselines/` files are the existing `table8|J->P`
starter jobs. They are **not** the five-condition focused study. Do not run the
entire folder as the revision transfer plan.

After selecting full SPECTRA, the stronger non-spectral sibling and an eligible
external comparator using validation, generate the five planned conditions:

```powershell
python scripts/build_revision_notebooks.py --transfer-key 'focused|source_only' --transfer-key 'focused|direct' --transfer-key 'focused|two_hop' --transfer-key 'focused|three_hop' --transfer-key 'focused|target_reference'
python scripts/prepare_revision_stage1.py
```

Run the selected models only. The focused plan contains 5 conditions × 3 models
× 3 seeds = 45 jobs, with 6,000 training pairs, 1,200 validation pairs and a
common Python-Python test; bridge conditions have matched target-label quotas.
Use `ARM='mean_max'` or `'attention'` in the selected SPECTRA sibling jobs; their
output IDs distinguish them from full. See the exact quotas and limitations in
[the protocol](../00_plan/EXPERIMENT_PROTOCOL.md) and
[focused_transfer_plan.csv](../queues/focused_transfer_plan.csv).

The generator emits extra comparator candidates. Their presence does not mean
they must all run. Test evaluation remains off until choices are frozen.
