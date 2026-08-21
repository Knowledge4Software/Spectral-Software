# Result status after the fidelity revision

The CSV/ZIP results archived before this directory was introduced are **not**
results of the paper-faithful adaptations.  Do not relabel or reuse them.

Required reruns:

- DECKARD, ASTNN, RtvNN, CDLH, DeepSim, FA-AST+GGNN, and FA-AST+GMN on all four datasets;
- every method reported on GPTCloneBench, because its pair set was rebuilt without identity pairs;
- cross-dataset experiments whose target is GPTCloneBench;
- any aggregate or “best method” table derived from one of the stale rows.

Each regenerated notebook writes `*_faithful_results.csv`, a training history
for neural methods, and `*_faithful_manifest.json`.  Only those manifests carry
the explicit implementation provenance needed by the revised paper.
