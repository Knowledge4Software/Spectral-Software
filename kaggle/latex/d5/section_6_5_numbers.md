# Section 6.5 - computed values

## BigCloneBench

- archive: `Section 6.5 timing breakdown - BigCloneBench-2.zip`
- accuracy this run: F1 **0.7619**, Acc **0.9323** (matches the RQ2 run)
- training batches: **112,608**
- mean measured cost per batch: **88.3 ms**
- total wall time: **234.3 min** over 4 epochs
- training throughput: **322 pairs/s**
- training: spectral block **5.1%**, of which eigendecomposition **1.1%**
- training: input handling **32.5%**
- inference: spectral block **7.1%**, of which eigendecomposition **1.6%**
- inference: input handling **60.9%**

## AtCoder

- archive: `Section 6.5 timing breakdown - AtCoder-2.zip`
- accuracy this run: F1 **0.8208**, Acc **0.8118** (matches the RQ2 run)
- training batches: **68,008**
- mean measured cost per batch: **83.7 ms**
- total wall time: **119.7 min** over 4 epochs
- training throughput: **339 pairs/s**
- training: spectral block **5.1%**, of which eigendecomposition **1.1%**
- training: input handling **31.7%**
- inference: spectral block **7.4%**, of which eigendecomposition **1.7%**
- inference: input handling **58.5%**

## CodeNet

- archive: `Section 6.5 timing breakdown - CodeNet_2.zip`
- accuracy this run: F1 **0.7391**, Acc **0.6864** (matches the RQ2 run)
- training batches: **8,724**
- mean measured cost per batch: **87.8 ms**
- total wall time: **16.1 min** over 4 epochs
- training throughput: **324 pairs/s**
- training: spectral block **5.2%**, of which eigendecomposition **1.1%**
- training: input handling **32.3%**
- inference: spectral block **7.1%**, of which eigendecomposition **1.6%**
- inference: input handling **60.8%**

## Across benchmarks

- spectral block during training: **5.1%–5.2%** of measured time

