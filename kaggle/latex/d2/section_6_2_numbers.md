# Section 6.2 - computed values

## Validation BCE loss

- epochs trained: **40**
- epoch 1: **0.6011**
- lowest: **0.4407** at epoch **37**
- final epoch: **0.4580**
- total improvement: **0.1604**
- plateau epoch (under 1% of the gain left): **37**
- gain remaining after the plateau: **0.0000**

- same-language mean, final epoch: **0.4390**
- cross-language mean, final epoch: **0.4705**
- cross minus same: **+0.0315**

## Test accuracy (Table 12)

- epoch 1: **0.6388**
- best: **0.7999** at epoch **36**
- final epoch: **0.7929**
- gain from epoch 1 to best: **+0.1611**
- change from best to final: **-0.0070**

- same-language mean, final epoch: **0.7972**
- cross-language mean, final epoch: **0.7901**

- easiest configuration: **Java–Java** (0.8372)
- hardest configuration: **C++–C++** (0.7487)

- the loss curve decreases on **28** of **39** epoch transitions, rising on 11
