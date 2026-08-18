<div align="center">

# Robust Contextual Optimization with Missing Covariates

Official implementation of **Robust Contextual Optimization with Missing Covariates**,  
including code for the **ICML 2026 Spotlight/Oral paper** and the corresponding **journal paper**.

[![ICML 2026](https://img.shields.io/badge/ICML%202026-Spotlight%20%2F%20Oral-blue)](https://openreview.net/forum?id=37KrsS7g7c)
![Journal](https://img.shields.io/badge/Journal-Coming%20Soon-lightgrey)

</div>

---

## Papers

| Venue | Paper | Resources |
|:---|:---|:---:|
| **ICML 2026** | **Robust Contextual Optimization with Missing Covariates**  <br> *Qingyuan Xu and Ruiwei Jiang*  <br> **Spotlight / Oral** | [Paper](https://openreview.net/forum?id=37KrsS7g7c) · [Citation](#citation) |
| **Journal** | **Robust Contextual Optimization with Missing Covariates** | Paper & Code — **In Submission** |

---


## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{xu2026robust,
  title     = {Robust Contextual Optimization with Missing Covariates},
  author    = {Xu, Qingyuan and Jiang, Ruiwei},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=37KrsS7g7c}
}
```

---

## Installation
Python 3.9 or later is recommended.

The default examples use **MOSEK**. In this case a valid MOSEK license should be available in
the runtime environment. 

## Data Format

The solver expects a `pandas.DataFrame` containing:

| Column | Description |
|---|---|
| `X0`, `X1`, ... | Covariates; missing entries may be represented by `NaN`. |
| `Y` | Observed outcome. |
| `missing_mask` | Binary mask with one entry per covariate: `1` means missing and `0` means observed. |



## Method Implementation

### MAR KL-DRO for Newsvendor Problem

[`mar_kl_dro.py`](mar_kl_dro.py) provides an implementation of the
**MAR KL-DRO** method for contextual newsvendor problems with
partially observed covariates.

- Example

```python
from mar_kl_dro import MissingCovariateDRO

model = MissingCovariateDRO(
    covariate_supports=[[0, 1, 2]] * 3,
    outcome_support=range(13),
)

model.fit(df, solver="MOSEK")

# Solve the conditional newsvendor problem at X = (0, 0, 2).
result = model.solve_newsvendor(
    context=(0, 0, 2),
    radius=0.001,
    overage_cost=3.0,
    underage_cost=6.0,
    decision_bounds=(0, 15),
    solver="MOSEK",
)

print("Order quantity:", result.order_quantity)
print("Worst-case cost:", result.worst_case_cost)
```

- Custom column names
  
``` python
model = MissingCovariateDRO(
    covariate_supports=[[0, 1], [0, 1]],
    outcome_support=range(10),
    covariate_columns=["region", "segment"],
    outcome_column="demand",
    mask_column="mask",
)
```

### More to come 

