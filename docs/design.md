# CytoAlign design

## Question

Given paired specimens measured with two cytometry panels, can common-marker
optimal transport transfer information from source-exclusive markers to
target-exclusive markers without paired cells?

## Training

1. Split patients into train, validation, and test.
2. Fit separate source and target marginal percentile transforms on training
   common markers.
3. Fit the deployable Ridge baseline `b(H, L)` on target training cells.
4. Within each training specimen and coarse cell type, compute balanced
   Sinkhorn transport using common markers only.
5. Project target-exclusive markers through the soft plan.
6. Train `r(H, X, L)` to predict the projected target residual over the
   baseline.
7. Select the residual scale on validation population Wasserstein.

At inference, CytoAlign uses source `H`, source-exclusive `X`, and coarse cell
type `L`. It does not use target cells or optimal transport.

## Comparisons

- Global target-training median
- Cell-type target-training median
- Ridge `H+L`
- kNN median `H+L`
- Direct MLP `H+L`
- OT residual distillation `H+L`
- CytoAlign OT residual distillation `H+X+L`

The `ot_hl` comparison isolates whether source-exclusive markers add value.

The residual baseline is configurable. The kNN-residual experiment replaces
`b(H,L)` with the cell-type-conditioned kNN median while leaving the OT teacher,
paired subsets, residual networks, and validation-selected scale unchanged:

`kNN(H,L) + alpha * r(H,X,L)`

Its count-zero condition is exactly plain kNN. Comparing the `H+L` and `H+X+L`
residuals tests whether specimen pairing and source-exclusive markers improve
on the strongest direct baseline.

## Paired-specimen dose response

For fold 0, target-training data, preprocessing, validation/test specimens, and
all baselines are fixed within each seed. Only the number of training specimens
whose source-target identity is exposed to the OT teacher changes:

`0, 1, 2, 4, 8, 16, 32`

The sets are nested prefixes of one seed-specific permutation. Count zero is
exactly the Ridge `H+L` baseline. The aggregate reports raw-count and
doubling-scale linear fits, monotonic improvement steps, the first count that
beats each baseline, and the first count where `H+X+L` beats `H+L` OT
distillation.

## Primary evaluation

All methods are evaluated on unseen paired specimens without constructing cell
pairs. The primary metric is patient-first, cell-type-stratified, normalized
1-Wasserstein distance. Five patient folds and three training seeds are
aggregated.
