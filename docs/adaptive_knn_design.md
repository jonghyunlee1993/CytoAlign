# Target-conditioned adaptive kNN: first prototype

## Scope

The first go/no-go model is deliberately smaller than a query-adaptive neural
metric. It replaces the single Euclidean distance used by the kNN/CyTOFmerge
baseline with one learned diagonal metric per target-exclusive marker.

It does not use cell-type labels, source-target cell pairs, OT teachers,
residual networks, or uncertainty gates.

## Model

Let `H` be the common-marker space after the existing train-only empirical-CDF
calibration, `Y_t` one target-exclusive marker, and `M` the common markers
available in a panel.

For target marker `t`, learn non-negative weights

```text
w[t, :] = softmax(a[t, :])
```

and compute

```text
d_t(x, r | M) =
    sum(j in M) normalize_M(w[t, j]) * (H_source[x, j] - H_target[r, j])^2
```

The prediction is the median `Y_t` among the `k` target-reference cells nearest
under `d_t`. Each missing marker can therefore use a different neighborhood.
When the panel mask changes, weights are renormalized over the markers that are
actually available.

## Learning

Weights are learned using target-training cells only:

1. Hold out complete patients from the neighbor reference bank.
2. Predict each held-out target marker with a differentiable soft-neighbor
   approximation.
3. Minimize marker-IQR-normalized Huber loss.
4. Apply the 19/15/12/8 panel masks during training so weights remain usable
   after marker removal.
5. Select temperature and regularization on validation patients.
6. Use hard top-k median prediction only for final evaluation.

This is label-free in the cytometry sense: measured target proteins supervise
imputation, but no cell-type annotation is used.

## Go/no-go comparison

Run fold 0 with seeds 4207, 4208, and 4209 on all four panel masks:

1. Plain kNN
2. One global learned diagonal metric
3. Target-marker-specific diagonal metrics
4. Target-specific metrics trained with panel-mask dropout
5. Official CyTOFmerge
6. CytoVI with no label-informed prior

The prototype advances only if it:

- lowers Wasserstein by at least 1% relative to plain kNN and CyTOFmerge;
- improves macro marker-median Spearman by at least 0.01;
- does not worsen patient-first median error;
- improves rather than collapses in the 12- and 8-marker stress settings; and
- selects marker weights consistently across seeds.

If target-specific diagonal weights show a reproducible signal, the second
prototype may make the weights query-dependent. If the diagonal prototype does
not beat plain kNN, the adaptive-metric direction stops before adding a neural
gating network.

## First-screen implementation

`sf_to_cytof_adaptive_knn.yaml` runs fold 0 at seeds 4207--4209 and evaluates
the 19/15/12/8 realistic panel masks in one job per seed. Hyperparameters are
selected by pooled validation populations without using cell-type labels.

To keep the marker-specific search tractable, inference first finds the exact
512 nearest cells under the plain panel Euclidean metric and then reranks those
candidates independently for every target marker. Plain kNN uses the exact
top 50 from the same search. This first screen compares items 1--4 above.
Official CyTOFmerge and label-free CytoVI runs remain a second-stage benchmark
and are only justified if the adaptive metric clears the internal plain-kNN
go/no-go check.
