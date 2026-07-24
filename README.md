# CytoAlign

CytoAlign은 cell-unpaired Spectral Flow `H+X`에서 CyTOF-exclusive `Y`를
translation할 때, H-only kNN이 잃는 rare fine-cell-type 정보를 `X`가
복원할 수 있는지 검증하는 연구 코드다.

현재 결론과 다음 실험 계약:

- [현재 결과와 다음 실험](docs/current_findings_and_next_experiment.md)
- [전체 실험 회고](docs/experiment_retrospective.md)
- [현재 구현 설계](docs/design.md)

과거 raw outputs, scheduler logs, compact records, archive slides, stopped
prototype와 local third-party baseline bundle은 문서로 통합한 뒤 제거했다.
`data/`는 로컬 원자료이며 Git에 포함되지 않는다.

## 현재 질문

다음 네 표현을 동일한 OT teacher와 residual capacity 아래 비교한다.

```text
1. kNN(H)
2. kNN(H) + residual(H)
3. kNN(H) + residual(H, shuffled X)
4. kNN(H) + residual(H, correct X)
```

primary endpoint는 translated `Y`만 사용한 DN/DP AUPRC와 recall이다.
cell-type label은 translator 학습에 들어가지 않고 held-out probe에만
사용한다.

## Setup

```bash
python -m pip install -e '.[neural,dev]'
```

LPC CUDA runtime과 맞는 PyTorch build가 필요하다.

## Test

```bash
pytest -q
```

## 5-fold run

```bash
QUEUE=dbeigpu \
GPU_REQUEST='num=1:mig=1/1:mode=shared:gmodel=NVIDIAH200' \
FOLDS='0 1 2 3 4' \
SEEDS='4207' \
scripts/submit_train.sh \
  configs/experiments/sf_to_cytof_rare_population_ablation_cv.yaml
```

queue, GPU request, CPU, memory, walltime와 Python executable은 각각
`QUEUE`, `GPU_REQUEST`, `CPU_CORES`, `MEMORY_MB`, `WALLTIME`,
`PYTHON_BIN`으로 지정한다.

모든 fold가 끝난 뒤 patient-first summary를 만든다.

```bash
python -m src.wrappers.cell_type_probe \
  --config configs/experiments/sf_to_cytof_rare_population_ablation_cv.yaml \
  --summarize
```

생성되는 `outputs/`와 `logs/`는 disposable run state이며 Git에
commit하지 않는다. 재실행 전에는 config와 Git commit을 결과 메타데이터에
기록한다.
