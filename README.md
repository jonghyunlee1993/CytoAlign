# CytoAlign

CytoAlign의 현재 연구 방향은 새 imputation architecture 개발이 아니라
paired cytometry marker-information landscape benchmark다. AML과 Nuñez에서
shared backbone으로 어떤 immune marker가 복원되고, observed marker와 target
marker의 predictive relationship이 무엇이며, within-modality information이
cross-modality에서 얼마나 손실되는지를 patient-level로 평가한다.

완료된 processed AML 실험과 통합 해석은
[research record](docs/research_record.md)에 정리한다. AML–Nuñez landscape의
실행 계약은 [benchmark plan](docs/benchmark_plan.md)을 사용한다. 대체된
상세 결과 문서는
[`docs/archive/`](docs/archive/README.md)에 provenance로 보존한다.

Active machine-readable design은
[`configs/benchmark/landscape_v1.yaml`](configs/benchmark/landscape_v1.yaml)에
있다. 기존
[`protocol_v1.yaml`](configs/benchmark/protocol_v1.yaml)은 processed-AML
phase-0 audit와 관련 테스트를 위한 legacy contract다.

## 현재 우선순위

1. AML/Nuñez marker alias, native H19/H20과 universal H9 manifest 동결
2. 두 cohort × 두 modality의 within-modality marker-difficulty landscape
3. AML H9 subset screen과 marker-count–information Pareto frontier
4. AML에서 universal minimal S*를 lock하고 Nuñez에서 외부 평가
5. 양방향 `H` 대 `H+exclusive` cross-modality degradation benchmark
6. matched, patient-deranged, pooled/unpaired pairing comparison

Primary method는 median/prior, ridge, exact kNN50과 fixed simple MLP다. Raw
AML은 primary 선결조건이 아니라 processed-event selection sensitivity다.

## Setup

```bash
python -m pip install -e '.[neural,dev]'
```

LPC CUDA runtime과 맞는 PyTorch build가 필요하다.

## Legacy protocol preflight

```bash
python -m src.wrappers.benchmark_preflight \
  --protocol configs/benchmark/protocol_v1.yaml
```

이 명령은 기존 processed-AML phase-0 contract를 검사한다. Landscape용
manifest/preflight는 `landscape_v1.yaml` schema에 맞춰 새 세션에서
연결한다.

## Tests

```bash
pytest -q
```

`data/`는 로컬 원자료이며 Git에 포함되지 않는다. `outputs/`, scheduler
logs, checkpoints, predictions와 sampled-row cache는 disposable run state다.
Release에는 frozen manifests, checksums, environment, patient-level OOF
summary와 figure source tables만 별도로 보존한다.
