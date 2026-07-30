# CytoAlign

CytoAlign의 현재 연구 방향은 새 imputation architecture 개발보다
cross-platform cytometry recoverability benchmark를 먼저 확립하는 것이다.
Common-marker backbone `H`만으로 target-exclusive marker `Y`가 언제
복원되고, 언제 calibration·support·conditional ambiguity 때문에 실패하는지
patient-level로 평가한다.

현재까지의 대화, processed AML 실험과 통합 해석은
[research record](docs/research_record.md)에 정리한다. 앞으로의 raw
benchmark 계약과 실행 gate는 [benchmark plan](docs/benchmark_plan.md)을
사용한다. 대체된 상세 결과 문서는
[`docs/archive/`](docs/archive/README.md)에 provenance로 보존한다.

Machine-readable protocol은
[`configs/benchmark/protocol_v1.yaml`](configs/benchmark/protocol_v1.yaml)에
있다. 현재 상태는 의도적으로 `draft`이며
data/marker/split/endpoint/reference-bank/stress manifest가 고정되기 전에는
full benchmark를 실행하지 않는다.

## 현재 우선순위

1. AML raw FCS technical-QC-only event-set rebuild
2. Raw data/marker/patient split/reference-bank manifest 동결과 full preflight
3. H19, clinical10과 frozen compact candidates의 within-platform 확인
4. Conditional ambiguity와 support failure-mode positive control
5. Target-prior/wrong-patient null을 포함한 AML 양방향 cross-platform benchmark
6. Validation-locked abstention과 Nuñez external validation

Clinical-flow H4 분석은 raw detector channel의 label-free audit를 통과한
경우에만 secondary stress test로 진행한다. 과거 OT/residual 실험은
processed-data conditional method triage로만 취급하며 benchmark primary
method로 사용하지 않는다.

## Setup

```bash
python -m pip install -e '.[neural,dev]'
```

LPC CUDA runtime과 맞는 PyTorch build가 필요하다.

## Protocol preflight

```bash
python -m src.wrappers.benchmark_preflight \
  --protocol configs/benchmark/protocol_v1.yaml
```

이 명령은 protocol digest와 누락된 manifest를 보고한다. Full run gate에서는
`--mode full`을 사용하며 frozen protocol, index/record/reference digest와
cross-manifest identifier consistency를 함께 검사한다.

## Tests

```bash
pytest -q
```

`data/`는 로컬 원자료이며 Git에 포함되지 않는다. `outputs/`, scheduler
logs, checkpoints와 large predictions도 disposable run state다. Release에는
frozen manifests, checksums, environment, patient-level OOF artifacts와
figure source tables를 별도로 보존한다.
