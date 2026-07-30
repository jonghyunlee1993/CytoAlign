# Frozen benchmark manifests

이 디렉터리는 작은 version-controlled benchmark contract만 저장한다.
각 하위 디렉터리는 protocol이 직접 가리키는 `index.yaml`, digest가 고정된
UTF-8 `records.csv`, 그리고 필요한 row-index/condition content file을 가진다.
대용량 row-index는 일반 Git object로 commit하지 않고 content-addressed release
artifact 또는 Git LFS에 두되, local full preflight에서는 실제 file과 digest를
검증한다.

Landscape v1에서 예정된 하위 디렉터리:

- `data/`: specimen, patient, visit, checksum, event-selection provenance
- `markers/`: AML/Nuñez alias, native H19/H20, universal H9, S* contract
- `splits/`: patient-grouped frozen outer/validation split
- `endpoints/`: target marker와 major/rare endpoint coverage
- `banks/`: role별 reference bank와 patient/specimen/row-index budget
- `panels/`: H9 subset screen과 locked Pareto candidates
- `pairing/`: pooled, patient-deranged, patient-matched fixed-budget conditions

Active design은 `configs/benchmark/landscape_v1.yaml`이다. 기존
`configs/benchmark/protocol_v1.yaml`은 processed-AML phase-0 audit test를
위한 legacy draft다. Landscape manifest schema와 preflight를 연결하기
전에는 active design을 `frozen`으로 바꾸지 않는다.
