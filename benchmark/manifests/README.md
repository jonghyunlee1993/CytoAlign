# Frozen benchmark manifests

이 디렉터리는 작은 version-controlled benchmark contract만 저장한다.
각 하위 디렉터리는 protocol이 직접 가리키는 `index.yaml`, digest가 고정된
UTF-8 `records.csv`, 그리고 필요한 row-index/condition content file을 가진다.
대용량 row-index는 일반 Git object로 commit하지 않고 content-addressed release
artifact 또는 Git LFS에 두되, local full preflight에서는 실제 file과 digest를
검증한다.

예정된 하위 디렉터리:

- `data/`: specimen, patient, visit, checksum, event-selection provenance
- `markers/`: direction별 H/Y/source-only/excluded marker contract
- `splits/`: patient-grouped frozen outer/validation split
- `endpoints/`: endpoint coverage, conflicts, eligibility
- `banks/`: role별 reference bank와 patient/specimen/row-index budget
- `stress/`: semi-synthetic condition과 random subset draws

현재 `configs/benchmark/protocol_v1.yaml`은 `draft`다. Current processed AML
audit는 source/row digest까지 생성됐지만 conditional sensitivity일 뿐이다.
Raw technical-QC-only AML manifest와 endpoint conflict rule이 확정되고
preflight가 통과하기 전에는 `frozen`으로 바꾸지 않는다.
