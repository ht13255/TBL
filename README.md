# OpenPhotonTwin

GUI 없이 Python에서 바로 import해 쓰는 **시간빈·광섬유 루프 양자 포토닉스 하드웨어 디지털 트윈**입니다. 작은 Fock 상태의 양자 간섭과 실제 장비의 시간축·손실·제어 지연·검출 과정을 한 패키지에서 다룹니다.

> Python 3.10+ · NumPy/SciPy 기반 · MIT License

## 빠른 시작

### 소스 코드 다운로드

GitHub에서 저장소를 clone한 뒤 editable 설치를 권장합니다.

```bash
git clone https://github.com/ht13255/TBL.git
cd TBL
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

저장소 전체가 필요하지 않다면 GitHub의 **Code → Download ZIP**으로 내려받을 수 있습니다. 패키지만 설치할 때는 저장소 루트에서 `python -m pip install .`을 실행합니다.

### 첫 실행

```python
import openphotontwin as opt

photon = opt.Wavepacket(temporal_width=20e-12, wavelength=1550e-9)
print(photon)
```

## 설치 방법

```python
import openphotontwin as opt
```

## 제공 기능

- 정규화 Gaussian 단일광자 파동묶음, 시간·주파수·편광·순도 기반 모드 중첩
- exact permanent 기반 수동 선형광학 Fock 시뮬레이션(작은 광자 수)
- 빔스플리터, 동적 빔스플리터, 위상천이, 지연선, 손실, 위상 드리프트
- 시간빈 큐비트, 광섬유 루프, 길이 오차, 동적 out-coupling
- EOM 스위치, 유한 extinction ratio, 전자 제어/FPGA feed-forward latency
- SNSPD 효율, time jitter, dead time, dark counts, time-tag 생성
- HOM dip, coincidence histogram, photon-number distribution
- CSV time-tag 로딩, SciPy HOM 자동 피팅, indistinguishability·loss 및 손실 구간 추정, ideal/experiment 비교
- Perceval, Strawberry Fields, GDSFactory/SAX S-parameter import 어댑터
- NumPy CPU 및 선택적 CuPy batch propagation
- pandas 결과 테이블과 Matplotlib 플롯(파일 저장 가능, GUI 불필요)

## 설치

Python 3.10 이상에서:

```bash
python -m pip install .
```

개발 도구까지 설치하려면:

```bash
python -m pip install ".[dev]"
```

CUDA/CuPy 환경에서는 해당 CUDA 버전에 맞는 CuPy를 설치한 뒤 `backend="cupy"`를 지정합니다. Perceval, Strawberry Fields, GDSFactory/SAX는 필요한 어댑터를 사용할 때만 별도로 설치하면 됩니다.

## 1. HOM 실험

```python
import numpy as np
import openphotontwin as opt

photon_a = opt.Wavepacket(temporal_width=20e-12, wavelength=1550e-9)
photon_b = opt.Wavepacket(temporal_width=20e-12, wavelength=1550e-9)

result = opt.hom_scan(
    np.linspace(-100e-12, 100e-12, 101),
    photon_a,
    photon_b,
    shots_per_delay=10_000,
    detector_efficiency=0.92,
    seed=7,
)
print(result.visibility)
result.as_dataframe().to_csv("hom.csv", index=False)
result.plot().figure.savefig("hom.png", dpi=160)
```

## 2. Exact Fock 회로

```python
import openphotontwin as opt

circuit = opt.LinearOpticalCircuit(2).beam_splitter(0, 1, reflectivity=0.5)
distribution = opt.FockSimulator(circuit).probabilities([1, 1])

print(distribution.probabilities)  # (1,1)은 HOM 억제, (2,0)/(0,2)는 각 0.5
print(distribution.loss_probability)
```

손실이 있는 transfer matrix에서는 출력 확률의 합이 1보다 작고 나머지는 `loss_probability`로 보고됩니다. 두 광자의 부분 비구별성은 정확히, 3광자 이상은 평균 pair-overlap 보간 근사로 계산됩니다.

## 3. 시간축 광섬유 루프 디지털 트윈

```python
import openphotontwin as opt

source = opt.SinglePhotonSource(
    repetition_rate=10e6,
    p_single=0.8,
    p_double=0.01,
    emission_jitter=5e-12,
)

loop = opt.FiberLoop(
    round_trip_time=20e-9,
    transmission=0.94,
    outcoupling={1: 0.0, 2: 0.2, 3: 1.0},
    phase_drift_std=0.01,
    max_roundtrips=4,
)

detectors = opt.DetectorArray({
    0: opt.SNSPD(efficiency=0.9, jitter=20e-12, dead_time=40e-9, channel=0)
})

twin = opt.DigitalTwin(source, [loop], detectors, time_bin_width=20e-9)
run = twin.run(10_000, seed=42)
print(run.photon_number_distribution)
run.save_time_tags("time_tags.csv")
```

## 4. 동적 스위칭과 feed-forward

```python
import openphotontwin as opt

controller = opt.FeedForwardController(latency=12e-9, jitter=100e-12)
controller.trigger(time=0.0, value=1.0)

switch = opt.EOMSwitch(
    mode_a=0,
    mode_b=1,
    control=controller,
    control_latency=2e-9,
    rise_time=500e-12,
    extinction_ratio_db=35,
)

dynamic_bs = opt.DynamicBeamSplitter(
    0, 1, reflectivity=0.5,
    schedule={0: 0.0, 2: 0.5, 4: 1.0},
    switching_time=200e-12,
)
```

## 5. 실험 데이터 자동 보정

CSV에는 최소한 `time,channel` 열이 필요하며 `shot`은 선택입니다.

```python
import openphotontwin as opt

tags = opt.load_time_tags("time_tags.csv")
hist = opt.coincidence_histogram(tags, 0, 1, bin_width=20e-12, max_delay=2e-9)

fit = opt.fit_hom_dip(delays, measured_coincidences)
loss = opt.estimate_loss(100_000, 72_000, detector_efficiency=0.9, passes=3)
comparison = opt.compare_to_ideal(measured_coincidences, ideal_coincidences)

# 측정 지점 또는 round-trip time-bin별 count로 가장 큰 손실 구간 찾기
profile = opt.locate_loss({
    "source": 100_000,
    "after_switch": 93_000,
    "after_loop": 61_000,
    "detector_input": 58_000,
})
print(profile.dominant_segment, profile.segments)

calibration = opt.auto_calibrate(
    delays=delays,
    coincidences=measured_coincidences,
    ideal_coincidences=ideal_coincidences,
    input_count=100_000,
    output_count=72_000,
    detector_efficiency=0.9,
    passes=3,
)
print(opt.calibration_report(calibration))
```

## 6. 외부 모델 가져오기

```python
# Perceval
circuit = opt.from_perceval(perceval_circuit)

# Strawberry Fields passive Program
circuit = opt.from_strawberry_fields(sf_program)

# GDSFactory/SAX style S-parameter dictionary
circuit = opt.from_sparameters({
    ("o1", "o1"): 0.7,
    ("o2", "o1"): 0.7j,
    ("o1", "o2"): 0.7j,
    ("o2", "o2"): 0.7,
}, ports=["o1", "o2"])
```

어댑터는 외부 패키지를 import-time 필수 의존성으로 만들지 않습니다. 직접 생성된 객체나 S-parameter를 받는 duck-typed 경계이므로 OpenPhotonTwin 자체는 독립적으로 설치됩니다.

## 모델 범위

- `FockSimulator`: 수동 선형 네트워크의 coherent 다광자 확률. photon 수가 커지면 permanent 계산이 지수적으로 증가합니다.
- `DigitalTwin`: 하드웨어 timing/noise 및 time-tag 생성용 event Monte Carlo. 여러 경로의 coherent 재결합은 Fock/HOM 계층에서 계산합니다.
- `FiberLoop`: round-trip별 독립 손실·Gaussian phase drift·제어된 out-coupling 모델입니다.
- 손실의 절대 위치는 input/output 한 쌍만으로 식별할 수 없습니다. `locate_loss`는 물리적 tap 또는 round-trip time-bin처럼 순서가 알려진 두 개 이상의 측정 지점이 필요합니다.
- HOM Gaussian fitting은 transform-limited Gaussian wavepacket 가정입니다.

모든 시간은 초, 길이는 미터, 주파수는 SI 단위를 사용합니다. 난수 기반 API는 `seed`를 받아 재현 가능합니다.

## 테스트

```bash
python -m pytest
python -m ruff check .
```

실행 가능한 전체 예시는 [`examples`](examples) 폴더에 있습니다.
