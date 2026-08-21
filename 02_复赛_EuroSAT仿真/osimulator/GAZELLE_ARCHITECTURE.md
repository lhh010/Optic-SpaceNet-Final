# Gazelle Photonic GEMM Simulator — Reverse Engineering Report

> **Image**: `lightelligence.docker:lt-simulator_v1.4.6` (12.7 GB, ~3.79 GB compressed)
> **Date**: 2026-07-09
> **Repository**: `https://gitlab.com/lightelligence/lt-simulator` (inferred)

---

## 1. Environment

| Item | Detail |
|---|---|
| Base OS | Ubuntu 20.04.6 LTS (Focal Fossa) |
| System Python | 3.9.5 |
| GCC | 9.4.0 |
| Conda Env | `moca_llm` (Python 3.9 + PyTorch 2.1.0+cu121 + Triton 3.0.0) |
| GPU Support | SM80 (A100) → SM86 (30-series) → SM89 (40-series/L-cards) → SM90 (H100) |
| DRM | BitAnswer LicenseManager (`lt-simulator__LicenseManager64`) |
| License SN | `B55V2OCAK46DUQ6N` (activated in running container) |

### Key Dependencies (conda env `moca_llm`)

```
osimulator==1.3.4
torch==2.1.0+cu121
triton==3.0.0
Cython==3.0.0
numpy==1.22.4
```

---

## 2. Package Inventory

### Container Filesystem

```
/workspace/
├── bitanswer/
│   └── lt-simulator__LicenseManager64    # BitAnswer DRM license manager (ELF64, stripped)
└── user_guide/
    └── example_load_gazelle_model.py     # Example: GEMM sim with uint4 input

/local/miniconda/envs/moca_llm/lib/python3.9/site-packages/
├── osimulator/                           # Python API package
│   ├── __init__.py                       # from .api import *
│   ├── api.py                            # Public API: load_gazelle_model(), load_approach_model()
│   ├── datasets/                         # Test vectors (.npy)
│   │   ├── input_vector_10000_{8,16,32}_uint4.npy
│   │   └── weight_matrix_10000_{8,16,32}_int4.npy
│   ├── models/
│   │   ├── gazelle/
│   │   │   └── 8X2_8a8w12o_dacenob7.5_power0.015_noise9e-11_10mclock_mean-1.49_std5.31/
│   │   │       ├── compass_with_lut_seed_None.inst      # 421 KB — LUT calibration table
│   │   │       ├── mapping_sum0.inst                     # 4.0 MB — per-tile mapping (tile 0)
│   │   │       ├── mapping_sum1.inst                     # 4.0 MB — per-tile mapping (tile 1)
│   │   │       ├── gain.inst                             # [1000, 1000] float32
│   │   │       ├── offset.inst                           # [2048, 2048] float32
│   │   │       ├── offset_error.inst                     # [0.00119, -0.149] float32
│   │   │       ├── noise_mse.inst                        # [2.85e-7, 2.85e-7] float32
│   │   │       ├── real_lsb.inst                         # [0.00147, 0.00147] float32
│   │   │       ├── scaling_factor.inst                   # [406.47, 406.47] float64
│   │   │       ├── input_tensors.inst                    # 435 B
│   │   │       ├── wght_tensors.inst                     # 467 B
│   │   │       └── results_model_instance_pytorch_ap_separated.inst  # 665 B
│   │   ├── pace2/   (models present but not analyzed in detail)
│   │   └── pace3/   (models present but not analyzed in detail)
│   └── __pycache__/                      # Bytecode-only (11 modules, source .py deleted)
│       ├── ApprxPICPyTorch.cpython-39.pyc
│       ├── ApprxPICPyTorchCuda.cpython-39.pyc
│       ├── PostProcess.cpython-39.pyc
│       ├── PreProcess.cpython-39.pyc
│       ├── device_info.cpython-39.pyc
│       ├── GPUMemoryTracker.cpython-39.pyc
│       ├── instantiate.cpython-39.pyc
│       ├── optical_matmul.cpython-39.pyc
│       ├── optical_mm.cpython-39.pyc
│       ├── optical_mm_turbo.cpython-39.pyc
│       └── rms.cpython-39.pyc
├── entrance.cpython-39-x86_64-linux-gnu.so    # Core runtime (6.8 MB, stripped)
└── osimulator-1.3.4.dist-info/                # Package metadata

/usr/lib/
└── lib00003C3A_00000311_shell_ext_x64.so      # BitAnswer DRM wrapper (1.3 MB, not stripped)
```

### Debug Symbol Audit

| Artifact | Debug Symbols | Source Code | Recoverable |
|---|---|---|---|
| `entrance.so` (6.8 MB) | ❌ stripped | ❌ | 6 dynsym exports only |
| `lib00003C3A_...so` (1.3 MB) | △ not stripped (DRM wrapper) | ❌ | All exports are `Bit_*` licensing functions |
| `kernel_*.cubin` (5 files) | △ `.debug_frame` only | ❌ | `.nv.info` has mangled kernel names |
| `LicenseManager` | ❌ stripped | ❌ | 1 source path found |
| Python `.pyc` (11 files) | ⚠️ line numbers preserved | ❌ `.py` deleted | ✅ Decompilable via pycdc |
| `.inst` model files | N/A | N/A | ✅ Unencrypted numpy arrays |

---

## 3. Gazelle Architecture

### 3.1 Model Path Decoding

```
8X2_8a8w12o_dacenob7.5_power0.015_noise9e-11_10mclock_mean-1.49_std5.31
│  │ │  │  │         │           │            │          │
│  │ │  │  │         │           │            │          └─ ADC output statistics
│  │ │  │  │         │           │            └─ Clock frequency: 10 MHz
│  │ │  │  │         │           └─ Noise floor: 9×10⁻¹¹
│  │ │  │  │         └─ Laser power: 0.015
│  │ │  │  └─ DAC ENOB: 7.5 bits
│  │ │  └─ Output precision: 12 bits
│  │ └─ Weight precision: 8 bits
│  └─ Activation precision: 8 bits
└─ Physical tile: k=8 × n=2
```

### 3.2 Physical Parameters

| Parameter | Value | Description |
|---|---|---|
| `physical_k_dimension` | 8 | Input channels per tile |
| `physical_n_dimension` | 2 | Output channels per tile |
| `input_precision` | 8 bits | Model's native input precision |
| `weight_precision` | 8 bits | Model's native weight precision |
| `output_precision` | 12 bits | Model's native output precision |
| `repeat_n` | n_logical / 2 | N-dimension expansion factor |
| `repeat_k` | 1 | K-dimension not expanded |

### 3.3 Data Flow Pipeline

```
int4/uint4 input[k=1..8]
        │
        ▼
   ┌─────────┐
   │   DAC   │  ENOB=7.5, offset_error[2]
   └────┬────┘
        │
        ▼
   ┌──────────┐
   │ Modulator │  vecmod/weightmod nonlinearity
   └────┬─────┘
        │
        ▼
   ┌──────────────────┐
   │ Photonic GEMM     │  LUT-calibrated (compass_with_lut)
   │ (compute tile)    │  mapping_sum[0,1]
   └────┬─────────────┘
        │
        ▼
   ┌─────────┐
   │   TIA   │  gain[2]=[1000, 1000], noise_mse[2]=2.85e-7
   └────┬────┘
        │
        ▼
   ┌─────────┐
   │   ADC   │  real_lsb[2]≈0.00147, offset[2]=2048, FSR
   └────┬────┘
        │
        ▼
   int12 output[n=1..2]
```

### 3.4 Calibration Parameters

```json
{
  "gain":             [1000.0, 1000.0],           // TIA gain per n-channel
  "offset":           [2048.0, 2048.0],           // ADC DC offset
  "offset_error":     [0.001186, -0.148993],      // DAC offset error per n-channel
  "noise_mse":        [2.846e-7, 2.846e-7],       // TIA noise MSE per n-channel
  "real_lsb":         [0.001465, 0.001465],       // ADC LSB step size
  "scaling_factor":   [406.475, 406.475],         // System scaling factor (float64)
  "mapping_factor_uint4bit": 272,
  "mapping_factor_int4bit":  256,
  "mapping_factor_uint2bit": 5440,
  "mapping_factor_int2bit":  4096,
  "mapping_factor_uint8bit": 1,
  "mapping_factor_int8bit":  1
}
```

---

## 4. LUT (Look-Up Table) Structure

### 4.1 Tensor Shapes

The simulator maintains 6 LUT tables in GPU/CPU memory, one per (signedness, bitwidth) combination:

| LUT Table | Shape | Size | Combo Count (last dim) |
|---|---|---|---|
| `lut_sum_uint2_` | [2, 2, 1, 8, 2, 16] | 4 KB | 16 = 2^(2×2) |
| `lut_sum_int2_` | [2, 2, 1, 8, 2, 16] | 4 KB | 16 = 2^(2×2) |
| `lut_sum_uint4_` | [2, 2, 1, 8, 2, 256] | 64 KB | 256 = 2^(4+4) |
| `lut_sum_int4_` | [2, 2, 1, 8, 2, 256] | 64 KB | 256 = 2^(4+4) |
| `lut_sum_uint8_` | [2, 2, 1, 8, 2, 65536] | 16 MB | 65536 = 2^(8+8) |
| `lut_sum_int8_` | [2, 2, 1, 8, 2, 65536] | 16 MB | 65536 = 2^(8+8) |

### 4.2 Dimension Semantics

```
[2,   2,   1,  8,   2,   N]
 │    │    │   │    │    │
 │    │    │   │    │    └─ Combo index: all (input, weight) pairs per k-element
 │    │    │   │    │         N = 2^(in_bitwidth + wght_bitwidth)
 │    │    │   │    │         Index: (input & mask) << wght_bitwidth | (weight & mask)
 │    │    │   │    └─ Output channel (n=2)
 │    │    │   └─ Input channel (k=8)
 │    │    └─ Batch/broadcast dimension
 │    └─ Tile sub-index 1
 └─ Tile sub-index 0
```

### 4.3 Combo Index Encoding

For uint4 (4-bit unsigned input, 4-bit signed weight):

```python
combo = (input_value & 0xF) * 16 + (weight_value & 0xF)
# input_value ∈ [0, 15], weight_value ∈ [-8, 7] (stored as 2's complement)
# 256 total combos covering all possible (activation, weight) pairs
```

### 4.4 LUT Value Range

```
Typical LUT values: ~2×10⁻⁶ to ~1.6×10⁻⁴ (raw analog units)
The LUT is NOT a direct output value — it encodes per-element analog contributions
that are combined through a non-trivial signal chain.
```

---

## 5. entrance.so API Surface

### 5.1 Exported Symbols (6 total)

```
PyInit_entrance                              Python module init
_Z10load_modelP7_objecti                    load_model(PyObject* path, int)
_Z17cosine_similarityP7_objectS0_i          cosine_similarity(PyObject*, PyObject*, int)
_Z25matmul_with_analog_inputsP7_objectS0_S0_S0_S0_i
    → matmul_with_analog_inputs(input, weight, model, config, ?, int)
_Z26matmul_with_mapping_factorP7_objectS0_S0_ddS0_S0_i
    → matmul_with_mapping_factor(input, weight, model, double, double, ?, ?, int)
_Z8entranceiPPc                              entrance(int argc, char** argv)
```

### 5.2 Dynamic Library Dependencies

```
NEEDED  lib00003C3A_00000311_shell_ext_x64.so   # BitAnswer DRM wrapper
NEEDED  libm.so.6
NEEDED  libgcc_s.so.1
NEEDED  libc.so.6
```

### 5.3 DRM Wrapper Functions (from `lib00003C3A_...so`)

The DRM library provides BitAnswer licensing functions including:
`Bit_CheckOut`, `Bit_CheckIn`, `Bit_DecryptFeature`, `Bit_DevLogin`, `Bit_DevLogout`, etc.
The entrance.so functionality depends on successful DRM activation.

### 5.4 Key Python-accessible Functions

| Function | Purpose |
|---|---|
| `entrance.load_model(path)` | Load `.inst` model instance file → C opaque handle |
| `entrance.load_instance_pickle(path, None)` | Load individual `.inst` numpy array |
| `entrance.gazelle_latency(b, m, k, n)` | Latency estimation (~16.6 µs per operation) |
| `entrance.stats(result, expected, scaling, in_bit, out_bit, ...)` | Compute accuracy statistics |
| `entrance.dump_stats(result, expected, in_bit, out_bit)` | Dump comparison stats |

### 5.5 CUDA Kernels

Single kernel: `calculate_lut` (mangled: `_Z14calculate_lutvPKiS0_S0_PKfS2_PfS0_...`)

```
Arguments: lut_output, input_data, weight_data, mapping_data, gain_table, noise_table, result, ...
Architectures: SM80 (A100), SM86 (30-series), SM89 (40-series), SM90 (H100)
Each cubin has .debug_frame but no source-level DWARF (.debug_info, .debug_line)
```

---

## 6. Python Source Architecture

### 6.1 Module Dependency Graph

```
api.py  (public API)
 ├── entrance (C extension)
 ├── osimulator.PostProcess (Triton dequant kernel)
 └── osimulator.ApprxPICPyTorchCuda (GPU path)

ApprxPICPyTorch.py  (CPU PyTorch model)
 ├── entrance.load_instance_pickle()
 ├── osimulator.Preprocess2.pre_process2()
 └── torch.nn.Module

ApprxPICPyTorchCuda.py  (GPU PyTorch model)
 ├── entrance
 ├── omatmul (CUDA matmul module)
 └── torch.nn.Module

optical_matmul.py / optical_mm.py  (GEMM simulators)
 ├── entrance.load_model()
 ├── osimulator.PreProcess.inject_variables_into_adc()
 └── device_info.compass_forward_hook

instantiate.py  (CLI tool)
 ├── argparse
 ├── entrance
 └── osimulator

PostProcess.py  (Triton dequantization)
 ├── triton.language
 └── dequant_kernel (bit-dim sum reduction)

PreProcess.py  (ADC injection)
 └── entrance
```

### 6.2 Decompilation Status

All 11 bytecode-only modules decompiled via `pycdc`. Quality varies:

| Module | Lines | Quality | Notes |
|---|---|---|---|
| `optical_matmul.py` | 217 | ✅ Clean | Full `oMAC_Matmul` class |
| `ApprxPICPyTorch.py` | 282 | ✅ Clean | Calibration loading, tiling logic |
| `ApprxPICPyTorchCuda.py` | 261 | ✅ Clean | CUDA-accelerated variant |
| `optical_mm.py` | 209 | ✅ Clean | Similar to optical_matmul |
| `optical_mm_turbo.py` | 44 | ✅ Clean | pace3 architecture wrapper |
| `instantiate.py` | 426 | △ 1 opcode | Full CLI with argparse |
| `api.py` | 245 | △ 2 opcodes | GPU device mapping, load_approach_model |
| `PostProcess.py` | 60 | △ 1 opcode | Triton dequant kernel visible |
| `device_info.py` | 32 | △ 1 node | Singleton, latency hook |
| `GPUMemoryTracker.py` | 51 | ❌ Incomplete | PyTorch profiler wrappers |
| `PreProcess.py` | 12 | ❌ Truncated | `inject_variables_into_adc()` head only |
| `rms.py` | 40 | ❌ Incomplete | Metric base class |

---

## 7. Simulator Behavioral Characterization

### 7.1 Key Finding: Approximately Linear

The photonic GEMM behavior is **approximately linear and additive** with small per-element non-idealities.

### 7.2 Per-Element Linearity

Single k-element (k=0, n=0) with weight=1, sweeping input 0..15:

```
in:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
sim: 0  1  3  3  4  4  4  7  9  7 10 12 10 13 13 12
Δ:   0  0 +1  0  0 -1 -2  0 +1 -2  0 +1 -2  0 -1 -3
```

Non-linearity: ±0~3 LSB. Approximate monotonicity with quantization noise.

### 7.3 K-Element Independence

Two-element additivity test — `f([a,0,...]) + f([0,b,...])` vs `f([a,b,...])`:

```
  a=5,  b=5:  independent=12, combined=9,  Δ=-3
  a=5,  b=10: independent=16, combined=15, Δ=-1
  a=10, b=5:  independent=13, combined=15, Δ=+2
  a=10, b=10: independent=19, combined=22, Δ=+3
```

Coupling deviation: ±1~3 LSB. Elements are largely independent.

### 7.4 Full-Scale Linearity

All 8 k-elements identical (weight=1, sweep scale 1..15):

```
scale:   1    2    4    8   12   15
ideal:   8   16   32   64   96  120
sim:     9   16   31   64   99  123
ratio: 1.13 1.00 0.97 1.00 1.03 1.03
```

No significant saturation across the full dynamic range.

### 7.5 N-Channel Crosstalk

Signal on n=0 only, measuring n=1 leakage:

```
n1 leakage: ±0~1 LSB (effectively zero crosstalk)
```

### 7.6 Ideal GEMM Approximation Accuracy

On 2000 random (input[8], weight[8,2]) pairs:

```
Residual (sim - ideal):
  Mean:  -2.44  (systematic bias toward slightly lower values)
  Std:    4.72
  Range: [-18, 14]

MAE = 4.18 LSB
Relative error = 0.6% (output range ~[-423, 302])
```

### 7.7 Supported Architectures

Three hardware architectures referenced in the codebase:

| Architecture | Model Path Pattern | Tile Config | Notes |
|---|---|---|---|
| **Gazelle** | `gazelle/8X2_8a8w12o_...` | k=8×n=2 | Primary target, most complete |
| **Pace2** | `pace2/8X8_4a4w8o_...` | k=8×n=8 | Present in code, models exist |
| **Pace3** | `pace3/32X16X4_4a4w10o_...` | k=32×n=16×? | Turbo variant, larger tile |

---

## 8. Python API

### 8.1 Loading Models

```python
from osimulator.api import load_gazelle_model

# Load default Gazelle model (CPU)
model = load_gazelle_model()
# Returns: entrance.ApprxPICPyTorch (CPU) or entrance.ApprxPICPyTorchCuda (GPU)

from osimulator.api import load_approach_model

# Load with explicit device and path
model = load_approach_model(
    device_type="sm80",      # GPU arch: sm80/sm86/sm89/sm90 or "cpu"
    instances_path="...",    # Path to model instance directory
    gpu_devices=[0,1],       # GPU device IDs
    k_dimension=0,           # 0 = use built-in k size
    n_dimension=0,           # 0 = use built-in n size
)
```

### 8.2 Running Inference

```python
import numpy as np

# Input: (batch, m, k), uint4 range [0, 15] or int4 range [-8, 7]
# Weight: (batch, k, n), int4 range [-8, 7]
input_t = np.random.randint(0, 15, (1, 1, 8), dtype=np.int32)
wght_t = np.random.randint(-8, 7, (1, 8, 2), dtype=np.int32)

# Run GEMM
result = model(input_t, wght_t, inputType="uint4")
# Returns: torch.Tensor shape (1, 1, 2), dtype=torch.int32
```

### 8.3 GPU Device Type Mapping

```python
gpu_device_type_dicts = {
    'sm80': ['A100', 'A800'],
    'sm86': ['3080', '3090', 'A2000', 'A3000', 'A4000', 'A5000', 'A6000', 'A40',
             '3060', '3070', '3050', 'A10', 'A16'],
    'sm89': ['4090', '4080', '6000', 'L20', 'L40', 'L4'],
    'sm90': ['H100', 'H200', 'H20'],
}
```

### 8.4 Post-Processing (Triton Dequant)

```python
# From PostProcess.py decompilation:
# dequant_kernel: bit-dimension sum reduction with per-bit scale factors
# result = Σ_b (data_a[bit_slice] * scale[bit_slice])
# Output: float16
```

### 8.5 Latency Model

```python
import entrance
latency = entrance.gazelle_latency(batch=1, m=1, k=8, n=2)
# Returns: ~1.66×10⁻⁵ seconds (16.6 µs per GEMM operation)
```

---

## 9. Model Instantiation (CLI)

The `instantiate.py` module provides CLI tooling for creating custom model instances:

```bash
python instantiate.py \
    --model_type gazelle|pace2|pace3 \
    --k_dimension 8 \
    --n_dimension 2 \
    --input_precision 4 \
    --weight_precision 4 \
    --output_precision 8 \
    --laser_power 0.015 \
    --clock_frequency 10e6 \
    --acc_tia_gain 1000 \
    --acc_tia_noise 2.85e-7 \
    --acc_adc_fsr ... \
    --modulator_dac_enob 7.5 \
    --tx_tia_gain ... \
    --dst_instance_path /path/to/output
```

Configurable parameters include:
- `laser_power`, `clock_frequency` — system-level
- `acc_tia_gain`, `acc_tia_noise` — accumulator TIA
- `acc_adc_fsr` — accumulator ADC full-scale range
- `modulator_dac_enob` — modulator DAC effective bits
- `tx_tia_gain` — Tx monitor TIA
- `vecmod_dac3dac4` — vector modulator DAC configuration

---

## 10. Artifact Inventory

### 10.1 Local Files

```
osimulator/
├── GAZELLE_ARCHITECTURE.md              # This document
├── gazelle_artifacts/                    # Dumped data from running simulator
│   ├── calibration_params.json          # Model calibration (gain, offset, etc.)
│   ├── behavioral_char.json             # Per-element, additivity, saturation, crosstalk data
│   ├── random_benchmark.json            # 2000 random (ideal, sim, delta) samples
│   ├── lut_sum_uint4_.npy               # Uint4 LUT [2,2,1,8,2,256] — 64 KB
│   ├── lut_sum_int4_.npy                # Int4 LUT [2,2,1,8,2,256] — 64 KB
│   ├── lut_sum_uint2_.npy               # Uint2 LUT [2,2,1,8,2,16] — 4 KB
│   ├── lut_sum_int2_.npy                # Int2 LUT [2,2,1,8,2,16] — 4 KB
│   ├── lut_sum_uint8_.npy               # Uint8 LUT [2,2,1,8,2,65536] — 16 MB
│   └── lut_sum_int8_.npy                # Int8 LUT [2,2,1,8,2,65536] — 16 MB
├── decompiled/                          # Decompiled Python source (from .pyc)
│   ├── api.py
│   ├── ApprxPICPyTorch.py
│   ├── ApprxPICPyTorchCuda.py
│   ├── optical_matmul.py
│   ├── optical_mm.py
│   ├── optical_mm_turbo.py
│   ├── instantiate.py
│   ├── PostProcess.py
│   ├── PreProcess.py
│   ├── device_info.py
│   ├── GPUMemoryTracker.py
│   └── rms.py
├── osimulator/                          # Original package from container
│   ├── __init__.py, api.py             # Source .py files (only these 2 present)
│   ├── __pycache__/                     # Bytecode for all 13 modules
│   ├── datasets/                        # Test vector .npy files
│   ├── models/gazelle/.../              # Model .inst files
│   └── kernel_*.cubin                   # CUDA kernels (sm80/86/89/90)
├── entrance.cpython-39-x86_64-linux-gnu.so  # Core C extension (stripped)
└── osimulator-1.3.4.dist-info/          # Package metadata
```

### 10.2 Remote Container

```
Container: gazelle_sim (ID: 274915157eeb)
Image:     lightelligence.docker:lt-simulator_v1.4.6
Docker context: fdusc-cpu-135 (SSH)
License:   Activated (SN: B55V2OCAK46DUQ6N)
```

### 10.3 Running the Simulator

```bash
# Connect to remote Docker context
docker --context fdusc-cpu-135 exec gazelle_sim bash

# Inside container:
source /local/miniconda/etc/profile.d/conda.sh
conda activate moca_llm
python3 -c "
from osimulator.api import load_gazelle_model
model = load_gazelle_model()
# ...
"
```

---

## 11. Limitations & Unknowns

### What We Know

- Calibration parameters (gain, offset, noise, LSB, scaling factor)
- LUT structure and how to index it (combo encoding)
- Behavioral characteristics (linearity, additivity, saturation, crosstalk)
- Python API surface and module dependency graph
- entrance.so exported C API (6 functions)
- CUDA kernel name and approximate signature
- Model instantiation CLI parameters
- Non-ideality statistics (Δ mean=-2.44, std=4.72)

### What Remains Black-Box (in entrance.so)

- Exact LUT computation algorithm (how mapping_sum files are used)
- Per-component transfer functions (DAC nonlinearity curve, TIA frequency response)
- Noise injection mechanism (additive Gaussian? at which stage? correlated?)
- Photonic interference model (coherent? MZI mesh? wavelength-dependent?)
- Pipeline timing beyond exposed `gazelle_latency()`
- Internal function implementations (5.8 MB .text section, all stripped)

### Blocked by BitAnswer DRM

- Runtime code decryption/obfuscation in `lib00003C3A_00000311_shell_ext_x64.so`
- Binary reverse engineering of entrance.so internals is **not feasible** without DRM defeat
