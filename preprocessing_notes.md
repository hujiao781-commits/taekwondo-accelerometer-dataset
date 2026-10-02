# Preprocessing and Data-Quality Notes

This document describes how the released CSV files were derived from the collected sensor data, the quality-control checks that were applied, and the conventions used in the repository.

## 1. Data acquisition

- **Device:** five ActiGraph GT3X tri-axial accelerometers, configured at a sampling rate of **30 Hz** with a dynamic range of **+/- 6 g** per axis.
- **Placement:** one unit on each wrist, one at the **L5** lower-back level, and one on each ankle. Devices were oriented consistently and secured with straps.
- **Signals:** 15 acceleration channels (5 sensors x 3 axes), reported in units of **g** (the native ActiGraph output). The static gravity component is retained.
- **Protocol:** each participant performed a fixed set of standardized techniques; each valid execution was annotated with the action name, participant identifier, and trial number.

The 30 Hz sampling rate is supported by the spectral content of the movements: the mean normalized power spectra (see `results/figures/spectrum_overview.png`) show that signal energy is concentrated at low frequencies and decays well before the 15 Hz Nyquist frequency.

## 2. Segmentation into windows

- The continuous recordings were segmented around each annotated technique execution.
- Each released sample is a fixed **4-second** window containing exactly **120 time points** (120 / 30 Hz = 4 s), with a single isolated technique per window.
- Windows from the five sensors were aligned by their synchronized timestamps so that the same time index corresponds to the same instant across all 15 channels.
- Windows that were incomplete (fewer than 120 points), contained missing samples, or had corrupted/desynchronized channels were discarded during quality control.

## 3. File-format conversion and label standardization

The source files were produced in two different formats and have been normalized to standard UTF-8 CSV:

- The kicking `data`/`label` workbooks were in fact comma-separated text (with a UTF-8 BOM) saved with an `.xlsx` extension; they are read directly as CSV.
- The upper-limb (punch/block) `data`/`label` workbooks were genuine Excel (`.xlsx`) files; they were converted to CSV.
- In the source kicking files the action names were written in Chinese, whereas the upper-limb files already used English names. For consistency across the repository and with the class names used in the analysis, the kicking action names were translated to English using the fixed mapping below. No signal values were altered.

| Original (Chinese) | Released (English) |
| --- | --- |
| 左前踢 | Left Front Kick |
| 右前踢 | Right Front Kick |
| 左横踢 | Left Roundhouse Kick |
| 右横踢 | Right Roundhouse Kick |
| 左侧踢 | Left Side Kick |
| 右侧踢 | Right Side Kick |
| 左后踢 | Left Back Kick |
| 右后踢 | Right Back Kick |

The sample identifier follows the format `ActionName_ParticipantID_TrialNumber`; for example the kicking id `左前踢_1_0` is released as `Left Front Kick_1_0`. The conversion and all integrity checks are implemented in `code/preprocessing.py`.

## 4. Integrity and quality-control checks

For every released window the following checks were performed:

1. **Length:** exactly 120 time points per window.
2. **Completeness:** no missing (NaN) values and no corrupted channels.
3. **Synchronization:** consistent time indices across all five sensors.
4. **Label consistency:** the set of sample ids in each `data` file matches the set in the corresponding `label` file.
5. **Range check:** acceleration values were checked against the sensor's +/- 6 g linear range.

The vast majority of values lie within +/- 6 g. A small number of transient instantaneous readings slightly exceed +/- 6 g at the moment of impact (see `results/tables/channel_statistics.csv`); these arise from sensor non-linearity beyond the linear range, zero-bias drift, and transient mechanical vibration. Such peaks are retained in the released signals for transparency but are **not** used for quantitative peak-acceleration biomechanical analysis.

## 5. Processing deliberately not applied

To keep the released data minimally processed and reversible, the following were **not** applied to the released CSV files:

- No low-pass / high-pass / notch filtering and no smoothing.
- No coordinate-system rotation or sensor-fusion reorientation.
- No removal of the gravity component.
- No amplitude scaling, normalization, or clipping.

During the downstream analysis, z-score standardization is computed **within each training fold** only (never across the full dataset), and the deep-learning models apply only an internal per-channel scaling; these are analysis-time operations and are not baked into the released data.

## 6. Participant information

- Aggregate cohort demographics (sex counts and group-level mean +/- SD for age, height, weight, and BMI, plus training-background and belt/competition-level counts) are provided in `data/cohort_summary.csv`.
- Demographics are released at the group level; per-participant demographic records and limb dominance (handedness) information are not included.
- All records were anonymized, and the raw continuous recordings are not publicly released, to protect participant privacy.

## 7. Reproducibility

Run `code/preprocessing.py` to repeat the format conversion and integrity checks, and run `code/taekwondo_recognition.ipynb` top to bottom to reproduce the feature extraction, cross-validation, baseline models, sensor ablation, and all figures and tables under `results/`.
