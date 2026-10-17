# irgen: parametric impulse response generator (v0.2)

irgen is a hybrid IR synthesizer. It builds every IR from four layers. Each layer can come from a physical room model or be shaped directly with perceptual parameters, and you can mix the two:

| Layer | Physical (`"room"`) | Perceptual / manual |
|---|---|---|
| Direct sound | distance, source and mic directivity, air absorption | gain, tone |
| Early reflections | shoebox image-source model: absorption per surface and octave, scattering, directivity, air absorption | `parametric` generator or a hand-placed `list` |
| Late tail | statistical decay (Eyring/Sabine). The "diffuse rain" is the energy scattering removes from the reflections, put back into the tail | T60 curve, onset, attack, double slope, tone, DRR, echo-density growth |
| Low-frequency modes | exact rectangular-room modal synthesis below the Schroeder frequency (small rooms) | — |

## GUI

```bash
pip install -r requirements.txt
python -m irgen gui          # opens http://127.0.0.1:8765 in your browser
```

The GUI is a local web page served by Python's standard library, so there are no extra dependencies. It has:

- **Controls for every layer:** output settings; the room, with draggable source and mic in a plan view and materials per surface; direct sound; early reflections; late tail; multichannel; and EQ.
- **Nine-band octave sliders** for T60 and tail tone.
- **Physical / Hybrid / Perceptual switch** for where the early and late layers come from.
- **Results after Render (or ⌘/Ctrl+Enter):**
  - envelope plot and key metrics
  - IR preview (channels 1/2 as stereo when multichannel)
  - full analysis report, plus the multichannel report with pass/fail badges against the JNDs
- **Audition:** convolve the IR with drums, plucks, a voice-like signal, clicks, or your own audio file. Pick any two channels as L/R to hear the decorrelation.
- **Preset JSON tab:** edit keys that have no control. Load and save presets in `presets/`.

GUI renders are written to `renders/gui/<name>/`.

## Command line

```bash
python -m irgen render presets/concert_hall.json -o renders
python -m irgen render presets/*.json -o renders                      # batch
python -m irgen render presets/concert_hall.json --set late.t60_scale=1.3 --set seed=4 --name hall_long
python -m irgen analyze my_measured_ir.wav --plot my_ir.png             # analyse any IR
python -m irgen materials                                              # absorption database
```

Each render writes three files:

- `name.wav`: the IR.
- `name.json`: the fully resolved preset, derived room data, and the measured metrics.
- `name_report.png`: early part, envelope, spectrogram, energy decay curves, T vs target, magnitude response, echo density, and a table of ISO 3382 metrics (EDT, T20, T30, C50, C80, D50, Ts, DRR, BR, Br).

From Python:

```python
from irgen import generate, analyze
ir, fs, info, cfg = generate("presets/perceptual_hall.json")   # or a dict
```

## Multichannel: 1–64 decorrelated, perceptually similar IRs

Set `multichannel.channels` to 2–64. Each channel gets its own random late tail. In every filterbank band, the tails are mixed with the symmetric square root of a target coherence matrix, so their pairwise correlation matches the target per frequency. All channels then share one energy envelope: the same T60, level, tone and density build-up. That gives statistically decorrelated but perceptually identical tails.

| key | options |
|---|---|
| `coherence.model` | `independent` (0); `constant` (`value`: a number or frequency curve); `diffuse` (sin(kd)/kd from the mic spacing); `diffuse_2d` (J0(kd)); `custom` (`matrix`) |
| `coherence.coherent_blend` | 0..1 curve that pushes towards full coherence, e.g. `{"freqs":[60,160],"values":[1,0]}` for mono bass |
| `array.type` | `none`, `line`, `circle`, `sphere`, `grid`, `random`, `custom` (with `spacing_m`, `radius_m`, `axis`, `orientation`: `same`/`outward`, `positions`) |
| `early` | `same` (identical early part); `jitter` (room: each mic at its own position, from the array or from `receiver_jitter_m`; parametric: per-channel `time_jitter_ms` / `gain_jitter_db`); `independent` (new random pattern per channel) |
| `direct` | `auto` (per channel if an array is set), `same`, `per_channel` |
| `export` | `interleaved` (one N-channel WAV), `separate` (one file per channel), `both` |

Every multichannel render also writes `name_multichannel_report.png` and a `multichannel_analysis` block in the JSON:

- **Decorrelation:**
  - pairwise late-part correlation matrix
  - octave-band coherence, measured vs target
  - max cross-correlation of the full IR within ±1 ms
- **Similarity:** per-channel T30, EDT, C80, level and 1/3-octave spectrum, checked against the JNDs:

  | parameter | JND |
  |---|---|
  | T30 / EDT | 5 % |
  | C80 | 1 dB |
  | level | 1 dB |
  | spectrum | 1 dB rms |

  T30, EDT and C80 use the ISO 3382 500 Hz–1 kHz averages; the worst single band is reported too.

`late.envelope_flatten` (on by default) removes slow random level fluctuations in each noise band. This is what keeps decay parameters within the JND from channel to channel, without making the channels correlated.

Things to know:

- **The direct sound and early reflections dominate full-IR correlation.** With `early: "same"`, the full IRs stay correlated (~0.5) even when the tails are independent. Use `jitter` or `independent`, or a real array with `direct: per_channel`, for lower values.
- **Low frequencies vary more between channels, and that is inherent.** Below ~150 Hz an 80 ms window holds only a few cycles, so C80 and EDT differ more per channel. `coherent_blend` for the bass reduces this and is also what real rooms do.
- **Cost grows with channel count.** 64 channels × 4 s at 48 kHz takes roughly 0.5–1.5 min and a few hundred MB of RAM.

## Quality notes

- **Each reflection is filtered individually.** Every reflection gets its own minimum-phase filter, built from its absorption, scattering loss, air absorption and tone, plus a band-limited fractional delay (Kaiser-windowed sinc). Arrival times are sub-sample accurate and there is no pre-ringing.
- **Late tail.** Noise goes through a perfect-reconstruction 1/3-octave filterbank (cos² masks that sum exactly to 1). Every band decays with its own T60, interpolated continuously over log-frequency. Air absorption is part of the T60.
- **T60 calibration.** `late.calibrate` (default 3 iterations) measures the octave-band T30 of the finished IR and corrects the internal decay until it matches the target. The report plots target against measured.
- **Physical consistency.** The tail level is the theoretical diffuse energy (4πc/V·e^(−13.8t/T)) on the same scale as the image sources (direct sound = 1/r). Image-model energy and tail energy agree to within about 1 dB, and the crossover uses complementary windows. Modal synthesis uses the same scale. Below the crossover it is joined with a Linkwitz-Riley split for the early (correlated) part and a power-complementary Butterworth split for the late (uncorrelated) part.
- **Realism details:**
  - Randomized image positions (`image_jitter_m`) prevent sweeping or flutter echoes in the shoebox.
  - Scattered energy is smeared causally into the tail.
  - Echo density can grow from sparse to dense (velvet noise, density ∝ t²) and crossfade into Gaussian noise.
  - Export: 64-bit internal processing, sample rates 44.1–192 kHz, PCM 16/24/32 or float, TPDF dither.

## Preset reference

A preset is JSON. Every key is optional and falls back to the defaults in `irgen/config.py`. Keys starting with `_` are comments. `"mode": "perceptual"` switches the early and late defaults to the parametric sources, so no room is needed.

### Frequency curves

Anywhere a T60, `tone` or absorption curve is expected, you can write any of these:

- `2.1`: a single value for all frequencies.
- `[v63, v125, v250, v500, v1k, v2k, v4k, v8k, v16k]`: nine octave values.
- `{"freqs": [100, 1000, 8000], "values": [1.0, 0.0, -6.0]}`: free breakpoints, log-interpolated.
- `{"mid": 2.0, "low_ratio": 1.3, "high_ratio": 0.6, "low_hz": 250, "high_hz": 2500}`: classic bass/treble ratio form (T60 only).
- A material name, e.g. `"wood_panel"`, or `{"material": "carpet_thick", "scale": 0.8}` (absorption only).

### `output`

| key | default | meaning |
|---|---|---|
| `sample_rate` | 48000 | any rate; the filters adapt |
| `length_s` | `"auto"` | auto = until the tail is `trim_db` down (capped at `max_length_s`) |
| `trim_db` | 90 | |
| `lead_ms` | 0.7 | silence before the direct sound |
| `align_to_direct` | true | room mode: remove the propagation delay before the direct sound |
| `fade_out_ms` | 30 | |
| `dc_highpass_hz` | 8 | |
| `normalize` | `"peak"` | `"peak"` (to `peak_dbfs`), `"energy"` (unit energy) or `"none"` (physical scale) |
| `format` | `"PCM_24"` | `PCM_16`, `PCM_24`, `PCM_32`, `FLOAT`, `DOUBLE` |
| `dither` | true | TPDF dither for PCM formats |

### `room` (physical model)

| key | meaning |
|---|---|
| `dimensions` | `[Lx, Ly, Lz]` in metres |
| `materials` | `{"default": ..., "west"/"east"/"south"/"north"/"floor"/"ceiling": ...}`. West/east are x=0 / x=Lx, south/north are y=0 / y=Ly. Values are a material name or a curve |
| `scattering` | number, nine octave values, or a per-surface dict as above |
| `source` / `receiver` | `position`; `directivity` (`omni`, `subcardioid`, `cardioid`, `supercardioid`, `hypercardioid`, `figure8`, or a number 0..1); `orientation` `[azimuth°, elevation°]`, azimuth 0 = +x |
| `air` | `temperature_c`, `humidity_pct`, `pressure_kpa` (ISO 9613-1; also sets the speed of sound) |
| `reverb_formula` | `eyring` or `sabine` |
| `max_order` | `"auto"` or an integer |
| `image_jitter_m` | 0.04; image-position randomization |
| `modal` | `enabled` (`auto`/true/false), `crossover_hz` (auto = Schroeder frequency), `detune_cents`, `gain_db` |

### `direct`

`enabled`, `gain_db`, `tone`. Turn the direct sound off for wet-only IRs.

### `early`

- `source`: `room`, `parametric`, `list` or `none`.
- `gain_db`, `tone`.
- `crossover_ms` / `crossover_width_ms`: where the image model hands over to the statistical tail. Default: the perceptual mixing time, 20·V/S + 12 ms.
- `parametric` (all times relative to the direct sound):

| key | meaning |
|---|---|
| `count` | number of reflections |
| `start_ms`, `end_ms` | reflection window |
| `distribution` | `power` or `regular` |
| `density_exponent` | 0 = uniform, 2 = t² growth (Polack) |
| `level_start_db`, `level_end_db` | level (dB re direct) at the window start and end |
| `level_curve` | shape of the level change between them |
| `level_jitter_db` | random level variation per reflection |
| `polarity` | `random`, `positive` or `negative` |
| `color_from_decay` | darken later reflections in line with the T60 curve |
| `hf_damping_db_per_oct` | extra high-frequency damping |
| `absorption`, `bounces_end` | color reflections like N bounces off a material |
| `cluster_size`, `cluster_spread_ms` | split each reflection into a short diffuse cluster |
| `air` | apply air absorption |

- `list`: `[{"time_ms": 7.3, "gain_db": -4, "polarity": -1, "tone": <curve>}, ...]`.

### `late`

| key | meaning |
|---|---|
| `source` | `room`, `parametric` or `none` |
| `t60` | override curve; in room mode it replaces Eyring/Sabine |
| `t60_scale` | multiplies the whole T60 curve |
| `calibrate` | calibration iterations; 0 = off |
| `gain_db`, `tone` | level trim and spectral shape of the tail |
| `drr_db` | direct-to-late energy ratio. Parametric default 0 dB; optional override in room mode |
| `onset_ms`, `attack_ms`, `attack_shape` | parametric tail start, fade-in time and bloom shape |
| `noise` | `gaussian`, `velvet` or `auto` |
| `density` | `initial_per_s`, `max_per_s` (auto = fs/4), `full_ms` (time to reach full density; auto = mixing time), `exponent`, `pulse_amplitude` (`gaussian`/`uniform`) |
| `double_slope` | `enabled`, `t60` or `t60_ratio`, `level_db` (second slope's level at onset, re the first) |
| `bands_per_octave` | filterbank resolution, default 3 |

### `eq` / `eq_late`

Lists of causal filters, applied to the whole IR (`eq`) or the tail only (`eq_late`):

```json
{"type": "peak", "freq": 3000, "gain_db": -4, "q": 1.2}
```

Types: `peak`, `lowshelf`, `highshelf`, `lowpass`, `highpass` (`order` > 2 = Butterworth), `bandpass`, `notch`.

## Included presets

| preset | what it shows |
|---|---|
| `small_studio_room` | physical model; treated room; modal low end turns on automatically |
| `concert_hall` | physical model, ~16,800 m³ shoebox; 96 kHz float export |
| `stone_church` | large reverberant space; cardioid mic facing the source |
| `hybrid_room_custom_decay` | room early reflections + overridden T60 + double-slope tail |
| `perceptual_hall` | no geometry; every parameter is a target |
| `plate_style` | no early reflections, instant density, bright |
| `tapped_reflections` | hand-placed reflections + parametric tail |
| `multichannel_hall_64` | 64 independent, perceptually matched hall IRs; partly coherent bass; interleaved + separate export |
| `spaced_array_room_8` | physical room recorded with an 8-mic circle array (diffuse-field coherence, per-mic direct and early sound) |

## Known limits

- **Geometry is a shoebox only.** Arbitrary polyhedra and ray tracing are a v2 path.
- **The modal layer assumes a rectangular room.**
- **The top two octaves can measure higher than target.** With very steep high-frequency T60 drops (≥ 8 kHz), the octave T30 reads above the per-octave target. Slower-decaying energy at the lower edge of the band dominates. A real measurement of the same room would read the same.
- **Material data are typical published values.** The 63 Hz and 8/16 kHz values are extrapolated.
- **Coherence models assume omnidirectional mics.** Directional patterns change the diffuse-field coherence, and the multichannel models do not account for that.
