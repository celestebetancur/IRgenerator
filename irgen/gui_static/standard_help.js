/* Help content for the standard view: one source for the help-mode overlay
   (t = title, s = short) and the docs page (b = basics, d = in depth). HTML allowed. */
"use strict";
window.IRGEN_HELP = (() => {
const S = [];
const P = {};
function sec(id, title, b, d, params){ S.push({id, title, b, d, params: Object.keys(params)}); Object.assign(P, params); P["section." + id] = {t: title, s: b.split(". ")[0] + ".", b, d, section: true}; }

sec("out", "Output",
  `Output sets the file format, the length and the level of the IR, and the random seed.`,
  `The order of operations is: synthesis, then EQ, the DC high-pass, the cosine fade-out at the end, and finally one common normalisation gain for all channels. The auto length runs until the longest octave decay has fallen by <i>Auto-length floor</i> dB, capped by max_length_s (30 s by default; set it in the JSON tab).`, {
  "output.sample_rate": {t: "Sample rate", s: "Sample rate of the IR.", b: `Match your project or convolver. Higher rates extend the air absorption and the bands to 16 kHz and beyond.`, d: `Every stage runs at this rate. The analysis skips octave bands above 0.95 × Nyquist.`},
  "output.format": {t: "Format", s: "WAV sample format.", b: `PCM_24 is a good default; use FLOAT or DOUBLE for further processing.`, d: `TPDF dither (output.dither) is applied for PCM_16 and PCM_24. Values are clipped to the PCM range.`},
  "output.length_s": {t: "Length", s: "Fixed length in seconds; auto stops when the decay has fallen to the floor.", b: `Auto is almost always right. A fixed length truncates or pads.`, d: `Auto: t_direct + max(T60 over the audible octaves) × trim/60 (including the second slope when double slope is on), minus the lead shift, + 50 ms, capped at max_length_s.`},
  "output.trim_db": {t: "Auto-length floor", s: "How far (dB) the tail must decay before auto length stops.", b: `90 dB keeps the whole tail; 60 dB gives shorter files.`, d: `Used only for the auto length. Calibration and the analysis are unaffected.`},
  "output.normalize": {t: "Normalize", s: "peak (to the Peak level), energy (unit energy) or none.", b: `Use <i>energy</i> when you compare or swap IRs, so they don't jump in loudness.`, d: `One gain for all channels. Energy: 1/√(mean channel energy). The gain is reported as normalize_gain_db.`},
  "output.peak_dbfs": {t: "Peak", s: "Target peak level for peak normalisation.", b: `−1 dBFS leaves a little headroom.`, d: `The highest absolute sample over all channels.`},
  "output.lead_ms": {t: "Lead-in", s: "Silence before the direct sound.", b: `A short lead-in avoids cutting the start of the direct sound.`, d: `It leaves room for the 32-tap half-length windowed-sinc fractional delay of the direct sound and the reflections. With align_to_direct, the propagation delay is removed and the direct sound lands at the lead-in.`},
  "output.dc_highpass_hz": {t: "DC high-pass", s: "Removes DC and sub-sonic content.", b: `8–20 Hz is safe; 0 = off.`, d: `2nd-order Butterworth high-pass on the final IR.`},
  "seed": {t: "Seed", s: "Random seed: the same seed gives exactly the same IR.", b: `Change it for another variation with the same parameters.`, d: `It drives the image jitter, the parametric reflections, the tail noise and the modal detune through separate SeedSequence streams. Channels ≥ 2 use their own streams.`},
});

sec("room", "Room (physical model)",
  `The room is a rectangular (shoebox) space with a source and a microphone inside. It feeds every layer whose source is set to "room": early reflections, the late tail and the low-frequency modes.`,
  `Early part: an image-source model with octave-band absorption per surface, scattering, source/mic directivity and air absorption. Each reflection is rendered as a minimum-phase filter × windowed-sinc fractional delay. Late part: a statistical decay (Eyring or Sabine) whose early region is fed by the "diffuse rain", the energy scattering removes from the specular images. Below the Schroeder frequency, exact rectangular-room modes replace the statistical tail (in small rooms).`, {
  "room.dimensions": {t: "Dimensions", s: "Room size: length x, width y, height z in metres.", b: `Volume sets the decay time and the mode density. Proportions change the colour: cubes and simple ratios sound boomy, uneven ratios smoother.`, d: `V and S enter Eyring/Sabine and the mixing time 20·V/S + 12 ms. Image sources are enumerated to the crossover time. The modes are cos products up to 2.5 × the crossover frequency.`},
  "room.source.position": {t: "Source xyz", s: "Source position in metres (drag the orange dot in the plan).", b: `Close to a wall gives strong early reflections; the centre gives a more even build-up.`, d: `It must lie strictly inside the room. Together with the mic it sets the direct delay r/c, the 1/r level and every image's geometry.`},
  "room.source.directivity": {t: "Source pattern", s: "Radiation pattern of the source: omni, cardioid, figure-8…", b: `A directional source aimed away from the mic sounds more distant and reverberant.`, d: `First-order pattern g(θ) = a + (1 − a)·cos θ (omni a = 1, cardioid 0.5, hypercardioid 0.25, figure-8 0), applied per image along the mirrored emission direction. The diffuse tail is scaled by the pattern's mean g² (the diffuse power).`},
  "room.receiver.position": {t: "Mic xyz", s: "Microphone position in metres (drag the blue dot in the plan).", b: `Farther from the source means more reverb relative to direct sound.`, d: `Per-channel positions are offset from it by the multichannel array or the jitter.`},
  "room.receiver.directivity": {t: "Mic pattern", s: "Polar pattern of the microphone.", b: `A cardioid aimed at the source reduces the reverb; aimed away, it increases it.`, d: `The same first-order family as the source, applied to the arrival direction of each image.`},
  "room.receiver.orientation.0": {t: "Mic azimuth", s: "Direction the mic points in the horizontal plane (0° = +x, 90° = +y).", b: `Only matters for directional patterns.`, d: `[azimuth, elevation] → unit vector (cos el·cos az, cos el·sin az, sin el). With an "outward" array orientation, each capsule points away from the array centre instead.`},
  "room.receiver.orientation.1": {t: "Mic elevation", s: "Up/down tilt of the mic in degrees.", b: ``, d: `+90° points at the ceiling.`},
  "room.materials": {t: "Surfaces", s: "Material of each wall, the floor and the ceiling; the default applies to any surface left unset.", b: `Hard materials (concrete, glass, marble) give long, bright decays. Soft ones (carpet, curtains, foam) shorten and darken. Mixing them, e.g. a carpeted floor in a plaster room, sounds most natural.`, d: `Random-incidence absorption α per octave (63 Hz–16 kHz) from published tables, with the 63 Hz and 8/16 kHz values extrapolated. West = x 0, east = x Lx, south = y 0, north = y Ly. Each image is attenuated by Σ counts·10·log₁₀((1 − α)(1 − s)), and T60 uses the area-weighted mean ᾱ.`},
  "room.scattering": {t: "Scattering", s: "How much each reflection is diffused instead of mirror-like (0–0.99).", b: `More scattering gives a smoother, denser early sound with fewer distinct echoes.`, d: `The specular image energy is × (1 − s) per bounce. The removed energy ("diffuse rain") is histogrammed per octave, smeared causally with a one-pole filter (τ ≈ half the mean free time, 2–30 ms) and fed into the tail's early region, so the total energy is conserved.`},
  "room.image_jitter_m": {t: "Image jitter", s: "Small random displacement of the image sources: avoids metallic sweeping echoes.", b: `A few centimetres makes the perfect-box regularity sound more like a real room.`, d: `Gaussian σ in metres, added to every image of order > 0. It breaks the comb-like regularity of the perfect lattice.`},
  "room.reverb_formula": {t: "Reverb formula", s: "Eyring (accurate for absorbent rooms) or Sabine (classic).", b: `Eyring is the better default; Sabine gives slightly longer decays in dead rooms.`, d: `Sabine A = S·ᾱ; Eyring A = −S·ln(1 − ᾱ); T = 24 ln10·V / (c·(A + 4mV)), with the air attenuation m per octave.`},
  "room.air.temperature_c": {t: "Air temperature", s: "Sets the speed of sound and the air absorption.", b: `Warm air has slightly faster sound and different high-frequency damping.`, d: `c = 331.3·√(1 + T/273.15). Air absorption follows ISO 9613-1, applied per path length to every reflection and in the Eyring 4mV term.`},
  "room.air.humidity_pct": {t: "Humidity", s: "Relative humidity: affects how fast the highs are absorbed by the air.", b: `Dry air (20–30 %) damps the high frequencies of large halls noticeably more.`, d: `It enters the ISO 9613-1 molecular relaxation frequencies of O₂ and N₂.`},
  "room.modal.enabled": {t: "Modal low end", s: "Exact room modes below the Schroeder frequency (auto = on for small rooms).", b: `It gives small rooms their real bass resonances (boom, peaks and dips) instead of smooth noise.`, d: `Auto turns it on when the Schroeder frequency 2000·√(T/V) > 60 Hz, and only when early, late and direct are all "room". Modes h ∝ 4πc²/(f<sub>s</sub>V)·ψψ/Λ·sin(ωt)/ω·e<sup>−δt</sup> are evaluated as an exact DTFT on the FFT grid.`},
  "room.modal.crossover_hz": {t: "Modal crossover", s: "Frequency below which the modes replace the statistical tail (auto = Schroeder).", b: `Raise it for more pronounced room resonances.`, d: `Auto = the Schroeder frequency, clamped to 50–400 Hz. The early (correlated) part uses a Linkwitz-Riley crossover (amplitude complementary); the late part uses Butterworth (power complementary, no −3 dB hole). The two are blended with the ISM crossover window.`},
});

sec("dir", "Direct sound",
  `The direct sound is the first, dry arrival from source to mic.`,
  `Rendered with the same minimum-phase + fractional-delay renderer as the reflections, at amplitude (1/r)·source and mic directivity, with air absorption over r.`, {
  "direct.enabled": {t: "Direct on/off", s: "Includes the direct sound in the IR.", b: `Turn it off for a fully wet IR.`, d: `With it off, the modal low end is not used.`},
  "direct.gain_db": {t: "Direct gain", s: "Extra gain on the direct sound.", b: `Lower it for a more distant feel without moving the mic.`, d: `It only multiplies the direct-path amplitude. Parametric tails set their level relative to the direct sound through the direct/late ratio.`},
});

sec("er", "Early reflections",
  `Early reflections are the first distinct echoes off walls, floor and ceiling. They tell the ear the size and shape of the room.`,
  `<b>room</b>: shoebox image sources up to the crossover time, faded out with a cos window while the tail fades in. <b>parametric</b>: reflections placed statistically from perceptual parameters. <b>list</b>: your own reflections (time_ms, gain_db, polarity, tone) in the JSON tab. Every reflection gets its own causal filter, so there is no pre-ringing.`, {
  "early.source": {t: "Early source", s: "room (physical) · parametric (perceptual controls) · list (hand-placed) · none.", b: `Use room for realism and parametric to design the early sound freely.`, d: `The header switch Physical / Hybrid / Perceptual sets this together with the late source.`},
  "early.gain_db": {t: "Early gain", s: "Level of all early reflections.", b: `Raise it for a more "present" room, lower it for a smoother wash.`, d: `It multiplies every reflection amplitude.`},
  "early.crossover_ms": {t: "Crossover → tail", s: "When the image sources hand over to the statistical tail (auto = mixing time).", b: `Later crossovers keep more distinct echoes.`, d: `Auto = the perceptual mixing time 20·V/S + 12 ms (Lindau et al. 2012), after the direct sound.`},
  "early.crossover_width_ms": {t: "Crossover width", s: "Duration of the fade between reflections and tail.", b: ``, d: `Auto = max(0.5 × crossover, 5 ms). The ISM is weighted by cos and the tail energy by 1 − cos².`},
  "early.parametric.count": {t: "Count", s: "Number of parametric early reflections.", b: `10–40 is typical; more gives a denser early field.`, d: `Each one can be split into a cluster (cluster size).`},
  "early.parametric.start_ms": {t: "Start", s: "Time of the first reflection after the direct sound.", b: `This is the pre-delay: short in small rooms, 20–40 ms in large halls.`, d: `The first reflection is pinned exactly at the start.`},
  "early.parametric.end_ms": {t: "End", s: "End of the early-reflection window.", b: ``, d: `Reflection times lie in [start, end].`},
  "early.parametric.density_exponent": {t: "Density exponent", s: "How the reflection density grows over time: 0 = even, 2 = real-room (∝ t²).", b: `2 mimics real rooms: sparse at first, then denser and denser.`, d: `Times t = (t₀<sup>k+1</sup> + u·(t₁<sup>k+1</sup> − t₀<sup>k+1</sup>))<sup>1/(k+1)</sup> with u ~ U(0, 1) sorted, so density ∝ t<sup>k</sup> (Polack: k = 2).`},
  "early.parametric.level_start_db": {t: "Level at start", s: "Level of the first reflections, re the direct sound.", b: ``, d: `The level is interpolated in dB from start to end.`},
  "early.parametric.level_end_db": {t: "Level at end", s: "Level of the last reflections, re the direct sound.", b: ``, d: ``},
  "early.parametric.level_curve": {t: "Level curve", s: "Shape of the level fade: 1 = linear in dB, <1 drops quickly, >1 holds longer.", b: ``, d: `lev = l₀ + (l₁ − l₀)·x<sup>curve</sup>, x = normalised time.`},
  "early.parametric.level_jitter_db": {t: "Level jitter", s: "Random level variation per reflection.", b: `A few dB makes it sound less synthetic.`, d: `Gaussian σ in dB.`},
  "early.parametric.polarity": {t: "Polarity", s: "Sign of the reflections: random, all positive or all negative.", b: `Random is natural. All-positive sounds harder and more "boxy".`, d: ``},
  "early.parametric.color_from_decay": {t: "Colour from decay", s: "Gives later reflections the same tone as the tail (highs fade faster).", b: `Keep it on for a seamless join between reflections and tail.`, d: `band_dB = −60·t·(1/T60(f) − 1/T60(1 kHz)) per reflection.`},
  "early.parametric.hf_damping_db_per_oct": {t: "HF damping", s: "Extra treble loss per reflection, growing with time.", b: `Darkens the early sound like soft walls.`, d: `−|hf| dB per octave above 1 kHz × (0.5 + x).`},
  "early.parametric.absorption": {t: "Absorption colour", s: "Colours the reflections as if they bounced off this material.", b: ``, d: `It applies (1 + x·bounces_end) bounces of 10·log₁₀(1 − α) per octave.`},
  "early.parametric.bounces_end": {t: "Bounces at end", s: "How many material bounces the last reflections have had.", b: ``, d: `Linear from 1 bounce at the start to 1 + this at the end.`},
  "early.parametric.cluster_size": {t: "Cluster size", s: "Splits each reflection into a small cluster of pulses (diffuse walls).", b: `2–4 softens the echoes.`, d: `Random offsets within the spread, Gaussian weights with an exponential taper, energy normalised per cluster.`},
  "early.parametric.cluster_spread_ms": {t: "Cluster spread", s: "Time spread of each cluster.", b: ``, d: ``},
});

sec("late", "Late reverberation",
  `The late tail is the dense, decaying wash after the early reflections. Its decay time per frequency (T60) is the main "size and material" impression.`,
  `Noise (Gaussian or velvet, with growing echo density) is split by a perfect-reconstruction fractional-octave filterbank (cos² masks summing to 1, zero phase). Each band gets its envelope √E(t): the room energy decay 4πc/V·diffuse_gain·e<sup>−13.8t/T</sup> with the diffuse rain in its early region, or the parametric decay with onset and attack. An optional per-band flattening removes the slow random level fluctuations of the noise.`, {
  "late.source": {t: "Late source", s: "room (from the room model) · parametric (designed decay) · none.", b: `Parametric lets you set the decay directly without a room.`, d: `room: the absolute level follows from the room's diffuse energy density. parametric: the level is set by the direct/late ratio (0 dB by default).`},
  "late.t60": {t: "Custom T60 per octave", s: "Draw the decay time per octave yourself (off = computed from the room).", b: `Typical halls: 2–2.5 s mid, longer in the bass, shorter in the treble.`, d: `9 octave values 63 Hz–16 kHz, interpolated in log-frequency for every band. It overrides Eyring/Sabine.`},
  "late.t60_scale": {t: "T60 scale", s: "Multiplies every decay time.", b: `A quick "bigger / smaller" knob.`, d: `Applied before calibration. Minimum 20 ms.`},
  "late.calibrate": {t: "Calibrate T30", s: "Iterations that measure the result and correct it until T30 matches the target.", b: `3 gives accurate decays; 0 renders fastest (or tick <i>draft</i> next to Render).`, d: `Each iteration analyses the octave T30 (T20 as fallback) of the rendered IR and multiplies the internal T by target/measured (clamped 0.5–2), until within 2 %. It runs on a mono render and is skipped with double slope or no tail. The resulting factors are cached per exact calibration setup (everything except name, output format, level and multichannel settings), so an identical re-render skips the passes and gives the same IR. Set IRGEN_CALIBRATION_CACHE=0 to disable the cache.`},
  "late.gain_db": {t: "Late gain", s: "Level of the late tail.", b: ``, d: `Applied after the direct/late scaling.`},
  "late.drr_db": {t: "Direct / late ratio", s: "Energy of the direct sound over the tail (auto = physical for room, 0 dB for parametric).", b: `Lower values (−10 dB) sound far away, higher values close.`, d: `Scales the tail energy to a_d²/10^(drr/10).`},
  "late.onset_ms": {t: "Onset", s: "When the parametric tail starts after the direct sound (auto = start of the early window).", b: ``, d: `Auto = early.parametric.start_ms, the first list reflection, or 2 ms.`},
  "late.attack_ms": {t: "Attack", s: "Fade-in time of the parametric tail.", b: `Longer attacks give a blooming, swelling reverb.`, d: `sin(π/2·x<sup>shape</sup>) ramp, x = t/attack.`},
  "late.attack_shape": {t: "Attack shape", s: ">1 blooms more slowly, <1 more quickly.", b: ``, d: `The exponent on the ramp's time axis.`},
  "late.noise": {t: "Noise", s: "Excitation: Gaussian (smooth) or velvet (sparse pulses, growing density).", b: `Velvet sounds more like real echoes building up; Gaussian is smoothest.`, d: `Auto = velvet for parametric, Gaussian for room (where the build-up comes from the scattered energy).`},
  "late.density.initial_per_s": {t: "Initial density", s: "Echo density at the start of the velvet tail (pulses per second).", b: `Low values give audible early "grain".`, d: `The density grows as t<sup>exponent</sup> to the maximum at "full density".`},
  "late.density.full_ms": {t: "Full density at", s: "Time at which the tail reaches full echo density.", b: ``, d: `Auto = the mixing time (room) or 80 ms. It then crossfades to Gaussian over 0.3 × that time.`},
  "late.density.max_per_s": {t: "Max density", s: "Maximum pulses per second.", b: ``, d: `Auto = f<sub>s</sub>/4.`},
  "late.density.pulse_amplitude": {t: "Pulse amplitude", s: "Velvet pulse heights: Gaussian (natural) or ±1 (classic velvet).", b: ``, d: `Unit mean power either way.`},
  "late.double_slope.enabled": {t: "Double slope", s: "Adds a second, slower decay: a coupled-room / chamber effect.", b: `Makes the reverb fall fast first, then hang on quietly.`, d: `E = E₁ + 10^(level/10)·E₂, with T₂ = ratio × T₁ (or set explicitly in the JSON). Calibration is off.`},
  "late.double_slope.t60_ratio": {t: "T60 ratio", s: "How much longer the second slope is.", b: ``, d: ``},
  "late.double_slope.level_db": {t: "Second slope level", s: "Level of the second slope re the first.", b: ``, d: ``},
  "late.tone": {t: "Tail tone per octave", s: "Draw the tail's spectral balance in dB per octave.", b: `Shape a darker or brighter tail without changing its decay.`, d: `A dB curve on the band levels (not on T60).`},
  "late.envelope_flatten": {t: "Flatten envelope", s: "Removes random level wobbles of the noise for an exact, smooth decay.", b: `Keep it on unless you want a more irregular tail.`, d: `Per band: the power is smoothed over flatten_cycles/bandwidth and divided out (gain clamped 0.5–2), keeping the fine structure random. It makes T and EDT match across seeds and channels.`},
  "late.bands_per_octave": {t: "Bands per octave", s: "Frequency resolution of the tail's decay shaping.", b: `3 is a good balance; more is finer but slower.`, d: `Bands from 20 Hz to 0.98 × Nyquist, cos² masks on a log-frequency axis.`},
});

sec("mc", "Multichannel (1–64 decorrelated)",
  `Multichannel renders 2–64 IRs of the same room: statistically decorrelated, but perceptually identical (same decay, level and tone). Use it for stereo, surround, arrays or auralization.`,
  `Each channel gets its own random tail. In every filterbank band, the tails are mixed with the symmetric square root of a target coherence matrix, so their pairwise correlation matches the target per frequency. All channels share one envelope. Early and direct parts can be identical, jittered per channel, or independent. The Multichannel tab verifies decorrelation and similarity against JNDs (T30/EDT 5 %, C80 1 dB, level 1 dB, spectrum).`, {
  "multichannel.channels": {t: "Channels", s: "Number of output channels (1–64).", b: ``, d: `Calibration runs on a mono render; the full set is rendered once at the end.`},
  "multichannel.coherence.model": {t: "Coherence model", s: "How similar the tails of different channels are: independent, constant, or diffuse-field from the mic spacing.", b: `Independent is widest. <i>Diffuse</i> behaves like real spaced mics: coherent in the bass, independent in the treble.`, d: `independent: 0. constant: value (a number or a curve). diffuse: sin(kd)/kd; diffuse_2d: J₀(kd), with d the mic spacing from the array. custom: a matrix (JSON).`},
  "multichannel.coherence.value": {t: "Coherence", s: "Target correlation between channels (constant model).", b: `0 = unrelated, 1 = identical.`, d: `Off-diagonal entries of the target matrix.`},
  "multichannel.coherence.coherent_blend": {t: "Bass coherent below", s: "Makes the channels identical below one frequency and independent above another: mono bass.", b: `Typical: coherent below 80 Hz, independent above 160 Hz.`, d: `A curve {freqs:[f₁, f₂], values:[1, 0]} blending the target matrix toward full coherence.`},
  "multichannel.array.type": {t: "Mic array", s: "Geometry of the channels' mic positions: line, circle, sphere, grid or random.", b: `It defines the spacing that the diffuse coherence and the per-channel early reflections use.`, d: `Offsets around the main mic position. Shown in the plan as green dots.`},
  "multichannel.array.spacing_m": {t: "Spacing", s: "Distance between neighbouring mics (line, grid).", b: ``, d: ``},
  "multichannel.array.radius_m": {t: "Radius", s: "Array radius (circle, sphere, random).", b: ``, d: ``},
  "multichannel.array.orientation": {t: "Orientation", s: "same (all mics point one way) or outward (each points away from the centre).", b: ``, d: ``},
  "multichannel.early": {t: "Early per channel", s: "same (identical early part) · jitter (each mic at its own position) · independent (new pattern per channel).", b: ``, d: `room: jitter renders the ISM at each mic position (the image jitter is shared). parametric: time_jitter_ms / gain_jitter_db on a common pattern.`},
  "multichannel.direct": {t: "Direct per channel", s: "auto (per channel when there is an array) · same · per_channel.", b: ``, d: `Per channel uses each mic's own distance and direction; the lead-in grows so every direct sound lands after it.`},
  "multichannel.receiver_jitter_m": {t: "Mic jitter (no array)", s: "Random mic displacement per channel when no array is set.", b: ``, d: `Gaussian σ in metres; channel 1 stays in place.`},
  "multichannel.time_jitter_ms": {t: "Time jitter", s: "Per-channel time variation of the parametric reflections.", b: ``, d: `Gaussian σ (ms), in jitter mode.`},
  "multichannel.gain_jitter_db": {t: "Gain jitter", s: "Per-channel level variation of the parametric reflections.", b: ``, d: `Gaussian σ (dB).`},
  "multichannel.export": {t: "Export", s: "One interleaved N-channel WAV, separate mono files, or both.", b: ``, d: `Separate files go into &lt;name&gt;_channels/.`},
});

sec("eq", "EQ",
  `A final EQ for the whole IR, and a separate one for the late tail only.`,
  `Causal RBJ biquads (peak, shelves, band-pass, notch); low-pass and high-pass with an order other than 2 are Butterworth. The tail EQ runs before the direct/late scaling; the whole-IR EQ runs after the modal stage.`, {
  "eq": {t: "EQ (whole IR)", s: "Filters applied to the complete IR.", b: `Add a band, then set its type, frequency, gain and Q.`, d: ``},
  "eq_late": {t: "EQ (late tail only)", s: "Filters applied only to the late reverberation.", b: `E.g. darken the tail while keeping the early sound bright.`, d: ``},
});

Object.assign(P, {
  "ui.render": {t: "Render", s: "Synthesises the IR with the current settings (Ctrl/⌘ + Enter).", b: `It writes to renders/gui/&lt;name&gt;/. The report is separate (Analyze).`, d: ``},
  "ui.analyze": {t: "Analyze", s: "Runs the acoustic analysis and report(s) of the last render (only when you ask).", b: `It fills T30, EDT, C80, D50 and DRR, the report image and, with several channels, the multichannel report.`, d: `ISO 3382: EDT, T20, T30, C50, C80, D50, Ts, DRR, BR, Br, echo density. Multichannel: pairwise late correlation, octave coherence vs target, and similarity vs JNDs.`},
  "ui.mode": {t: "Physical / Hybrid / Perceptual", s: "Sets where the early and late layers come from in one click.", b: `<b>Physical</b>: everything from the room. <b>Hybrid</b>: room reflections + a designed tail. <b>Perceptual</b>: no room, everything designed.`, d: `physical = early room + late room; hybrid = early room + late parametric; perceptual = early parametric + late parametric.`},
  "ui.strange": {t: "Strange spaces", s: "Opens the strange-spaces layer: bizarre, miniature, gigantified and scale-tuned IRs.", b: `Your settings here are kept while you are there.`, d: ``},
  "ui.presets": {t: "Presets", s: "Loads a preset from presets/.", b: ``, d: `Presets are partial JSON merged over the defaults.`},
  "ui.save": {t: "Save preset", s: "Saves the current settings to presets/&lt;name&gt;.json.", b: ``, d: ``},
  "ui.view": {t: "Full / Simple view", s: "Simple shows only the parameters with the biggest audible effect; Full shows everything.",
    b: `Simple keeps the room plan, dimensions, surfaces, T60 scale, direct/late ratio, early gain, pre-delay and tail attack (the last two only in parametric mode), and channels. Use the Physical / Hybrid / Perceptual switch to change the layers.`,
    d: `Hidden parameters keep their values and still render; switching to Full restores every control and the open sections.`},
  "ui.blend_link": {t: "Blend", s: "Opens the Blend page (new tab): mix or morph two multichannel IR folders into a new one.", b: `Render or pick two IR folders first (sets, GUI renders or earlier blends).`, d: `See the Blend docs at /blend/docs.`},
  "ui.draft": {t: "Draft", s: "Renders without the T30 calibration passes: several times faster, decays a few % off target.", b: `Tick it while you tweak and listen; untick it for the final render. The preset is not changed.`, d: `It sends late.calibrate = 0 for that render only. Calibrated renders are cached per exact calibration setup (renders/cache/calibration), so re-rendering the same room (other seed-independent changes such as level, format or channels) reuses the factors and is just as fast; the output is identical to a fresh calibration.`},
  "ui.help": {t: "Help mode", s: "Point at any control to see what it does. Click to pin the card. Esc or Help again to exit.", b: ``, d: ``},
  "ui.plan": {t: "Room plan", s: "Top view of the room: drag the source (orange) and the mic (blue); green dots are the array mics.", b: ``, d: `x to the right, y up. Heights are set in the xyz fields.`},
  "ui.envelope": {t: "Envelope plot", s: "Level of the rendered IR over time (channel 1, dB).", b: `A straight line is a clean exponential decay.`, d: `2 ms RMS frames, normalised to the peak.`},
  "ui.cards": {t: "Result cards", s: "Key numbers of the last render; the measured ones appear after Analyze.", b: ``, d: `The volume, mixing time and Schroeder frequency come from the room model.`},
  "ui.audition": {t: "Audition", s: "Convolves a test sound (or your file) with the IR so you can hear it.", b: `With several channels, pick which two go left and right.`, d: `FFT convolution, level matched, wet/dry by equal power.`},
  "ui.tabs": {t: "Report tabs", s: "Report, multichannel report, metrics table, and the full preset JSON (edit keys without a control).", b: ``, d: ``},
});
return {sections: S, params: P};
})();
