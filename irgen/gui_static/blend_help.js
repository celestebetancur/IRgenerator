/* Help content for the Blend page: one source for the help-mode overlay
   (t = title, s = short) and the docs page (b = basics, d = in depth). HTML allowed. */
"use strict";
window.IRGEN_HELP = (() => {
const S = [];
const P = {};
function sec(id, title, b, d, params){ S.push({id, title, b, d, params: Object.keys(params)}); Object.assign(P, params); P["section." + id] = {t: title, s: b.split(". ")[0] + ".", b, d, section: true}; }

sec("inputs", "Inputs",
  `Choose two multichannel IR folders, A and B. Channel 1 of A is blended with channel 1 of B, channel 2 with channel 2, and so on, so the new folder has as many channels as the smaller of the two.`,
  `Any folder irgen writes works: an IR set from the API (renders/sets/&lt;id&gt;, read through its manifest), a standard or strange GUI render, an earlier blend (blends can be blended again), or any folder of WAVs. In a plain folder the WAV with the most channels is used, or else the mono WAVs sorted by name, one per channel; files starting with "_" (previews, auditions) are ignored. If the sample rates differ, B is resampled to A's rate (polyphase). Both are aligned at their direct sound (the manifest's direct_index, or the first sample within 20 dB of the peak) and scaled to the same mean channel energy, so the amount is balanced whatever their original levels.`, {
  "blend.a": {t: "Folder A", s: "First IR folder: the result is pure A at amount 0.", b: `Pick it from the list (blends, API sets, standard and strange renders), or type a path, which takes priority over the list.`, d: `Accepted forms: a path (absolute or relative to the project), set/&lt;id&gt; or a bare set id, blend/&lt;name&gt;, gui/&lt;name&gt;, strange/&lt;name&gt;.`},
  "blend.b": {t: "Folder B", s: "Second IR folder: the result is pure B at amount 1.", b: `If B has a different number of channels, only the first min(A, B) of each are used.`, d: `B is resampled to A's rate if they differ.`},
});

sec("blend", "Blend",
  `Morph creates a new space in between the two; Mix layers them. Amount slides from A (0) to B (1).`,
  `<b>Mix</b>: y = w<sub>A</sub>·A + w<sub>B</sub>·B. Linear weights (1 − a, a) for the first 2.5 ms after the direct sound, where both IRs share the same impulse and add coherently, then equal-power weights (√(1 − a), √a) for the uncorrelated rest, with a 2.5 ms raised-cosine transition.<br><b>Morph</b>, per 1/3-octave band (zero-phase cos² masks) and channel: each band's energy envelope is measured in 1 ms frames (smoothed over about 8 cycles, at most 20 ms) and its decay slope fitted on the band's Schroeder curve (−5…−35 dB). The target decay time is geometric, T = T<sub>A</sub><sup>1−a</sup>·T<sub>B</sub><sup>a</sup>, because decay-time differences are heard relatively (the T60 JND is about 5 %), so equal amount steps sound like equal steps. Each envelope is time-warped onto the target decay (so its shape, such as a build-up or a double slope, is kept), and the two are averaged in dB (level and spectrum interpolate). Envelopes that hit their noise floor or end early are extrapolated along their slope. The fine structure is an equal-power mix of the two band signals, each re-shaped by gain to the target envelope. Where one input has decayed below its reliable range (70 dB under its peak), the other carries the band alone. Measured on a small studio (T30 0.2 s) → concert hall (1.8 s) morph: T30 within ±3 % of the geometric target from 250 Hz up; the modal low end of small rooms (125 Hz band) lands about 10 % high. Channel decorrelation is kept.`, {
  "blend.mode": {t: "Mode", s: "Morph = a space in between (decay, spectrum, level interpolated). Mix = both spaces layered.",
    b: `<b>Morph</b> at 50 % of a small room and a cathedral gives a medium-sized room, not a small room with a cathedral behind it. <b>Mix</b> gives exactly that: both at once, as if the sound were in two places.`,
    d: `Morph processes channel groups in parallel. About 1 s for 8 channels × 3.6 s, about 15 s for 64 channels × 6 s. Mix is instant.`},
  "blend.amount": {t: "Amount", s: "0 = A, 1 = B; in between, the blend.",
    b: `In Morph, equal steps of amount are heard as equal steps of size/decay. 0 and 1 reproduce A or B (aligned and level-matched).`,
    d: `Morph: T = T<sub>A</sub><sup>1−a</sup>·T<sub>B</sub><sup>a</sup>, level = (1 − a)·L<sub>A</sub> + a·L<sub>B</sub> dB per band. Mix: amplitude weights (1 − a, a) at the direct sound, √(1 − a), √a after.`},
});

sec("output", "Output",
  `The blend is written to renders/blends/&lt;name&gt;/ as a folder in the same format as an IR set, with one WAV per channel and a manifest.json, so AURA (or any host) can load it like any other set.`,
  `The folder is written to a temporary name and renamed into place, so a reader never sees a half-written blend; overwriting replaces it the same way. The manifest has engine "blend", the sources, mode and amount, the sample rate, channels, length, direct_index (the same sample in every channel), normalisation gain and the file list. It is served at /v1/blends/&lt;name&gt; and can be created with POST /v1/blends or <code>python -m irgen blend</code>. A 20 ms fade-out is applied at the end.`, {
  "blend.name": {t: "Name", s: "Name of the new folder: renders/blends/<name>.", b: `Letters, digits, - and _ (anything else becomes _). If it exists you are asked before it is replaced.`, d: ``},
  "blend.normalize": {t: "Normalize", s: "energy = equal loudness between sets (best for hot-swapping); peak = −1 dBFS; none.", b: `Keep energy when the blend will be swapped with other sets in AURA.`, d: `One gain for all channels: 1/√(mean channel energy), or −1 dBFS peak. It is reported as gain_db in the manifest.`},
  "blend.layout": {t: "Files", s: "One mono WAV per channel (ch01.wav…), one interleaved WAV, or both.", b: `Per-channel files are easiest for convolution plugins.`, d: `The same layouts as IR sets.`},
  "blend.format": {t: "Format", s: "WAV sample format (FLOAT keeps full precision).", b: ``, d: `TPDF dither for the PCM formats.`},
});

Object.assign(P, {
  "ui.blend": {t: "Blend", s: "Creates the new folder (Ctrl/⌘ + Enter).", b: ``, d: ``},
  "ui.swap": {t: "Swap A ↔ B", s: "Exchanges A and B and mirrors the amount, so the result stays the same.", b: ``, d: ``},
  "ui.refresh": {t: "Rescan", s: "Re-reads the list of IR folders (after rendering new sets elsewhere).", b: ``, d: ``},
  "ui.summary": {t: "Summary", s: "Where the blend will be written, how many channels it will have, and any channel or sample-rate mismatch.", b: ``, d: ``},
  "ui.envelope": {t: "Envelopes", s: "Level over time of channel 1 of A, B and the result, aligned and on one energy scale.", b: `In Morph the result's decay sits between A and B; in Mix it follows A early and B late.`, d: `2 ms RMS frames.`},
  "ui.cards": {t: "Result", s: "Channels, mode, length, sample rate and the direct-sound sample of the new folder.", b: ``, d: ``},
  "ui.preview": {t: "Preview", s: "Plays channels 1–2 of the blend.", b: ``, d: ``},
  "ui.audition": {t: "Audition", s: "Convolves a test sound (or your file) with the blend.", b: `With 3 or more channels, pick which go left and right.`, d: ``},
  "ui.manifest": {t: "Manifest", s: "The manifest.json written with the blend: what AURA reads to load it.", b: ``, d: ``},
  "ui.back": {t: "← Standard", s: "Opens the standard IR generator.", b: ``, d: ``},
  "ui.help": {t: "Help mode", s: "Point at any control to see what it does. Click to pin the card. Esc or Help again to exit.", b: ``, d: ``},
});
return {sections: S, params: P};
})();
