# Development Task — Handoff Document

> **Purpose.** This file is a context handoff between two collaborations:
> - **Claude Code (in IDE)** wrote this file and will execute the resulting plan.
> - **Claude (chat / claude.ai)** will read this file, discuss design with the user,
>   and fill in the **Plan** section at the bottom (discussion + decisions + tasks).
>
> When Claude (chat) is done, the file is pushed back to GitHub. Claude Code pulls
> it and executes the tasks under "Numbered Task List".

---

## 1. Project Snapshot

**Repo:** https://github.com/MSelmanS/NoiseSuppression
**Goal:** Mic-input noise suppression + dominant-speaker selection. Final target is
an Android service; PC work is for model selection and methodology.
**Constraint:** Single channel (mono). 16 kHz internal processing pipeline.
**Status:** Benchmark infrastructure is solid; speaker-selection, real-time, and
mobile deployment work are still ahead.

### 1.1 What's been built (✅ done)

- **Common model interface** — `models/base.py:BaseDenoiser` (abstract `load()` + `process()`)
- **Audio I/O** — `audio_io/file_io.py:load_audio` (resamples to 16 kHz mono float32) + `save_audio`
- **Seven denoiser wrappers** — all derived from `BaseDenoiser`:
  - `spectral_subtraction` (classical DSP baseline, no ML)
  - `rnnoise` (48 kHz GRU, lightweight)
  - `deepfilternet` (48 kHz internal, 2.1M params, ~0.04 RTF on CPU)
  - `demucs_dns48` / `demucs_dns64` / `demucs_master64` (18.9M / 33.5M / 33.5M params)
  - `metricgan_plus` (SpeechBrain, GAN, 1.9M params)
- **Benchmark engine** — `benchmark/`:
  - `metrics.py` — `PeakRSSTracker` (threaded RSS sampler, ~50 ms), `si_sdr`, `stoi_score`, `pesq_score`, `time_it`, `model_size_mb`, `param_count`
  - `runner.py` — `run_model` (load + N-repeat process with warmup) and `run_processing_only` (process-only, for the synthetic loop)
  - `mixer.py` — `mix_at_snr` (RMS-scaled clean + noise, exact target SNR verified to 0.01 dB)
  - `report.py` — CSV, formatted XLSX, per-SNR pivot XLSX (one sheet per metric), matplotlib metric-vs-SNR PNGs
- **Two entry points** — `scripts/`:
  - `bench_real.py` — single wav, performance only, auto-finds a wav if `--wav` omitted
  - `bench_synthetic.py` — clean × noise × SNR sweep, perf + SI-SDR + STOI + PESQ, saves denoised wavs into per-model subfolders + shared `_input/`
  - `_model_registry.py` — `MODEL_REGISTRY` dict; CLI `--models all` or `--models rnnoise,deepfilternet`
- **VSCode integration** — `.vscode/launch.json` has four ready-to-run configurations: full sweep, quick sweep, bench_real auto-find, bench_real prompt
- **Legacy reference** — old monolithic script preserved at `legacy/benchmark_all.py`

### 1.2 Current Repo Layout

```
NoiseSuppression/
├── config.py              # MODEL_SR=16000, CAPTURE_SR=48000, CHANNELS=1
├── audio_io/file_io.py    # load_audio, save_audio
├── models/                # BaseDenoiser + 7 wrappers
├── benchmark/             # metrics, runner, mixer, report
├── scripts/               # bench_real, bench_synthetic, _model_registry
├── legacy/                # old benchmark_all.py
├── data/                  # data/clean/, data/noise/ (currently small placeholder set)
├── output/                # zaman damgalı sonuçlar (gitignored)
├── pretrained_models/     # SpeechBrain cache (gitignored)
├── .venv/                 # Python 3.11.9 (gitignored)
├── requirements.txt
├── tasarim_dokumani.md    # main design doc (Turkish)
└── development_task.md    # ⬅ this file
```

### 1.3 Sample Benchmark Numbers (CPU, 10s wav, 3 repeats)

| Model | RTF | Peak RAM | Params | Notes |
|---|---|---|---|---|
| spectral_subtraction | 0.004 | 231 MB | 0 | Baseline, no ML |
| metricgan_plus | 0.018 | 910 MB | 1.9 M | GAN, PESQ-optimized |
| deepfilternet | 0.044 | 404 MB | 2.1 M | Best quality/size trade-off so far |
| demucs_dns48 | 0.046 | 681 MB | 18.9 M | |
| rnnoise | 0.066 | 259 MB | 0 | (C lib, no PyTorch) |
| demucs_dns64 | 0.075 | 893 MB | 33.5 M | |
| demucs_master64 | 0.077 | 893 MB | 33.5 M | |

All real-time-feasible on CPU. **DeepFilterNet is the leading mobile candidate**
(low RAM, low param count, RTF < 0.05) but no Android-side measurement yet.

> **Caveat on synthetic quality scores:** current `data/clean/` is populated
> with *already-denoised* wavs from prior runs (model outputs, not pristine speech).
> Real assessment needs LibriSpeech / VCTK clean speech + DEMAND / MUSAN noise.

---

## 2. Open Roadmap

Pulled from `tasarim_dokumani.md` § 9 ("Sıradaki"):

### 2.1 Dominant-speaker selection (VAD + RMS)
- **Why:** the project's *second* goal after noise suppression — pick the loudest/closest speaker and attenuate weaker background voices.
- **Approach already decided in design doc:** energy-based. VAD to find speech regions → RMS in short windows → keep the dominant speaker, suppress weaker ones.
- **Open questions for Claude:**
  - Which VAD? (Silero VAD, WebRTC VAD, py-webrtcvad, custom energy thresholding?)
  - Window size + hop for RMS calculation?
  - How to "attenuate" the weaker speaker? Hard gate, soft mask, or spectral subtraction?
  - Where in the pipeline does this sit — *before* or *after* the denoiser?
  - Module structure: a new `models/speaker_selector.py` à la `BaseDenoiser`, or a separate `pipeline/` package because it's not a denoiser per se?
- **Suggested files to touch:** `pipeline/speaker_selector.py` (new), `scripts/run_pipeline.py` (new — chain denoiser + speaker selector), possibly an extension to `BaseDenoiser` or a new `BaseProcessor` abstraction.

### 2.2 Live microphone pipeline (real-time)
- **Why:** all current benchmarks are file-based. Need to verify chosen model survives streaming constraints (frame-by-frame, no future context, bounded latency).
- **Open questions for Claude:**
  - Use `sounddevice` (already in requirements) callback model, or pull mode with a separate worker thread?
  - Frame size? 10 ms / 20 ms / 30 ms (RNNoise wants 10 ms; DeepFilterNet has internal hop)
  - How to handle each model's natural chunk size mismatch with the audio device's callback size? Ring buffer?
  - Need a "live mode" wrapper around `BaseDenoiser` — current API is `process(audio: ndarray) -> ndarray`, not streaming. Should we add `process_chunk()` to the interface, or build a wrapper?
- **Suggested files to touch:** `audio_io/mic_capture.py` (new), `pipeline/live_runner.py` (new), maybe `models/base.py` extension.

### 2.3 ONNX / TFLite conversion
- **Why:** PyTorch models can't ship to Android directly. Need to export to ONNX or TFLite and verify the exported model produces identical output to the PyTorch one.
- **Open questions for Claude:**
  - Which model(s) to target first? Recommendation: **DeepFilterNet** (already the leading mobile candidate). MetricGAN+ as backup.
  - ONNX or TFLite? ONNX is broader; TFLite is Android-native via TFLite Java/Kotlin. For Android end target, TFLite probably wins. But ONNX → TFLite via `onnx2tf` is a common path.
  - DeepFilterNet has internal DSP state (`df_state`) — does that export cleanly? May need to unroll or rewrite.
  - Tolerance for output divergence? (Quantization will introduce some.)
- **Suggested files to touch:** `export/onnx_export.py` (new), `export/verify_export.py` (new), maybe a new "exported model" wrapper inheriting `BaseDenoiser` so the same benchmark works on the exported binary.

### 2.4 Android-side measurement + integration
- **Why:** the actual deployment target. Need on-device RTF, latency, RAM, battery numbers.
- **Open questions for Claude:**
  - Out of scope for this Python repo, but: what minimum Android app stub do we need to host the model and measure? Native NNAPI? TFLite runtime?
  - Where do the measurements get reported back into the benchmark XLSX format?

---

## 3. Cross-Cutting Concerns / Tech Debt

These came up during the refactor; useful context when scoping new work.

- **PESQ on Windows** needs MSVC Build Tools. Currently set up on the dev machine; the `pesq_score()` wrapper falls back to NaN if missing. No action needed unless we want to make the install path smoother for new contributors.
- **`pretrained_models/`** is gitignored. SpeechBrain caches MetricGAN there on first run; we may want a script to pre-fetch all weights to make CI / fresh-clone friendly.
- **Clean / noise data sources** — currently 2 clean + 2 noise wavs (placeholders). For real synthetic benchmarks, need a script to download a subset of LibriSpeech `test-clean` + DEMAND.
- **`benchmark_all.py` (now in legacy/)** — kept as reference, can be deleted once nothing references it.
- **Type checking / linting** — no `ruff`, `pyright`, or pre-commit currently. Light touch — discuss whether to add.

---

## 4. Constraints & Preferences

- **Python 3.11.9, Windows + Linux dev** (current dev box is Windows; deployment target is Android).
- **Mono, 16 kHz internal** is the standard. Don't break this without a migration plan.
- **Turkish documentation language** in `tasarim_dokumani.md`. Code comments are mixed (mostly Turkish in modules, English in the new ones).
- **Each benchmark run gets its own timestamped folder** — no overwrites. Keep this for new pipeline outputs too.
- **No future context** in real-time mode — every chunk decision must be based only on past samples (rules out look-ahead in any speaker selector).
- **Module file size discipline** — current files are small and focused; new modules should follow this style.

---

## 5. Plan (← Claude, fill in here)

### 5.1 Discussion / Design Decisions

> *(Claude — chat side: use this section freely. Discuss trade-offs, raise concerns,
> propose alternatives. Reference section numbers above when relevant. Anything
> you can't decide without the user's input, ask explicitly and write the question
> here so Claude Code sees it when it pulls the file back.)*

_Empty — to be filled by Claude in chat._

### 5.2 Decisions Summary

> *(Crisp bullet list of what was decided, after the discussion above. This is
> what Claude Code will treat as binding.)*

- _Empty — to be filled._

### 5.3 Numbered Task List

> *(One task per numbered item. Format below. Order matters — Claude Code will
> execute top-to-bottom. Mark dependencies explicitly where they exist.)*

**Template:**

```
### Task N — <short title>
- **Why:** <one sentence — what problem this solves>
- **Files to touch:** <bullet list of paths; mark (new) for new files>
- **Approach:** <2-5 sentences — the key implementation idea, not the full code>
- **Acceptance criteria:**
  - [ ] <verifiable thing #1>
  - [ ] <verifiable thing #2>
  - [ ] <test or smoke command that should pass>
- **Out of scope:** <what NOT to do in this task; saves churn>
- **Depends on:** <Task X, or "none">
```

_No tasks yet — to be filled._

---

## 6. Handoff Protocol

1. **Claude Code → GitHub** (this commit): file created on `master`.
2. **User → Claude (chat):** shares this file. Discussion happens in chat. Claude in chat fills sections 5.1, 5.2, 5.3 inline.
3. **User → GitHub:** updated file pushed back (manually or via Claude in chat).
4. **Claude Code:** pulls the file, reads section 5.3, executes tasks in order, marks acceptance-criteria checkboxes as they pass, commits per task.
5. **Repeat** for next planning round: blank out section 5 (or move old plan into `development_task_archive/<date>.md`) and re-handoff.