# Track C project demo video script

**Target length:** 3:30–4:00 at a calm speaking pace.  
**Format:** screen recording with your own narration or voice-over. Keep the dashboard and README readable; avoid showing private participant audio or personal information.

## Before recording

1. Start the local project server and open `evaluation.html` in a browser. Maximize the browser and close unrelated tabs. Confirm the page has loaded the benchmark examples.
2. Prepare the dashboard on one clear benchmark comparison. Keep the one-edit filter checked. Have the README results section ready in another tab.
3. Record the screen at 1080p if available. Speak slowly, pause briefly between sections, and avoid moving the pointer while explaining a chart.
4. If a live interaction fails, record the narration separately and use a clean screen capture. Do not claim a live upload unless you actually demonstrate one.

## Timed narration and screen plan

### 0:00–0:25 — Opening and problem

**Show:** Project title card or dashboard header.  
**Say:**

“Hello, we are presenting Track C Speech Analysis, an offline-first prototype for comparing two readings of the same text. The challenge is that delivery differences can be subtle: a pause may be too long, a word may be rushed or stretched, or its recorded level may change. A useful tool should point to the time region and show the evidence, while leaving the final judgment to a person.”

### 0:25–0:55 — Solution and impact

**Show:** Dashboard overview and the reference/take selectors.  
**Say:**

“Our prototype compares a reference recording with a take, aligns the words, measures timing and acoustic differences, and highlights candidate regions for review. The goal is to make comparison more concrete and repeatable—not to grade a speaker automatically. The interface keeps the reference, take, transcript, measurements, and flagged intervals together so a reviewer can inspect what led to each result.”

### 0:55–1:35 — Methods and architecture

**Show:** Select a benchmark passage; point to the two audio players, chart, and candidate list.  
**Say:**

“The workflow begins with a same-text pair. Word boundaries come from the supplied Montreal Forced Aligner alignments for the archive; the live comparison can use forced alignment, with a bounded dynamic-time-warping fallback when needed. We compare word duration, inter-word gaps, recorded energy, and available pitch and spectral features. Threshold-based localization proposes intervals, and a separate paired classifier recognizes the known digital edit type and programmed strength. The classifier’s type filter is used only on this benchmark, where each edited clip contains one programmed edit.”

### 1:35–2:15 — Product demonstration

**Show:** Play a short part of the reference and take. Move the cursor over the chart, then show the candidate interval details. Toggle the one-edit filter off and on once.  
**Say:**

“Here, the two audio players let us listen to the pair. The chart places the reference and take measurements by word position, and the highlighted region marks a candidate for listening. The event details show the estimated interval and the acoustic measure behind it. With the benchmark filter enabled, candidates of other edit types are removed. Turning it off shows the detector on its own. This reduces cross-type false alerts on the controlled benchmark. For a newly uploaded recording, the filter stays off because real speech can contain several kinds of differences at once.”

### 2:15–2:55 — Results

**Show:** README results table or dashboard metrics. Pause so the figures can be read.  
**Say:**

“On the fixed Obama-speaker digital-edit test split, the separate classifier identified the programmed edit type in 104 of 104 clips, and the programmed strength in 101 of 104. For interval localization at intersection-over-union of at least 0.5, the detector alone reached 42.3 percent precision, 60.4 percent recall, and 49.8 percent F1. The benchmark-only combined pipeline reached 65.2 percent precision, the same 60.4 percent recall, and 62.7 percent F1. It reduced candidate intervals from 137 to 89 while retaining the 58 correctly localized edits.”

### 2:55–3:30 — Validation, limitations, and close

**Show:** README limitation paragraph, then the GitHub repository title.  
**Say:**

“These results come from controlled digital edits of archival speech. The same fixed test set had been used for earlier classifier reporting, so this combined score is exploratory, not a new independent benchmark. The system has not established accuracy on natural speech, and the timings and severity levels have not been validated by independent human raters. Its role today is to support review by showing measurable differences and candidate times. The prototype, source code, dataset documentation, evaluation, and reproducible pipeline are available in the project repository. Thank you.”

## Editing checklist

- Use simple cuts between the opening, dashboard, results, and repository views.
- Add captions for the key result: “62.7% F1 · 65.2% precision · 60.4% recall” and “controlled digital-edit benchmark.”
- Keep the test-set and natural-speech limitations audible; do not cover them only with small text.
- End on the repository page with the project name and GitHub URL visible for a few seconds.
