# AGENTS.md — MLHAT Paper Reproduction (Introduction to Machine Learning)

> **Purpose:** Persistent project context and working rules for AI coding agents (Codex, Copilot, etc.). Place this file at the **root of the team's GitHub repository**. Read it before planning, generating files, coding, or editing documentation.
>
> **Current status (as of 2026-10-28):** Team of **3 students** has **selected MLHAT as its intended paper**. Registration acceptance, member-to-role mapping, exact deadlines, datasets, and reproduction scope are **not yet confirmed here**. **Do not claim these are confirmed.** This file is a project brief, not proof that code or experiments already exist.

## 1. Project identity and research question

- **Course:** Introduction to Machine Learning (Nhập môn Máy học).
- **Project:** Study, independently reimplement, reproduce and critically assess a recent published ML research paper.
- **Paper:** *Hoeffding adaptive trees for multi-label classification on data streams* — Aurora Esteban, Alberto Cano, Amelia Zafra, Sebastián Ventura (2024), *Knowledge-Based Systems*.
- **Paper:** https://arxiv.org/abs/2410.20242
- **DOI:** https://doi.org/10.1016/j.knosys.2024.112561
- **Authors' reference repository:** https://github.com/aestebant/mlhat
- **Method:** **MLHAT** = Multi-Label Hoeffding Adaptive Tree.
- **Problem:** Online/streaming multi-label classification under potentially changing data concepts. Study incremental tree growth, relationships among labels, adaptive leaf prediction, and drift adaptation.
- **Scientific objective:** Understand what is novel, reimplement the core proposed contribution from paper descriptions (not by copying the authors' code), run a meaningful subset of the paper's experiments, quantify deviations, and explain them.

### Sources of truth (priority)
1. **Actual course/Moodle instructions, grading rubric and deadlines** supplied by the students.
2. **The paper** for method details, mathematical claims, baselines, datasets and protocols.
3. Team decisions recorded in this repo (`README.md`, `docs/decisions.md`, configs).
4. Authors' repository **only for independent verification, running comparisons and debugging reference outputs; not as code to copy into the team's implementation**.

When this file conflicts with updated teacher instructions, flag the conflict and update the plan; never quietly assume this document overrules the teacher.

## 2. Scope of the FIRST deliverables (Midterm)

**There are TWO SEPARATE Moodle submissions now; final poster is later.**

### Deliverable 1 — Public GitHub repository URL
- One team representative submits **one publicly accessible repository link**.
- Repository should progressively contain the team's actual implementation, reusable experiment scripts, setup instructions, documentation, and authentic Git history.
- Submission is **an informational deadline, not the last permitted commit**: development can continue after the link submission until the final poster deadline.
- Never pretend a freshly scaffolded, empty repo is a completed reproduction.

### Deliverable 2 — Midterm PPTX (45% of course project grade)
- **Single file:** `ID.pptx`, where `ID` is the team's registered numeric ID (still unspecified).
- **20–40 slides in total**, including appendices; **under 20 MB**.
- Combines **Paper Reading/Understanding + Experimental Reproduction** (not just a summary of the article).
- **Every slide must have presenter/speaker notes, at most 500 words**, sufficient for examiners to read without oral presentation.
- Prefer **team-created diagrams, mathematical explanations, plots and tables** over dense text. Redraw rather than paste paper figures.
- Cite references **IEEE-style at the bottom of each slide that uses them**; course rules say **no standalone references slide**.
- Report actual reproduction runs compared side-by-side with paper figures/metrics, with **relative deviations** where quantitative. Explain meaningful differences.
- Every reported experimental figure/number must trace back to an actual **public W&B run** and the corresponding team code/configuration.
- **Do not fabricate results, charts, citations, runs or completion claims.**

### Later — Final poster (not current priority)
The final stage covers theoretical criticism, limitations and experiments deliberately designed to investigate those limitations, then a poster/oral poster defense. Keep future extension in mind but **prioritize the two Midterm submissions**.

## 3. Rules for scientific integrity and reproducibility

These are non-negotiable course requirements:

- **Implement the core novel MLHAT method independently from the paper's description.** Do not copy/paste/port/translate the paper authors' core source code or another team's implementation. Do not disguise copying through renaming or AI rewriting.
- The reference repository may be **read, run, and compared** strictly for validation. Report any such comparison transparently.
- Standard libraries and non-novel infrastructure (dataset loaders, standard metrics, general-purpose modules) may be reused **with clear citations/attribution in README**; always distinguish reused infrastructure from original work.
- Use **Python >= 3.9**. Standard scientific tools include `numpy`, `scipy`, `scikit-learn`, and, if necessary, `torch`, `tensorflow`, or `jax`. Explain non-standard dependencies and their purpose in README; do not secretly depend on a ready-made MLHAT implementation.
- Record random seeds, library versions, relevant hardware/software details, chosen hyperparameters and prequential/evaluation protocol so runs can be repeated.
- Every important function must have a docstring covering purpose, parameters and return value.
- Load all experiment datasets **programmatically** from Hugging Face or Kaggle at run time. Prefer existing public mirrors; if none exists, publish a properly documented dataset to one of those platforms (subject to permission) and fetch it automatically. **Avoid manual data-copy setup.**
- Maintain **public Weights & Biases (W&B)** tracking for each run: configuration, metrics, plots/artifacts as appropriate. README must map **experiment -> script/notebook -> W&B run URL -> one-line result**.
- Never commit secrets, W&B API keys, private credentials, huge local datasets, caches, or generated checkpoints by default.
- Never generate fake measurements or claim that an experiment executed without its actual log/run.

## 4. Reproduction strategy

The team must make and document a deliberate choice:

- **Full scope:** Reproduce all main experiments on the same or equivalent datasets.
- **Core scope (likely practical for a small team, NOT yet officially selected):** Reproduce **at least two of the most important experiments, including the paper's central comparison**, and replace omitted work with a **team-designed complementary analysis** (e.g., sensitivity to an important configuration).

**Before coding full experiments**, inspect the paper's experiment section and produce a protocol table containing: research claim, dataset, baseline, exact metric, configuration, seed/run count, output, and which paper figure/table it tests. Example ideas (MLHT comparisons, drift streams, sensitivity sweeps) are **candidates**, not an approved final protocol.

Use true online/test-then-train evaluation if required by the paper. Validate dataset/label format and metric definitions carefully; do not compare incompatible metric scales or different splits as though they were equivalent.

## 5. Team work allocation — proposed; confirm names later

Each member owns both a **research/code workstream** and the **associated PPTX content**. All three must read the entire paper, review one another's modules, and be able to explain any part during the final defense.

| Person | Primary scientific work | Primary coding work | Midterm slides | Extra ownership |
|---|---|---|---|---|
| **A — Core Algorithm & Theory** | Hoeffding Tree/HAT, Hoeffding Bound, node statistics, multi-label split criterion, method derivation, complexity | Incremental tree/node logic, sufficient statistics, split-selection logic; unit tests | **7–13, 18–20** (10 slides) | Validate math and theoretical claims; lead a complementary analysis with B/C help |
| **B — Multi-label, Data & Slide Lead** | Multi-label classification/prediction at leaves, datasets, prior work, metrics and evaluation | Leaf predictors, auto dataset loading, streaming evaluation, metric calculations and W&B plumbing | **1–6, 14–15, 21–22** (10 slides) | Own Experiment 1 execution with support; integrate PPTX layout/notes/citations |
| **C — Drift & Experiment Lead** | Concept drift, ADWIN/background branches, experiment reproduction and discrepancy analysis | Drift adaptation, background-tree/branch handling, integration tests and reproduction scripts | **16–17, 23–30** (10 slides) | Own Experiment 2 with support; coordinate baseline comparison/results table |

**Balance rule:** C does **not** do all integration, baselines, data, W&B, results and slides alone. B owns data/metrics/W&B and an experiment; A tests and supports implementation/debugging and an extra analysis. C coordinates the experiment report. Owners request code review from other members.

**Slide breakdown (proposal, adaptable without violating 20–40 rule):**
- B 1–6: problem, motivations, predecessor methods and limitations.
- A 7–9: gaps, contributions, notation and problem formulation.
- A 10–13: overall MLHAT workflow, incremental decision tree and splitting.
- B 14–15: multi-label leaf prediction and evaluation intuition.
- C 16–17: drift detectors and adaptive replacement.
- A 18–20: theoretical claims, complexity and worked explanation.
- B 21–22: original datasets, metrics, baselines and evaluation protocol.
- C 23–24: reproduction scope, implementation architecture and configs.
- C 25–28: **real** results from the two reproduction experiments (with B and A contributing evidence).
- C 29–30: discrepancy/additional analysis and evidence-based Midterm conclusion.

Each slide owner must submit **editable team-original diagrams/plots, concise slide content, IEEE citation(s) and speaker notes**. B integrates; all three review the completed deck.

## 6. Proposed staged workflow (NOT official deadlines)

1. **Understand:** All read abstract -> introduction -> conclusion -> related work -> experiments -> method; collectively write paper summary and open questions.
2. **Agree on architecture/protocol:** Confirm module contracts, dataset/metrics/baselines, Core-vs-Full decision, experiment success criteria, Git branching/review conventions.
3. **Scaffold:** Initialize readable repo, environment setup, README, tests, docs, experiment/config skeleton, W&B integration stub; commit incremental changes.
4. **Build a baseline end-to-end:** Load a small public dataset automatically; exercise test-then-train loop, metrics and an actual W&B run. Do not mislabel baseline as MLHAT.
5. **Reimplement and unit-test MLHAT:** A/B/C build independently owned modules, integrate progressively and verify expected behavior on tiny synthetic cases.
6. **Reproduce:** Run agreed paper comparisons; archive configs, run IDs, output tables/figures and reasons for numerical gaps.
7. **Create Midterm deck:** 20–40 slides, real evidence, speaker notes, original figures, citations; final mutual review.
8. **Submit separately:** one public GitHub URL and one correctly named `ID.pptx` on Moodle. Record actual submission deadlines when known.

## 7. Proposed repository layout (suggestion, not evidence of existing files)

```text
mlhat-reproduction/
├── AGENTS.md                 # this agent-facing project context
├── README.md                 # human-facing overview, setup, runs, W&B mapping
├── requirements.txt          # or pyproject.toml, after environment decision
├── .gitignore
├── docs/
│   ├── paper-notes.md
│   ├── method-and-math.md
│   ├── experiment-protocol.md
│   └── decisions.md
├── src/
│   └── mlhat/
│       ├── tree.py
│       ├── splitting.py
│       ├── leaves.py
│       ├── drift.py
│       └── __init__.py
├── experiments/
│   ├── data.py
│   ├── evaluate.py
│   ├── baselines.py
│   ├── reproduce_exp1.py
│   ├── reproduce_exp2.py
│   └── sensitivity.py
├── configs/
├── tests/
├── figures/
├── slides/
├── poster/                   # can remain empty until later stage
└── data/
    └── README.md             # describe remote automatic retrieval, not raw data
```

The exact file names/modules may change **after reading the paper**. No need to create empty directories/files solely for appearance. Use `.gitkeep` only if necessary. Keep the public repository clean and readable.

## 8. Coding agent behavior rules

When asked to help in this repository:

1. **First inspect the existing file tree and README**. Do not assume a scaffold, implementation, datasets, teammates' commits or successful runs already exist.
2. **Restate the requested task and its limits**, and propose a short set of steps for substantial changes. Prioritize current Midterm deliverables.
3. Distinguish clearly among **course requirement**, **paper-proven fact**, **team decision** and **agent suggestion**. Put unresolved choices in `docs/decisions.md` with `TODO` status; don't silently guess deadlines, registered team ID, final scope, metrics or experiment results.
4. For a repository scaffold request, start with **small usable infrastructure**, README, `.gitignore`, environment, tests and clear task placeholders. **Do NOT generate a supposedly finished MLHAT algorithm from vague recollection**; first reference specific paper sections/equations.
5. Make small reviewable changes, preserve other members' work, don't overwrite code or force-push without explicit permission. Prefer branches/PRs when working collaboratively.
6. Write Python with readable names, type hints where helpful, concise docstrings, explicit seeds/configuration and focused tests for new logic.
7. After code changes, execute the relevant tests and report what passed, failed or was not executable. Do not claim a W&B run or source comparison took place unless actually performed.
8. For figures/slides, produce original visualizations and traceable citations. Never invent results or fill slide tables with made-up numbers as if measured.
9. If model behavior/equations are unclear, **state the ambiguity**, identify the paper section that needs verification and suggest a targeted small test; don't invent a scientific definition.
10. For substantial work, close with: **files changed, commands/tests run, actual result, remaining risks, and next tasks by A/B/C**.
11. Communicate to the team in **Vietnamese** by default; keep technical identifiers, Python APIs and paper terminology in standard English.

## 9. What a successful Midterm hand-off looks like

- [ ] Registration/paper and team ID confirmed on Moodle.
- [ ] Public repo and README are accessible; 3 members have identifiable work/history.
- [ ] Independent MLHAT implementation runs for the agreed scope and has meaningful tests.
- [ ] Datasets are fetched programmatically; experiment environment/seeds documented.
- [ ] At least 2 key reproduction experiments including central paper comparison if Core scope; complementary analysis included.
- [ ] Each result is linked to a public W&B run and an executable script/notebook.
- [ ] Reproduced vs published metrics shown together; important deviations analyzed.
- [ ] PPTX is 20–40 slides, <20 MB, correct `ID.pptx` name, with notes and per-slide IEEE references.
- [ ] Original, team-created figures; each author reviewed their section and the full deck.
- [ ] GitHub URL and PPTX separately submitted before their actual Moodle deadlines.

## 10. Example prompts to use with the agent

**First session (safe repository initialization):**
> Read AGENTS.md carefully. Our 3-person group has selected the MLHAT paper and is just starting. Inspect the current repository. Propose the smallest clean public GitHub structure for the Midterm phase. After showing the proposed tree, create README, Python environment setup, .gitignore, documentation templates, and starter tests if authorized. Do not pretend MLHAT or any reproduction experiment is already complete. Explain what A, B, and C should do next.

**Later, to start implementation:**
> Read AGENTS.md and the paper method section. Help A design the MLHAT node/splitting interfaces, with mathematical assumptions cited by section, unit tests on small synthetic streams, and integration interfaces for B (leaf prediction/evaluation) and C (drift adaptation). Don't copy the authors' code. First identify any missing method specifications.

**Later, for Midterm experiments:**
> Read AGENTS.md, the team's experiment protocol, and the current code. Check which paper experiments are reproducible. Propose a Core-scope matrix (dataset x baseline x metric x configuration) and implement an auditable script + W&B logging for only the approved experiment. Never invent run results.

---

*Maintained by the MLHAT student team. Update this file when team responsibilities, Moodle deadlines, paper interpretation, approved datasets or reproducibility decisions change.*
