# Manga Inpainting Pipeline — Instructions & Checkpoint

## Last Updated
**Branch**: `feature/inpaint-regression-repair-iteration`  
**Last Commit**: ce8f2a5 (2 weeks ago, "Update minimal pipeline inpaint repair")  
**Status**: IN PROGRESS — Multi-sample regression testing & vision-guided iteration

---

## Project Overview

**Free Manga Translator** is an offline-first manga/manhwa/manhua page translator with an **8-stage local pipeline** and a Chrome/Brave browser extension.

### Pipeline Stages (Steps 1-8)
1. **Source Text Detection** (in extension/backend)
2. **Layout Constraint Building** (in extension/backend)
3. **Text Translation** (in extension/backend)
4. **Inpainting** (Step 4 — `run_step4_inpaint.py`) — **CURRENT FOCUS**
5. **OCR** (Step 5 — `run_step5_ocr.py`)
6. **Layout Export** (Step 6 — `run_step6_layout.py`)
7. **Translation** (Step 7 — `run_step7_translate.py`)
8. **Typesetting** (Step 8 — `run_step8_typeset.py`)

### Branch Status
- **main**: 2 commits ahead of feature branch
- **feature/inpaint-regression-repair-iteration**: 2 commits behind main, active work branch

---

## Current Mission: Single Kaggle Brain + Local Orchestration

### Architecture Goal
Use **one Kaggle notebook** running **Qwen3.6-27B-v2-GGUF with Vision** (27B params, qwen35 arch, GGUF format) as the autonomous decision brain, while all pipeline execution, file I/O, cropping, logging, and state persistence happen **locally**.

### Key Constraints
- **Kaggle**: 32GB RAM total (dual T4 GPU)
- **Local**: 8GB VRAM (sufficient for vision eval, NOT for 27B inference)
- **Model Choice**: `Qwopus3.6-27B-v2-GGUF` by Jackrong (HuggingFace)
  - **Model Size**: 27B parameters
  - **Architecture**: qwen35
  - **Chat Template**: Available
  - **GGUF Support**: Yes (Q2_K 10.9GB, Q3_K_M 12.6GB, Q4_K_M 16.8GB, Q5_K_S 19GB, Q6_K 22.1GB, Q8_0 28.6GB)
- **Vision**: mmproj model required for vision tasks
- **No Cross-Device Sharding**: Cannot distribute 27B across Kaggle+Local due to internet latency

---

## Problem Statement: Inpainting Regression Issues

### Context
The **Step 4 inpainting** stage is designed to remove source text (Japanese/Chinese/Korean) from manga pages and reconstruct the background artwork cleanly. It handles:
- **Bubble text**: Text inside speech bubbles (uses ONNX LaMa on local crops)
- **Floating text**: Text on flat/paper backgrounds or detailed art
- **Background types**: Flat, halftone/screentone, detailed line art, etc.

### Observed Issues
Across a diverse set of 20+ test samples (Japanese, Chinese, Korean sources):
1. **Chinese/Korean samples**: Text removal often leaves visible "ghosts" or fails to reconstruct screentone/halftone backgrounds properly
2. **Screentone regions**: Hard-coded heuristics fail for certain grayscale gradient patterns
3. **White bubble residue**: Some text-removal masks are too conservative, leaving faint stroke artifacts
4. **Floating text**: Detection heuristics miss certain low-contrast text blocks
5. **Tone matching**: Post-inpainting tone/texture doesn't always match surrounding context

### Sample Evidence (User-Provided Prompts — Verbatim)
> "Test on 20+ diverse samples: Japanese shonen (bright), Korean webtoon (grayscale), Chinese manhua (screentone heavy), etc."
>
> "For each problematic sample:
> - Crop exact problem region (e.g., 512×512 around failing text box)
> - Feed crop to Kaggle vision brain with prompt: 'Does this region show clean background reconstruction, or are there text ghosts, uneven tones, or artifacts? If issues exist, describe exact pixel coordinates and nature of defect.'
> - Log brain verdict: RESOLVED / PARTIAL / STILL_FAILING / NEW_ISSUE
> - If STILL_FAILING or PARTIAL: identify exact code path (e.g., `_maybe_fill_screentone_background` line 789) and propose generic fix
> - Apply fix, re-run pipeline on ALL samples (not just failing one), re-evaluate with vision crops
> - Iterate until all 20+ samples pass"

---

## Workflow Design: Vision-Guided Iterative Refinement

### High-Level Loop
```
1. Local orchestrator runs full 8-stage pipeline on sample set
2. For each output image:
   a. Compare to reference (source image)
   b. Detect problem regions (via heuristics or manual annotation)
   c. Crop problem region (e.g., 512×512 px around text box)
   d. Send crop + context to Kaggle brain (Qwen3.6-27B-v2 + Vision)
   e. Brain returns verdict: RESOLVED / PARTIAL / STILL_FAILING / NEW_ISSUE + exact defect description
3. If any sample fails:
   a. Identify exact code path causing failure (e.g., function + line)
   b. Propose GENERIC fix (not sample-specific)
   c. Apply fix locally
   d. Re-run pipeline on ALL samples
   e. Re-evaluate with vision crops
   f. Update state.md with iteration log
4. Repeat until all samples pass
5. Push to branch with patch naming (e.g., "patch-001-screentone-heuristic-fix")
```

### State Persistence
- **File**: `state.md` (local, git-ignored or tracked separately)
- **Format**:
  ```md
  ## Iteration 1
  - Sample: chinese_manhua_01
  - Issue: Screentone background not reconstructed (see crop_chinese_manhua_01_region_1.png)
  - Brain Verdict: STILL_FAILING — uneven gray gradient at (x=256, y=128, w=64, h=48)
  - Code Path: `_maybe_fill_screentone_background` line 789, ring_median cutoff too strict
  - Fix: Relax ring_median range from [90,190] to [85,200]
  - Status: APPLIED, re-running on all samples

  ## Iteration 2
  - Sample: ALL (20 samples)
  - Re-eval: 18 RESOLVED, 2 PARTIAL (korean_webtoon_03, japanese_shonen_07)
  - Next: Investigate tone-matching for korean_webtoon_03
  ```

### Logging
- **File**: `agent_log.jsonl` (append-only, one JSON object per line)
- **Schema**:
  ```json
  {
    "timestamp": "2025-01-15T10:32:45Z",
    "iteration": 1,
    "sample": "chinese_manhua_01",
    "action": "vision_eval",
    "crop_path": "crops/chinese_manhua_01_region_1.png",
    "brain_verdict": "STILL_FAILING",
    "defect_coords": {"x": 256, "y": 128, "w": 64, "h": 48},
    "defect_description": "uneven gray gradient, text ghost visible",
    "code_path": "run_step4_inpaint.py:_maybe_fill_screentone_background:789",
    "proposed_fix": "Relax ring_median range [90,190] -> [85,200]",
    "fix_applied": true
  }
  ```

---

## Technical Setup

### 1. Kaggle Notebook (Brain)
- **Model**: `Qwopus3.6-27B-v2-GGUF` (Q4_K_M or Q5_K_S quantization, ~17-19GB RAM)
- **Vision**: mmproj model for vision tasks
- **Server**: Expose REST API on Kaggle via ngrok (or Kaggle's native port forwarding if available)
  - `POST /vision_eval` — accepts image crop + prompt, returns verdict JSON
- **Input Format**:
  ```json
  {
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "prompt": "Does this region show clean background reconstruction, or are there text ghosts, uneven tones, or artifacts? If issues exist, describe exact pixel coordinates and nature of defect."
  }
  ```
- **Output Format**:
  ```json
  {
    "verdict": "STILL_FAILING",
    "defect_coords": {"x": 256, "y": 128, "w": 64, "h": 48},
    "description": "Visible text ghost: uneven gray gradient at top-left quadrant"
  }
  ```

### 2. Local Orchestrator (Python)
- **Entry Point**: `core_pipeline/python/runtime/run_extension_pipeline_server.py` (or new script `orchestrate_vision_loop.py`)
- **Dependencies**:
  - `cv2` (OpenCV): Image cropping, comparison
  - `requests`: HTTP client for Kaggle API
  - `json`: State/log persistence
  - `pathlib`: File I/O
- **Workflow**:
  1. Load sample list from `samples/` or env var
  2. For each sample:
     a. Run `run_step4_inpaint.py` (and steps 5-8 if needed)
     b. Compare output to reference (pixel diff or manual annotation)
     c. If problematic: crop region, call Kaggle brain
     d. Log verdict to `agent_log.jsonl`
  3. Aggregate results: if any STILL_FAILING/PARTIAL, prompt for fix
  4. Apply fix, re-run pipeline on all samples
  5. Repeat until convergence

### 3. File Server (Local)
- **Purpose**: Serve local output images to Kaggle brain (if ngrok is used in reverse)
- **Not Required**: If using base64 encoding for image transmission
- **Alternative**: Use ngrok on local machine to expose a Flask server, then send URL to Kaggle

### 4. State Files
- **state.md**: Human-readable iteration log (markdown)
- **agent_log.jsonl**: Machine-readable event log (JSONL)
- **crops/**: Directory for cropped problem regions (timestamped)

---

## Detailed Step-by-Step Plan (20+ Steps)

### Phase 1: Setup (Steps 1-5)
1. **Clone Repo Locally**
   ```bash
   git clone https://github.com/Lin-2352/Free-Manga-Translator.git
   cd Free-Manga-Translator
   git checkout feature/inpaint-regression-repair-iteration
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # Ensure: opencv-python, numpy, torch, onnxruntime, requests
   ```

3. **Download Models**
   - ONNX LaMa: `models/lama/`
   - Anime/Manga LaMa (optional): `models/lama/anime-manga-big-lama.pt`
   - Manga Cleaner (optional): `models/manga_cleaner/ComfyUI/models/lama/`

4. **Prepare Sample Set (20+ Samples)**
   - Japanese: `samples/japanese_shonen_01/`, `samples/japanese_shoujo_02/`, ...
   - Chinese: `samples/chinese_manhua_01/`, `samples/chinese_manhua_02/`, ...
   - Korean: `samples/korean_webtoon_01/`, `samples/korean_webtoon_02/`, ...
   - Each sample has:
     - `source.png` (original manga page)
     - `step_4_inpaint/` (output from Step 4)
     - `step_6_layout/layout_constraints.json` (text box coordinates)

5. **Setup Kaggle Notebook**
   - Create new Kaggle notebook: `manga-vision-brain`
   - Download `Qwopus3.6-27B-v2-GGUF` from HuggingFace (use Kaggle's "Add Data" or wget)
   - Load model with llama.cpp Python bindings or ctransformers:
     ```python
     from llama_cpp import Llama
     llm = Llama(
         model_path="/kaggle/input/qwopus3-6-27b-v2-gguf/Q4_K_M.gguf",
         n_ctx=4096,
         n_gpu_layers=-1,  # Use GPU
         chat_format="qwen"
     )
     # TODO: Add vision support (mmproj model)
     ```
   - Expose REST API via Flask + ngrok:
     ```python
     from flask import Flask, request, jsonify
     app = Flask(__name__)

     @app.route("/vision_eval", methods=["POST"])
     def vision_eval():
         data = request.json
         image_b64 = data["image_base64"]
         prompt = data["prompt"]
         # Decode image, run vision inference
         # Return verdict JSON
         return jsonify({"verdict": "RESOLVED", "description": "..."})

     app.run(host="0.0.0.0", port=5000)
     ```
   - Start ngrok: `!ngrok http 5000` → get public URL (e.g., `https://abc123.ngrok.io`)

### Phase 2: Baseline Run (Steps 6-10)
6. **Run Pipeline on All Samples (Baseline)**
   ```bash
   python core_pipeline/python/runtime/run_extension_pipeline_server.py --samples-dir samples --steps 4-8
   ```
   - Outputs: `samples/<sample_name>/step_4_inpaint/inpainted.png`, `step_5_ocr/`, ..., `step_8_typeset/final.png`

7. **Visual Inspection (Manual or Semi-Automated)**
   - For each sample:
     - Open `source.png` vs. `step_4_inpaint/inpainted.png` side-by-side
     - Identify problem regions (text ghosts, uneven tones)
     - Note coordinates (x, y, w, h) in `state.md`

8. **Annotate Problem Regions**
   - Create `samples/<sample_name>/problem_regions.json`:
     ```json
     [
       {"x": 100, "y": 50, "w": 200, "h": 150, "issue": "text ghost"},
       {"x": 300, "y": 200, "w": 180, "h": 120, "issue": "uneven screentone"}
     ]
     ```

9. **Crop Problem Regions**
   ```python
   import cv2, json
   from pathlib import Path

   for sample_dir in Path("samples").iterdir():
       if not sample_dir.is_dir():
           continue
       problem_path = sample_dir / "problem_regions.json"
       if not problem_path.exists():
           continue
       problems = json.loads(problem_path.read_text())
       inpainted = cv2.imread(str(sample_dir / "step_4_inpaint" / "inpainted.png"))
       for i, region in enumerate(problems):
           crop = inpainted[region["y"]:region["y"]+region["h"], region["x"]:region["x"]+region["w"]]
           crop_path = f"crops/{sample_dir.name}_region_{i}.png"
           cv2.imwrite(crop_path, crop)
   ```

10. **Send Crops to Kaggle Brain (Iteration 1)**
    ```python
    import requests, base64
    from pathlib import Path

    KAGGLE_API = "https://abc123.ngrok.io/vision_eval"

    for crop_path in Path("crops").glob("*.png"):
        with open(crop_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        response = requests.post(KAGGLE_API, json={
            "image_base64": image_b64,
            "prompt": "Does this region show clean background reconstruction, or are there text ghosts, uneven tones, or artifacts? If issues exist, describe exact pixel coordinates and nature of defect."
        })
        verdict = response.json()
        print(f"{crop_path.name}: {verdict['verdict']} — {verdict['description']}")
        # Log to agent_log.jsonl
    ```

### Phase 3: Iterative Fix (Steps 11-18)
11. **Analyze Verdicts**
    - Count: RESOLVED, PARTIAL, STILL_FAILING, NEW_ISSUE
    - If all RESOLVED: DONE, push to branch
    - Else: proceed to fix

12. **Identify Failing Code Paths**
    - For each STILL_FAILING/PARTIAL sample:
      - Trace execution in `run_step4_inpaint.py`
      - Identify exact function + line (e.g., `_maybe_fill_screentone_background:789`)
      - Examine heuristic cutoffs (e.g., `ring_median [90,190]`, `edge_density > 0.16`)

13. **Propose Generic Fix**
    - Example: "Relax ring_median range from [90,190] to [85,200] to capture wider grayscale screentones"
    - Document in `state.md`:
      ```md
      ## Iteration 1 Fix
      - **Issue**: Chinese manhua screentone not reconstructed
      - **Code Path**: `_maybe_fill_screentone_background` line 789
      - **Current**: `if not (90.0 <= ring_median <= 190.0): return False`
      - **Proposed**: `if not (85.0 <= ring_median <= 200.0): return False`
      - **Rationale**: Widens tolerance for lighter/darker screentones
      ```

14. **Apply Fix Locally**
    ```bash
    # Edit core_pipeline/python/steps/run_step4_inpaint.py
    # Line 789: Change ring_median range
    git diff run_step4_inpaint.py
    ```

15. **Re-Run Pipeline on ALL Samples**
    ```bash
    python core_pipeline/python/runtime/run_extension_pipeline_server.py --samples-dir samples --steps 4-8 --force-rerun
    ```

16. **Re-Crop Problem Regions**
    - Same as Step 9 (reuse `problem_regions.json` or update if new issues found)

17. **Re-Send Crops to Kaggle Brain (Iteration 2)**
    - Same as Step 10
    - Compare verdicts: iteration 1 vs. iteration 2

18. **Update State & Logs**
    - Append to `state.md`:
      ```md
      ## Iteration 2 Results
      - Samples tested: 20
      - RESOLVED: 18
      - PARTIAL: 2 (korean_webtoon_03, japanese_shonen_07)
      - STILL_FAILING: 0
      - Next: Investigate tone-matching for korean_webtoon_03
      ```
    - Append to `agent_log.jsonl`:
      ```json
      {"timestamp": "2025-01-15T11:05:23Z", "iteration": 2, "action": "fix_applied", "fix_description": "Relaxed ring_median range to [85,200]", "samples_resolved": 18, "samples_partial": 2}
      ```

### Phase 4: Regression Test & Push (Steps 19-22)
19. **Full Regression Test (All Samples, All Steps)**
    ```bash
    python core_pipeline/python/runtime/run_extension_pipeline_server.py --samples-dir samples --steps 1-8 --regression-mode
    ```
    - Ensure no new failures in steps 5-8 due to Step 4 changes

20. **Visual QA (Spot Check)**
    - Manually inspect 5-10 random samples end-to-end
    - Confirm final typeset output (`step_8_typeset/final.png`) looks clean

21. **Commit & Push**
    ```bash
    git add core_pipeline/python/steps/run_step4_inpaint.py
    git commit -m "patch-001: Relax screentone ring_median range [85,200] for Chinese/Korean samples"
    git push origin feature/inpaint-regression-repair-iteration
    ```

22. **Branch Naming Convention**
    - Main feature branch: `feature/inpaint-regression-repair-iteration`
    - Sub-patches (optional): `patch-001-screentone-fix`, `patch-002-tone-matching`, etc.
    - Merge to main only after all 20+ samples pass

---

## Checklist for Resuming Work

- [ ] Local repo cloned and on `feature/inpaint-regression-repair-iteration` branch
- [ ] Dependencies installed (`requirements.txt`)
- [ ] Models downloaded (ONNX LaMa, Anime LaMa, Manga Cleaner)
- [ ] Sample set prepared (20+ samples with `source.png` and `step_6_layout/layout_constraints.json`)
- [ ] Kaggle notebook created and `Qwopus3.6-27B-v2-GGUF` loaded
- [ ] Vision support (mmproj) integrated in Kaggle brain
- [ ] Flask API + ngrok running on Kaggle, public URL obtained
- [ ] Local orchestrator script (`orchestrate_vision_loop.py`) ready
- [ ] `state.md` and `agent_log.jsonl` initialized
- [ ] Baseline run completed (Step 6)
- [ ] Problem regions annotated (Step 8)
- [ ] First batch of crops sent to Kaggle brain (Step 10)
- [ ] Verdicts analyzed, fixes proposed (Steps 11-13)
- [ ] Iterative loop running (Steps 14-18)

---

## Next Immediate Actions

1. **Complete Kaggle Setup**:
   - Integrate mmproj model for vision tasks
   - Test vision inference with a sample crop
   - Confirm REST API responds correctly

2. **Write Local Orchestrator**:
   - Script: `core_pipeline/python/runtime/orchestrate_vision_loop.py`
   - Functions:
     - `run_pipeline(sample_paths, steps)` — runs steps 4-8
     - `crop_problem_regions(sample_path, regions)` — crops and saves to `crops/`
     - `call_kaggle_brain(crop_path, prompt)` — sends HTTP request, returns verdict
     - `log_verdict(sample, iteration, verdict)` — appends to `agent_log.jsonl`
     - `update_state(iteration, summary)` — appends to `state.md`

3. **Run Baseline (Iteration 0)**:
   - Execute pipeline on all 20 samples
   - Manually inspect outputs
   - Annotate first batch of problem regions

4. **Start Vision Loop (Iteration 1)**:
   - Crop regions
   - Send to Kaggle brain
   - Log verdicts
   - Identify failing code paths
   - Propose first fix

---

## References

- **Repo**: https://github.com/Lin-2352/Free-Manga-Translator
- **Branch**: `feature/inpaint-regression-repair-iteration` (2 commits behind main)
- **Model**: [Qwopus3.6-27B-v2-GGUF on HuggingFace](https://huggingface.co/Jackrong/Qwopus3.6-27B-v2-GGUF)
- **Leaderboard**: [LLM Leaderboard (AI Analysis)](https://artificialanalysis.ai/leaderboards/models?reasoning=reasoning&weights=open)
- **Step 4 Code**: `core_pipeline/python/steps/run_step4_inpaint.py`
- **Runtime Orchestrator**: `core_pipeline/python/runtime/run_extension_pipeline_server.py`

---

## Appendix: User Instructions (Verbatim)

> "Use a single Kaggle brain to drive the LLM (Qwen-based) with vision, while all code edits, logging, and pipeline orchestration happen locally."
>
> "The agent should autonomously run the 8-stage manga-pipeline, identify problematic samples, zoom into the exact regions, judge whether issues are resolved or require rework, and iteratively refine toward a universal, robust pipeline."
>
> "Maintain persistent state locally (state.md) and an ongoing, auditable log of actions (logs)."
>
> "Ensure the system can handle 20+ diverse samples, with special attention to Chinese/Korean samples where text removal and background reconstruction are challenging."
>
> "Provide a verbose, evolved plan (20+ steps) and ensure visual inspection via zoom/crop for validation."
>
> "After implementing fixes, rerun on all samples and push to a designated Git branch, with patch naming conventions, while avoiding changes to another repo you called out (the 'free manga translator' now shouldn't be touched)."
>
> "Cropping/zooming must be used to minimize tokens and maximize visual fidelity for vision evaluation."
>
> "All state and logs must be persisted (state.md, agent_log.jsonl or similar)."
>
> "Output must be reproducible, with branch naming conventions, and changes must be pushed to a persistent branch (stability-art-aware-quality-hardening, patch versions, etc.)."

---

**END OF INSTRUCTIONS.MD**
