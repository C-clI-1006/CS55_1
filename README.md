# CS55_1_demo

Runnable CS55-1 cross-modal evidence-chain prototype built **on top of the project-provided `libevchain` framework**.

## System flow

`Storyboard → Animatic → Final → CLIP embeddings → cosine similarity → sequence-aware dynamic-programming matching → evidence-chain metrics → libevchain Pipeline → JSON outputs`

## Important project structure

```text
CS55_1_demo/
├── CS55_1_demo.ipynb        # easiest live-demo entry point
├── run_demo.py              # command-line entry point
├── config.sample.json       # packaged sample
├── config.sollevante.example.json
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── cs55_demo/           # CS55-1 implementation
│   └── libevchain/          # COMP3988-provided source, kept unchanged
├── docs/
├── data/sample/
├── cache/
├── outputs/
└── test/
```

The separation is intentional: **`libevchain` is the shared evidence-chain framework; `cs55_demo` is the CS55-1 system that uses it.**

## libevchain compatibility

The source files under `src/libevchain/` are copied from the supplied COMP3988 version without source-code edits. The CS55-specific workarounds and custom relationship types live only under `src/cs55_demo/`.

See `docs/LIBEVCHAIN_ORIGINAL.md` for details and source-file hashes.

## Metrics used by this demo

These definitions match the main measurement section in `CS55_CrossModal_Coherence-step 2`:

- **Chain Coherence**: for each complete Storyboard → Animatic → Final chain, average the two relationship similarities; then average across all storyboard chains.
- **Confidence**: for each chain, `1 - |Storyboard→Animatic similarity - Animatic→Final similarity|`, clipped to `[0, 1]`; then average across chains.
- **Evidence Coverage**: average of (unique animatic shots used / all animatic shots) and (unique final shots used / all final shots).
- **Completeness**: each final shot used by the evidence chain is `available = 1.0`; each unused final shot is `partial = 0.5`; then average across expected final shots.
- **Stage similarities**: mean sequence-aware similarity for Storyboard→Animatic and Animatic→Final separately.

For the validated Sol Levante run, the notebook reported approximately:

```text
Chain Coherence:              0.7701
Confidence:                   0.9145
Evidence Coverage:            0.5972
Completeness:                 0.7258
Storyboard → Animatic:        0.7903
Animatic → Final:             0.7402
EvidenceChain artefacts:      223
Evidence relationships:       222
```

## Quick start

In Terminal:

```bash
cd CS55_1_demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py --config config.sample.json
```

The first real CLIP run downloads `openai/clip-vit-base-patch32`. Later runs can reuse the embedding cache in `cache/`.

## Quick package check without downloading CLIP

```bash
PYTHONPATH=src python test/test_cs55_smoke.py
```

This uses a deterministic dummy embedder only to verify the package wiring, matching, scoring and `libevchain` integration.

## Jupyter live demo

Open:

```text
CS55_1_demo.ipynb
```

The default config is `config.sample.json`. Run the cells from top to bottom.

## Sol Levante demo

1. Copy `config.sollevante.example.json` to `config.sollevante.json`.
2. Replace the placeholder paths with paths on the demo computer.
3. Ensure the processed directories already contain storyboard panel PNGs, animatic keyframe PNGs and final keyframe PNGs.
4. Run:

```bash
python run_demo.py --config config.sollevante.json
```

or set `CONFIG_FILE = "config.sollevante.json"` in `CS55_1_demo.ipynb`.

## Outputs

Each run writes:

```text
outputs/final_evidence_chain_result.json
outputs/evidence_chain_bundle.json
```

The first file contains the final metrics. The second is the serialised EvidenceChain JSON.

## Preprocessing note

This distributable demo starts from already-extracted storyboard panels and video keyframes. For a completely new raw dataset, preprocess the storyboard/video files into the three processed image directories before running the evidence-chain pipeline.
