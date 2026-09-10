# CS55 Cross-Modal Evidence Chain Demo

This package is the runnable CS55 prototype built on top of the project-provided `libevchain` evidence-chain library.

## What the system does

`Storyboard → Animatic → Final → CLIP embeddings → similarity matrices → sequence-aware dynamic-programming matching → evidence-chain metrics → libevchain Pipeline → JSON outputs`

The current metrics are:
- **Integrity / Coherence**: mean of the two stage-level sequence-aware CLIP similarities.
- **Confidence**: `1 - |Storyboard→Animatic mean - Animatic→Final mean|`, clipped to `[0, 1]`.
- **Coverage**: average of storyboard→animatic shot coverage and animatic→final coverage.
- **Completeness**: number of available final keyframes divided by the expected final keyframe count.

## Folder structure

```text
CS55_Evidence_Chain_Demo/
├── README.md
├── requirements.txt
├── pyproject.toml
├── run_demo.py
├── demo.ipynb
├── config.sample.json
├── config.sollevante.example.json
├── src/
│   ├── cs55_demo/          # CS55 application code
│   └── libevchain/         # project-provided evidence-chain library
├── docs/
│   └── json_format.md
├── data/
│   └── sample/             # tiny smoke-test media
├── cache/                  # generated CLIP embedding cache
├── outputs/                # generated JSON results
└── test/                   # original libevchain tests + CS55 smoke test
```

## Quick start

From Terminal:

```bash
cd CS55_Evidence_Chain_Demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py --config config.sample.json
```

The first run downloads `openai/clip-vit-base-patch32` from Hugging Face and creates an embedding cache. Later runs reuse the cache.

For the live client demo, run the chosen dataset once beforehand so `cache/<dataset>_embeddings.pt` already exists. This avoids relying on network access during the presentation.

## Fast package validation (no CLIP download)

After dependencies are installed, you can check that the package wiring, dynamic-programming matching and libevchain integration work without downloading CLIP:

```bash
PYTHONPATH=src python test/test_cs55_smoke.py
```

This test uses a deterministic dummy embedding only for packaging validation. The real demo uses CLIP.

## Run with Sol Levante

1. Copy `config.sollevante.example.json` to `config.sollevante.json`.
2. Replace `/path/to/SolLevante/...` with the local paths on the demo computer.
3. Make sure the processed folders contain:
   - storyboard panel PNGs,
   - animatic keyframe PNGs,
   - final keyframe PNGs.
4. Run:

```bash
python run_demo.py --config config.sollevante.json
```

The previously validated Sol Levante pipeline produced approximately:

```text
Integrity / Coherence: 0.7653
Confidence:             0.9499
Coverage:               0.8714
Completeness:           1.0000
Storyboard → Animatic:  0.7903
Animatic → Final:       0.7402
Artefacts:              223
Relationships:          222
```

## Notebook demo

Open `demo.ipynb` and run the cells from top to bottom. It uses relative paths and the packaged `src/` directory, so it does not depend on `/Users/.../COMP3988_Evidence_Chains-master`.

## Outputs

Each run writes:
- `outputs/final_evidence_chain_result.json` — score/metric summary.
- `outputs/evidence_chain_bundle.json` — the serialised libevchain evidence-chain graph.

## Important note about preprocessing

This package starts from **already extracted storyboard panels and video keyframes**. The full Sol Levante preprocessing notebooks are separate from this demo package. To use a completely new raw dataset, first extract storyboard panels and keyframes into the three processed directories, then run this package.

## libevchain integration notes

`src/libevchain/` is based on the COMP3988 project-provided library. The packaged copy contains only small compatibility/runtime fixes encountered during integration; see `PATCHES.md`.
