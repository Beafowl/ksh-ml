# sdvx-ml

Machine-learning tooling for generating SDVX/KSH charts, kept separate from the
editor repo (`../ksm-editor`).

## Phase 1 — dataset builder (this folder)

Matches the converted KSH charts against the official `music_db.xml` (per-difficulty
level + radar labels: notes / peak / tsumami / tricky / hand-trip / one-hand, 0–200),
parses every chart into a compact JSON representation, and samples an
onset-strength value per 1/16-note grid cell from each chart's audio.

```
python -m sdvx_dataset.build ^
    --charts "C:/Users/Berkb/Downloads/SDVX" ^
    --db "D:/Spiele/KFC-2026020300/contents/data/others/music_db.xml" ^
    --out out
```

Outputs in `out/`:
- `charts.jsonl.gz` — one JSON object per chart (meta, labels, notes/lasers, onsets)
- `report.txt` — sanity report (match rates, radar/level distributions, failures)
- `unmatched.json` — template for manual title aliases; fill values with music ids
  and save as `aliases.json` next to this README, then rebuild.

Flags: `--no-audio` (fast pass without onset extraction), `--limit N` (smoke run),
`--workers N`.

`tools/validate_parser.py` cross-checks the Python KSH parser against the editor's
`ksh.js` on a random sample of real charts.

## Phase 2 — tokenizer + model + training (`sdvx_model/`)

Chart <-> token round-trip is validated on the real dataset
(`python tools/test_roundtrip.py` — tokens -> chart -> .ksh -> reparse must be
identical). Conditioning: level + the six radar axes (bucketed 0-10) + BPM as
prefix tokens, with classifier-free-guidance dropout so the sliders steer
generation.

On the training PC (RTX 3070):

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy tqdm

# copy this folder + out/charts.jsonl.gz over, then:
python -m sdvx_model.train --data out/charts.jsonl.gz --out runs/base
```

Defaults: ~14M-param model (8 layers, d=384), ctx 2048, batch 16 x accum 2,
6000 steps, bf16 — roughly 45-90 min on a 3070. Watch the printed val loss;
`runs/base/best.pt` tracks the best one. Resume an interrupted run with
`--resume runs/base/last.pt`.

Generate charts from a checkpoint (sliders are 0..1):

```
python -m sdvx_model.sample --ckpt runs/base/best.pt --out gen.ksh ^
    --level 17 --notes 0.7 --peak 0.6 --tsumami 0.5 --tricky 0.1 ^
    --hand-trip 0.2 --one-hand 0.4 --bpm 180 --measures 64 --guidance 2.0
```

Drag the resulting .ksh into the editor to inspect it. `--guidance 1.0`
disables slider steering (pure unconditional style); 2-3 pushes harder.

## Later phases (planned)

3. Audio conditioning (per-measure onset features are already in the dataset)
4. ONNX export + in-editor generation dialog
