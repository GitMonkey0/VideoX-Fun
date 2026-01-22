# HL JSON → NPZ Converter

This tool converts per-frame hand landmark JSON to the `.npz` format expected by the HL pipeline.

## Input JSON (per frame)
Each item should contain:
- `frame_index`
- `multi_hand_landmarks` (list of hands, each with 21 landmarks in image coords)
- `multi_hand_world_landmarks` (list of hands, each with 21 landmarks in world coords)
- `multi_handedness` (list of `{label, score}`)

## Output NPZ
- `hl_ids`: shape `[F, J]` (int64)
- `hl_dirs`: shape `[F, J, 3]` (float32)

`J=20` for single hand, `J=40` for two hands.

## Usage
```bash
python scripts/wan2.1_fun/hl_json_to_npz.py \
  --input /path/to/hand.json \
  --output /path/to/hand.npz \
  --use_world \
  --two_hands
```

Convert a directory of JSON files:
```bash
python scripts/wan2.1_fun/hl_json_to_npz.py \
  --input /path/to/json_dir \
  --output /path/to/out_dir \
  --use_world
```

## Notes
- `--dir_mode center` maps each vector to one of 26 cube directions (default).
- `--dir_mode raw` keeps unit vectors directly.
- Missing hands are zero-filled.
