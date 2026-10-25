# ROI full-view close-up retrain

## Problem

The previous ROI checkpoint was trained on far targets, cropped targets, and
close-ups up to 90% **canvas** width. Because the browser canvas has 10% blank
padding on each side, that usually caps the visible marker pattern well below
the full frame width (roughly 65–80%, depending on canvas aspect and pose). When a phone screen is held close enough that the marker
pattern covers almost the whole camera view, the old model predicted almost
nothing: on a held-out 197-frame full-view suite its center containment was
**0.078** and mask IoU **0.073**. The presence head classified almost all of
those frames as negative (presence accuracy 0.031 on the full-view group).

## Dataset change

`RoiDatasetConfig` now has a dedicated **full-view** branch. It fits the
projected **marker hull** directly instead of the padded browser canvas:

- `full_view_fraction = 0.60` (fraction of non-cropped positives),
- marker-hull width target 95–115% of the frame width,
- marker-hull height at least 80% of the frame height, so 16:9 canvases are
  scaled up enough to avoid a large blank band (this can push the realized
  width slightly above the sampled target),
- ±3.5% center jitter, so the target usually extends past two or more frame
  edges,
- the usual perspective, roll, defocus, frame blur, camera sharpening, canvas
  aspect (4:3, 16:9, 1:1), board-gray, and color randomization.

The other positive branches remain:

- 20% cropped targets (one side past the edge, at least 8% visible),
- 20% close-ups at 75–95% canvas width,
- the remaining regular targets at 15–72% width.

The training run used 3,000 composites with 45% negative frames. Of the
positives, approximately 20% were cropped, 48% full-view, 6% close-up, and 26%
regular. Full-frame horizontal flips doubled the training split, and the two
annotated real frames were repeated 8×.

Because the mask is clipped at the frame border for full-view scenes, the model
cannot use a background margin there. Negatives remain important: the full-view
branch makes large masks correct, not any large mask.

## Training

```bash
cd python
uv run python scripts/train_roi.py \
  --samples 3000 --epochs 80 --batch-size 32 \
  --channels 16,24,24,24 --input-width 256 --input-height 192 \
  --natural-fraction 0.45 \
  --cropped-fraction 0.20 --closeup-fraction 0.20 --full-view-fraction 0.60 \
  --full-view-min 0.95 --full-view-max 1.15 --full-view-min-height 0.80 \
  --real-positive-repeat 8 --seed 7
```

The architecture is unchanged from the browser model: a four-level U-Net with
73,986 parameters, 256×192 input, 128×96 output stride 2, false-negative weight
10, and Tversky weight 1.0. The run took about 41 minutes on Apple M3 Pro MPS.

## Held-out results

The evaluation suite is a fresh 600-sample mixed set (seed 1001): 197
full-view, 30 close-up, 103 cropped, 90 regular, 71 heavy-blur, and 180 negative
frames. It is generated after training and is not the training/validation/test
split.

![Old vs retrained full-view mask](../python/notebooks/blob_lab_data/full_view_before_after.png)

| branch | old containment | new containment | old IoU | new IoU |
|---|---:|---:|---:|---:|
| full view | 0.0778 | **1.0000** | 0.0727 | **0.9871** |
| close-up | 1.0000 | 0.9999 | 0.9746 | 0.9512 |
| cropped | 0.9998 | 0.9996 | 0.9678 | 0.9426 |
| regular | 0.9838 | 0.9776 | 0.9381 | 0.9041 |
| heavy blur | 0.5774 | **0.9972** | 0.5548 | **0.9554** |
| negative | — | — | 1.0000 | 0.9944 |

Overall containment went from 0.5639 to **0.9951**, and presence accuracy from
0.6800 to **0.9983**. The previous checkpoint's heavy-blur number was low
because roughly half of the heavy-blur frames in that suite were also
full-view. The new model keeps mean negative mask area at 2.2×10⁻⁵ (about 7
pixels per negative frame on average) and still produces an empty mask on the
training-split negatives.

On the training test split (250 positives / 200 negatives, with the new
distribution):

| branch | containment | IoU | samples |
|---|---:|---:|---:|
| full view | 1.0000 | 0.9864 | 120 |
| close-up | 1.0000 | 0.9519 | 16 |
| cropped | 0.9997 | 0.9429 | 51 |
| regular | 0.9776 | 0.9126 | 63 |
| heavy blur | 0.9989 | 0.9625 | 63 |
| negative mask area | 0.0000 | — | 200 |

Overall test center containment is 0.9943, mask IoU 0.9759, presence accuracy
0.9956, and area ratio 1.022.

## Real full-view crops

The two annotated real photos were cropped to their hand-authored mask bounding
boxes and stretched to 640×480, which approximates holding the phone close
enough that the pattern fills the view:

| frame | old presence | old IoU | new presence | new IoU |
|---|---:|---:|---:|---:|
| `real_positive_01` | 0.005 | 0.000 | **1.000** | **0.916** |
| `real_positive_02` | 0.103 | 0.503 | **1.000** | **0.913** |

The real masks are clipped at the crop border, matching the full-view training
setup.

## Browser export

The checkpoint was exported with:

```bash
uv run python scripts/export_roi_noconcat_onnx.py
```

The ONNX graph has no `Concat` nodes, matches PyTorch to 1.3×10⁻⁷, and works at
all browser input heights (96–224 px in 16 px steps). The exported file is
`public/models/roi_tiny_no_concat.onnx` (303,700 bytes, SHA-256
`1207adee2f7fee9bc8caa67483bfaa75902abc112ff57e441fdf1422e1a2641e`).

`model-manifest.json` now points at a `models-v2` release with that hash. Upload
the ONNX under the `models-v2` tag before a fresh checkout or CI run fetches
weights; a matching local file is reused without a download.

## Files

- Checkpoint: `python/notebooks/blob_lab_data/roi_net_best.pt`
- Metrics: `python/notebooks/blob_lab_data/roi_metrics.json`
- Before/after comparison: `python/notebooks/blob_lab_data/roi_view_eval.json`
- Old checkpoint backup: `python/notebooks/blob_lab_data/roi_net_best_before_fullview.pt`
- Old ONNX backup: `python/notebooks/blob_lab_data/roi_tiny_no_concat_before_fullview.onnx`
- Full-view previews: `python/notebooks/blob_lab_data/roi_fullview_candidate_preview/`
- Evaluation overlays: `python/notebooks/blob_lab_data/roi_view_eval_after_preview/`
