# PRIMT

**PR**eference-based re**I**nforcement learning with **M**ultimodal feedback and **T**rajectory synthesis from foundation models (NeurIPS 2025).

---

## Requirements

- Python 3.9, PyTorch, a CUDA GPU
- `numpy`, `scipy`, `pandas`, `hydra-core`, `pyyaml`, `tqdm`, `termcolor`
- `gym`, `metaworld` (MuJoCo) for the environments
- `ruptures` (keyframe change-point detection)

Additionally required only for the real foundation-model path (`backend=openai`):

- `openai` (API client)
- `clip` + `Pillow` (VLM keyframe embedding / rendering)
- A working MuJoCo off-screen renderer (GL)

The simulated path (`backend=scripted`) needs none of these extras and makes **no network calls**.

---
 
## Selecting the feedback backend (the key hyperparameter)

The single switch is `primt.fm.backend`:

| `primt.fm.backend` | Feedback source | API calls | Renderer / CLIP | Use for |
|--------------------|-----------------|-----------|-----------------|---------|
| `openai`           | Real LLM/VLM via the API | yes | loaded | **The actual PRIMT method** |
| `scripted`         | Simulated LLM/VLM evaluators | none | not loaded | API-free pipeline validation, fast iteration |

```bash
python train_PEBBLE_PRIMT.py primt.fm.backend=openai    # PRIMT
python train_PEBBLE_PRIMT.py                            # simulated (default)
```

### ⚠️ On the `scripted` backend

**Running the full PRIMT pipeline against the API can be expensive**, due to the multimodal nature of our method. We therefore ship `backend=scripted` as a **cost-free simulation of the feedback source**, so the rest of the pipeline can be developed, debugged and regression-tested without spending API budget.

---

## Reproducing

```bash
# Requires OPENAI_API_KEY
python train_PEBBLE_PRIMT.py primt.fm.backend=openai seed=12345

# API-free pipeline check 
python train_PEBBLE_PRIMT.py primt.fm.backend=scripted seed=12345
```

