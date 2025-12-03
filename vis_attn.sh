#!/bin/bash
# Visualize SD3 cross-attention maps with an optional LoRA checkpoint
# Usage:
#   bash visualize_with_lora.sh <LoRA_CKPT_PATH>

# --------------- User Config -----------------
MODEL_PATH="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3"
# PROMPT="A cozy living room with a wooden coffee table, a white mug, a small potted plant, and stacked books, warm natural light from a window."
PROMPT="A wooden coffee table is placed in the center, with a white mug on the left, a small green succulent plant slightly to its right, and three stacked books on the far right. In the background, a beige sofa sits against the wall, softly illuminated by natural light from the nearby window."
IMAGE_PATH="/inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3/test_resized.jpg"
OUTPUT_DIR="./results/attn_vis_complex_prompt/attn_vis_out"


# Diffusion & Visualization
TIMESTEP=750
HEIGHT=1024
WIDTH=1024
LAYERS="0 2 4 6 8 10 12 14 16 18 20 22 23"
# TOKEN_WORDS="A cozy living room with wooden coffee table white mug small pot plant stacked books light from window"
TOKEN_WORDS="wooden coffee table center white mug left small green succulent plant right background"
ALPHA=0.5
CMAP="jet"
DEVICE="cuda"
SEED=42
# ---------------------------------------------

echo "[INFO] Running visualization with LoRA: ${LORA_CKPT}"

python visualize_sd3_cross_attention.py \
  --model "${MODEL_PATH}" \
  --prompt "${PROMPT}" \
  --image "${IMAGE_PATH}" \
  --timestep-idx ${TIMESTEP} \
  --layers ${LAYERS} \
  --token-words ${TOKEN_WORDS} \
  --height ${HEIGHT} --width ${WIDTH} \
  --output "${OUTPUT_DIR}" \
  --alpha ${ALPHA} \
  --cmap ${CMAP} \
  --device ${DEVICE} \
  --seed ${SEED} \


echo "[DONE] All attention maps saved under: ${OUTPUT_DIR}"


#   python visualize_sd3_cross_attention.py \
#       --model /inspire/hdd/project/chineseculture/public/yuxuan/base_models/diffusion/sd3 \
#       --prompt "A cozy living room with a wooden coffee table, a white mug, a small potted plant, and stacked books, warm natural light from a window." \
#       --image /inspire/hdd/project/chineseculture/public/yuxuan/REPA-sd3/test_resized.jpg \
#       --timestep-idx 250 \
#       --layers 0 2 4 6 8 10 12 14 16 18 20 22 23 \
#       --token-words A cozy living room with wooden coffee table white mug small pot plant stacked books light from window \
#       --height 1024 --width 1024 \
#       --output ./attn_vis_out
