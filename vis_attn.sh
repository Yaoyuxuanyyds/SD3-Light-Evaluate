
# --------------- User Config -----------------
MODEL_PATH="/inspire/hdd/project/chineseculture/public/yuxuan/base_models/Diffusion/sd3"
# PROMPT="A cozy living room with a wooden coffee table, a white mug, a small potted plant, and stacked books, warm natural light from a window."
PROMPT="A wooden coffee table is placed in the center, with a white mug on the left, a small green succulent plant slightly to its right, and three stacked books on the far right. In the background, a beige sofa sits against the wall, softly illuminated by natural light from the nearby window."
IMAGE_PATH="/inspire/hdd/project/chineseculture/public/yuxuan/SD3-Residual/test_resized.jpg"
OUTPUT_DIR="/inspire/hdd/project/chineseculture/public/yuxuan/SD3-Residual/logs/results/attn_vis_complex_prompt/attn_vis_out-residual"


# Diffusion & Visualization
TIMESTEP=900
HEIGHT=1024
WIDTH=1024
LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
# TOKEN_WORDS="A cozy living room with wooden coffee table white mug small pot plant stacked books light from window"
TOKEN_WORDS="wooden coffee table center white mug left small green succulent plant right background"


RES_ORIGIN=1

RES_TARGET="$(seq -s ' ' 4 13)"

RES_WEIGHT="$(printf '0.1 %.0s' $(seq 4 13))"



ALPHA=0.5
CMAP="jet"
DEVICE="cuda"
SEED=42
# ---------------------------------------------

echo "[INFO] Running visualization"

python visualize_sd3_cross_attention_pro.py \
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
  --residual_target_layers $RES_TARGET \
  --residual_origin_layer $RES_ORIGIN \
  --residual_weights $RES_WEIGHT \


echo "[DONE] All attention maps saved under: ${OUTPUT_DIR}"
