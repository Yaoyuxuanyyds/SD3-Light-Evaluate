# LightSD3 集成说明

本文档说明如何在本项目中使用 **LightSD3Pipeline**（自定义层数的 Stable Diffusion 3）进行推理，并加载保存的 EMA 权重。主要改动集中在 `sampler.py` 及各采样脚本的参数对齐。

## 改动概览

- `sampler.py`：  
  - 新增 `sd3_variant` 参数（`sd3` / `light`），支持从 `LightSD3Pipeline` 载入自定义 SD3 结构。  
  - 新增 `ema_ckpt_path`，可在加载主 transformer 权重后再叠加 EMA 权重（仅作用于 transformer）。  
  - 自动兼容不同 checkpoint key 格式（`transformer.xxx`、嵌套 `state_dict` 等）。  
  - LightSD3 模式下会使用 `sd3_light.py` 中的 `LightSD3Pipeline.load_from_pretrained` 读入保存目录，不支持 8bit 文本编码器。
- 采样脚本（`sample.py`、`generate_t2i.py`、`generate_dpg.py`、`generate_geneval.py`）与 `train_lora.py`：  
  - 新增 `--sd3_variant`、`--ema_ckpt` 参数并传递给 `SD3Euler`，在所有入口保持一致的用法。

## 使用 LightSD3 生成图片

1. **准备模型目录**  
   - `model_key` 指向 `LightSD3Pipeline.save_pretrained` 生成的目录（包含 `model_index.json`、`transformer/` 等子目录）。  
   - 如果有 EMA，提供对应的 `*.pt`/`*.pth` 路径，内部键可以是 `transformer.xxx` 或已去前缀的 state dict。

2. **单张/数据集采样（sample.py）**  
   ```bash
   python sample.py \
     --model sd3 \
     --sd3_variant light \
     --model_key /path/to/light_sd3_dir \
     --ema_ckpt /path/to/ema_transformer.pth \
     --prompt "a castle over clouds" \
     --save_dir ./outputs \
     --save_name demo_light
   ```
   其余参数（`--NFE`、`--cfg_scale`、`--img_size` 等）保持原用法。`--sd3_variant sd3` 即沿用标准 SD3。

3. **批量文本采样（generate_t2i.py）**  
   ```bash
   python generate_t2i.py \
     --model /path/to/light_sd3_dir \
     --sd3_variant light \
     --ema_ckpt /path/to/ema_transformer.pth \
     --dataset_dir /path/to/prompts \
     --outdir_base ./t2i_out
   ```

4. **DPG / Geneval 基准采样**  
   - `generate_dpg.py` 与 `generate_geneval.py` 同样接受 `--sd3_variant light --ema_ckpt ...`，其余流程不变。

5. **LoRA 训练（train_lora.py）**  
   - 需要在 LightSD3 上微调时，添加 `--sd3_variant light --model_key /path/to/light_sd3_dir --ema_ckpt ...`；其余 LoRA 配置保持不变。

## 注意事项

- LightSD3 模式不支持 `--use_8bit`，请使用全精度/半精度加载文本编码器。  
- EMA 仅作用于 transformer；VAE 与文本编码器继续使用保存目录内的权重。  
- 若传入的 checkpoint 不含 `transformer` 权重会报错，请确认文件内容。  
- 采样代码默认使用 `SD3Euler` 与残差采样逻辑，不影响此前对 SD3 的用法。

