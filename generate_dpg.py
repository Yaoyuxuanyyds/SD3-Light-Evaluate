import argparse
import numpy as np
import random, os, glob
import torch
from torchvision.utils import save_image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as torch_transforms
from PIL import Image

from sampler import SD3Euler

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5]),
    ])
    return transform


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def make_grid_2x2(imgs):
    """imgs: list of 4 tensors, each (3,1024,1024)"""
    assert len(imgs) == 4

    # 转 PIL 更简单
    pil_imgs = [(torch.clamp(img * 0.5 + 0.5, 0, 1) * 255).permute(1, 2, 0).byte().cpu().numpy() for img in imgs]
    pil_imgs = [Image.fromarray(p) for p in pil_imgs]

    w, h = pil_imgs[0].size
    grid = Image.new("RGB", (w * 2, h * 2))

    grid.paste(pil_imgs[0], (0, 0))
    grid.paste(pil_imgs[1], (w, 0))
    grid.paste(pil_imgs[2], (0, h))
    grid.paste(pil_imgs[3], (w, h))

    return grid


# ================================================================
#                       主脚本
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--NFE", type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=1024)
    
    parser.add_argument("--model", type=str, default="sd3")
    parser.add_argument('--load_dir', type=str, default=None, help="replace it with your checkpoint")
    parser.add_argument("--save_dir", type=str, required=True)

    # dpg bench prompt path
    parser.add_argument("--prompt_dir", type=str, default="/inspire/hdd/project/chineseculture/public/yuxuan/benches/ELLA/dpg_bench/prompts")


    # residual
    parser.add_argument("--residual_target_layers", type=int, nargs="+", default=None)
    parser.add_argument("--residual_origin_layer", type=int, default=None)
    parser.add_argument("--residual_weights", type=float, nargs="+", default=None)
    # 多 GPU 分片参数
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total number of workers for sharded prompts (e.g., 4 GPUs).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="This worker's rank in [0, world_size-1].",
    )


    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    if args.model != "sd3":
        raise ValueError("Only sd3 is supported for this benchmark.")
    sampler = SD3Euler(use_8bit=False, load_ckpt_path=args.load_dir)


    sampler.denoiser.to(torch.float32)
    torch.set_default_dtype(torch.float32)

    # prepare dirs
    os.makedirs(args.save_dir, exist_ok=True)

    # # 扫描 prompts
    txt_files = sorted(glob.glob(os.path.join(args.prompt_dir, "*.txt")))
    total_prompts = len(txt_files)

    # 多 GPU 分片：按下标对 world_size 取模
    if args.world_size > 1:
        txt_files = [f for i, f in enumerate(txt_files) if i % args.world_size == args.rank]

    print(f"[DPG] World size = {args.world_size}, rank = {args.rank}")
    print(f"[DPG] Total prompts: {total_prompts}, this rank will handle: {len(txt_files)}")


    # ================================================================
    #                        遍历每个 prompt
    # ================================================================
    for txt_path in txt_files:
        base = os.path.basename(txt_path)
        name = os.path.splitext(base)[0]
        out_path = os.path.join(args.save_dir, f"{name}.png")

        # 若已生成则跳过
        if os.path.exists(out_path):
            print(f"[Skip] {name}.png already exists.")
            continue

        # 读取 prompt
        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        print(f"\n[DPG] Generating for: {name}")

        # ------------------------------------------------
        #    一次输入 prompt，输出 4 张图像
        # ------------------------------------------------
        prompts = [prompt] * 4

        with torch.inference_mode():
            if args.residual_origin_layer is None:
                imgs = sampler.sample(
                    prompts,
                    NFE=args.NFE,
                    img_shape=(args.img_size, args.img_size),
                    cfg_scale=args.cfg_scale,
                    batch_size=4,
                )
            else:
                imgs = sampler.sample_residual(
                    prompts,
                    NFE=args.NFE,
                    img_shape=(args.img_size, args.img_size),
                    cfg_scale=args.cfg_scale,
                    batch_size=4,
                    residual_target_layers=args.residual_target_layers,
                    residual_origin_layer=args.residual_origin_layer,
                    residual_weights=args.residual_weights,
                )

        # imgs shape: [4, 3, 1024, 1024]
        # 拼接成 2×2 = 2048×2048
        grid = make_grid_2x2([imgs[i] for i in range(4)])

        # 保存
        grid.save(out_path)
        print(f"[Saved] {out_path}")

    print("\n[DPG] All done.")
