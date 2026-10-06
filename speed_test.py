import time
import torch
import torch.nn as nn
from thop import profile  # 用于计算FLOPs和参数量

# ====== GlassSegNet model ======
from GlassSegNet import GlassSegNet

def build_models():
    """返回要对比的模型字典"""
    return {
        "GlassSegNet": GlassSegNet(),
        # "Baseline": BaselineModel(),
        # "CCFM": CCFMModel(),
        # ...
    }


# ====== 输入分辨率 (RGB only) ======
INPUT_SPECS = [
    ("352", (1, 3, 352, 352)),
]

# ====== 工具函数 ======
def count_trainable_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

@torch.no_grad()
def measure_fps(model, sample, warmup=30, iters=100, amp=False):
    """测 FPS (平均)"""
    device = next(model.parameters()).device
    # 预热
    for _ in range(warmup):
        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    # 正式计时
    t0 = time.time()
    for _ in range(iters):
        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.time()
    avg_ms = (t1 - t0) / iters * 1000.0
    fps = sample.shape[0] / (avg_ms / 1000.0)
    return fps

@torch.no_grad()
def measure_flops(model, sample):
    """计算FLOPs"""
    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=(sample,))
    flops_g = flops / 1e9  # FLOPs单位是G（十亿次操作）
    params_m = params / 1e6  # 参数量单位是M（百万个参数）
    return flops_g, params_m


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device.upper()}, AMP: OFF")

    models = build_models()
    for name, model in models.items():
        model = model.to(device).eval()
        params_m = count_trainable_params(model) / 1e6
        for tag, shape in INPUT_SPECS:
            sample = torch.randn(*shape, device=device)
            fps = measure_fps(model, sample, warmup=30, iters=100, amp=False)
            flops, params = measure_flops(model, sample)  # 测量FLOPs
            print(f"[{name} @ {tag}] Params={params_m:.2f}M FPS={fps:.2f} FLOPs={flops:.2f}G")
