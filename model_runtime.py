# 模型/處理器載入、預熱、裝置/資料型別搬運工具
import os
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers.utils.logging import set_verbosity_error

# 全域環境設定
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_verbosity_error()

# 啟用 TF32 / cuDNN 最佳化
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# Test processor and model loading
try:
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        use_fast=True,
    )
    print("Processor loaded successfully!")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        do_sample=False,    # 取消隨機取樣
        torch_dtype=torch.float16,
        device_map={"": 0},   # 指定丟到第 0 張 GPU
    )
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model/processor: {e}")
    raise

def _to_model_device_and_dtype(x):
    """
    遞迴處理:
    - 浮點 torch.Tensor -> 搬到 model.device + 轉為 model.dtype + unsqueeze(0)
    - 整數/布林 torch.Tensor -> 只搬到 model.device + unsqueeze(0)
    - list/tuple -> 逐一處理並保留型別
    - dict -> 逐一處理並保留鍵值
    - 其他型別 -> 原樣返回
    """
    if torch.is_tensor(x):
        if x.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            return x.to(device=model.device, dtype=getattr(model, "dtype", torch.float16)).unsqueeze(0)
        else:
            return x.to(device=model.device).unsqueeze(0)

    if isinstance(x, (list, tuple)):
        proc = [ _to_model_device_and_dtype(v) for v in x ]
        return type(x)(proc)

    if isinstance(x, dict):
        return { k: _to_model_device_and_dtype(v) for k, v in x.items() }

    return x

# --- 啟動時做一次預熱---
def _warmup_once():
    try:
        img = Image.new("RGB", (64, 64), (0, 128, 0))
        dummy = processor.process(images=[img], text="Warm up.")
        dummy = _to_model_device_and_dtype(dummy)  # 只做一次、遞迴處理
        with torch.inference_mode():
            _ = model.generate_from_batch(
                dummy,
                GenerationConfig(max_new_tokens=8, do_sample=False),
                tokenizer=processor.tokenizer
            )
        print("Warmup done.")
    except Exception as e:
        print("Warmup skipped:", e)

_warmup_once()