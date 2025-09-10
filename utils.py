# 通用工具（CUDA 同步、時間/效能輔助等）
import torch

def _cuda_sync_if_available():
    # 在量測 GPU 時間點做同步，避免非同步造成時間失真
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass