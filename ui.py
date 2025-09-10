import chainlit as cl
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers.utils.logging import set_verbosity_error
from PIL import Image
from pathlib import Path
import csv
import torch
import os
import time
import json
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_verbosity_error()

# 啟用 TF32 / cuDNN 最佳化
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# 預設提示（當使用者沒有輸入任何文字時使用）
DEFAULT_PROMPT = """
You are a forest aerial-imagery analysis assistant. Please review the provided aerial photo and complete the following tasks in **two stages**:

## Execution order (do not skip)
1) Run Stage 1. If NOT forest aerial ⇒ output Stage 1 result and STOP.
2) Only if Stage 1 confirms forest aerial ⇒ run Stage 2.

## Stage 1: Scenario check
- If the image is NOT an outdoor aerial photo of a forest (including indoor photos, close-up objects, animals, portraits, city, desert, ocean, farmland, etc.), you MUST classify it as:
  * "stage": "Stage 1"
  * "priority": "No related event"
  * "event_name": "Not Related to Forest Aerial Analysis"
- Only if it is clearly an outdoor aerial photo of a forest, continue to Stage 2.

## Stage 2: Forest event detection (conservative)
- **Wildfire (p0)** — Use only if there are **clear and obvious** signs **within the forested area**, not clouds/fog/industrial plumes:
  - At least **one** must be clearly visible **and** localized: **smoke plume/column**, **distinct fire line**, **visible flame**, **burned/scorched/charred** band.
  - "reasons" must include a **spatial reference** (e.g., “top-left quadrant”, “south of the river”).
  - If chosen, set:
    - "priority": "p0"
    - "event_name": "Forest Wildfire"

- **Landslide / Debris Flow (p0)** — Use only if there are **clear and obvious** geomorphic signatures in forested terrain (do not confuse with logging clearings or natural river bars):
  - Examples of qualifying evidence (at least **one** clearly visible **and** localized): **fresh slope-failure scar** with bare soil/rock; **debris-flow runout channel** or **leveed tongue**; **sediment/debris fan** at a valley mouth; **boulder/mud deposits** along a channel; **channel blockage/damming** or abrupt avulsion; **turbid sediment plume** downstream of a failure.
  - "reasons" must include a **spatial reference**.
  - If chosen, set:
    - "priority": "p0"
    - "event_name": "Landslide / Debris Flow"

- **Deforestation / Illegal Logging (p1)** — Use only if **both** conditions hold (and are not negated/uncertain):
  1) **Heavy machinery present** (e.g., excavator/harvester/logging truck), **AND**
  2) **Large cleared area and/or tree stumps and/or bare soil/earthworks** consistent with tree removal.
  - "reasons" must include a **spatial reference**.
  - If one of the two conditions is missing or unclear, **do not** output p1; use "No event detected".
  - If chosen, set:
    - "priority": "p1"
    - "event_name": "Deforestation / Illegal Logging"

- **No event detected** — If it is a forest aerial photo but the above event criteria are **not fully satisfied**, return:
  - If chosen, set:
    - "priority": "No event detected"
    - "event_name": "Normal Forest Condition"

---
## Output rules (HARD CONSTRAINTS):
- You MUST always output a field called "stage":
  * If you stopped at Stage 1 (image is not a forest aerial photo), set "stage": "Stage 1".
  * If you continued to Stage 2, set "stage": "Stage 2".

Return one JSON object with exactly the keys: priority, event_name, reasons.

Now, output the result for the given image strictly in the following JSON format (no extra text):

{
"stage": "Stage 1 | Stage 2",
"priority": "p0 | p1 | No event detected | No related event",
"event_name": "<event type in words>",
"reasons": "<specific evidence supporting the assigned priority>"
}
""".strip()

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

@cl.on_chat_start
async def start():
    await cl.Message("歡迎使用圖像分析應用!請上傳一張圖片，然後輸入您的問題或描述要求。").send()

@cl.on_message
async def main(message: cl.Message):
    if not message.elements:
        await cl.Message("請先上傳一張圖片，然後再輸入你的問題。").send()
        return
    
    image = message.elements[0]
    if not image.mime.startswith("image"):
        await cl.Message("請上傳一個有效的圖片文件。").send()
        return
    
    user_prompt = message.content.strip() if message.content else DEFAULT_PROMPT
    
    await process_image(image.path, user_prompt)
    
def _cuda_sync_if_available():
    # 在量測 GPU 時間點做同步，避免非同步造成時間失真
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

# 後處理函數：若 stage 為 Stage2 且 reasons 含 "possible"（不分大小寫），強制改為 No event detected
def postprocess_possible_downgrade(text: str, replace_reason: bool = False) -> str:
    """
    入參:
      text: 模型原始輸出（預期為 JSON 字串；若不是 JSON，將原樣返回）
      replace_reason: 若為 True，會把 reasons 改成固定句子

    回傳:
      後處理後的 JSON 字串（若非 JSON 或解析失敗，就原樣返回）
    """
    # 嘗試解析為 JSON；若不是純 JSON，嘗試擷取第一個 {...}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return text
        try:
            data = json.loads(m.group(0))
        except Exception:
            return text

    # 僅在 stage 為 Stage 2/Stage2 時才檢查 possible
    stage_str = str(data.get("stage", ""))
    is_stage2 = re.search(r"\bstage\s*2\b", stage_str, flags=re.I) is not None

    reasons = str(data.get("reasons", ""))
    has_possible = re.search(r"\bpossible\b", reasons, flags=re.I) is not None

    if is_stage2 and has_possible:
        data["stage"] = "Stage 2"
        data["priority"] = "No event detected"
        data["event_name"] = "Normal Forest Condition"
        if replace_reason:
            data["reasons"] = "No event detected in the aerial image"

    # 確保必要鍵存在（避免下游取值報錯）
    for k in ("stage", "priority", "event_name", "reasons"):
        data.setdefault(k, "")

    return json.dumps(data, ensure_ascii=False)

async def batch_process_images(
    dir_path="images",
    prompt=DEFAULT_PROMPT,
    recursive=True,
):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
    base = Path(dir_path)
    if not base.exists():
        await cl.Message(f"`{dir_path}` 不存在。").send()
        return

    paths = (base.rglob("*") if recursive else base.glob("*"))
    paths = [p for p in paths if p.suffix.lower() in exts and p.is_file()]
    paths.sort()

    if not paths:
        await cl.Message(f"`{dir_path}` 內沒有可處理的圖片。").send()
        return

    await cl.Message(f"🔎 發現 {len(paths)} 張圖片，開始批次推論……").send()

    for p in paths:
        try:
            # include_filename=True：訊息會把檔名和 JSON 一起顯示
            # send_perf=False：批次時不刷效能摘要，避免洗版
            text = await process_image(str(p), prompt, include_filename=True, send_perf=False)
            out = text if (text and text.strip()) else "<EMPTY>"
        except Exception as e:
            out = f"<ERROR: {e}>"

        # console 一行輸出：檔名 \t JSON
        print(f"{p.name}\t{out}")
        
@cl.on_chat_start
async def start():
    await cl.Message("歡迎使用圖像分析應用！已自動掃描 `images/` 並執行情境真測。你也可以再上傳單張圖片做互動分析。").send()
    await batch_process_images(dir_path="images", prompt=DEFAULT_PROMPT, recursive=True)
        
async def process_image(image_path, user_prompt, include_filename=False, send_perf=True):
    try:
        overall_s = time.perf_counter()
        print("=" * 50)
        print(f"開始處理圖片: {image_path}")
        print(f"用戶提示: {user_prompt}")
        
        # 1) 讀圖
        t_img_s = time.perf_counter()
        try:
            image = Image.open(image_path).convert("RGB") 
            image.thumbnail((1536, 1536), Image.Resampling.LANCZOS) # 預先縮圖（避免超大圖吃算力與記憶體）
            print(f"圖片載入成功: {image.size}, 模式: {image.mode}")
        except Exception as e:
            await cl.Message(f"無法載入圖片: {str(e)}").send()
            return
        
        t_img_e = time.perf_counter()
        
        # 2) 前處理（processor.process）
        t_proc_s = time.perf_counter()
        try:
            print("開始處理輸入...")
            inputs = processor.process(
                images=[image],
                text=user_prompt
            )
            print(f"處理器返回類型: {type(inputs)}")
            if inputs:
                print(f"處理器返回的鍵: {list(inputs.keys())}")
                for key, value in inputs.items():
                    if value is not None:
                        print(f"{key}: {type(value)}, 形狀: {getattr(value, 'shape', 'N/A')}")
                    else:
                        print(f"{key}: None")
            else:
                print("處理器返回 None!")
                await cl.Message("處理器返回空結果，請檢查模型安裝。").send()
                return
                
        except Exception as e:
            print(f"處理器錯誤: {e}")
            await cl.Message(f"處理器錯誤: {str(e)}").send()
            return
        t_proc_e = time.perf_counter()
        
        # Check essential inputs
        if 'input_ids' not in inputs or inputs['input_ids'] is None:
            await cl.Message("處理器沒有返回有效的 input_ids").send()
            return
        
        # 3) 搬到裝置並加 batch 維度
        t_move_s = time.perf_counter()
        try:
            print("移動到設備並添加批次維度...")
            inputs = _to_model_device_and_dtype(inputs)  # 一次到位，結構保留
            #（可選）印出關鍵張量的 dtype / shape 以確認
            if torch.is_tensor(inputs.get("input_ids", None)):
                print(f"input_ids: {inputs['input_ids'].shape}, {inputs['input_ids'].dtype}, {inputs['input_ids'].device}")
        except Exception as e:
            print(f"設備移動錯誤: {e}")
            await cl.Message(f"設備移動錯誤: {str(e)}").send()
            return
        t_move_e = time.perf_counter()
        
        # 4) 生成（推論）
        _cuda_sync_if_available()
        t_gen_s = time.perf_counter()
        try:
            print("開始生成...")
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=120, stop_strings=["\n}", "\n}\n", "<|endoftext|>"], do_sample=False),
                tokenizer=processor.tokenizer
            )
            print(f"生成完成，輸出形狀: {output.shape}")
            
        except Exception as e:
            print(f"生成錯誤: {e}")
            await cl.Message(f"生成錯誤: {str(e)}").send()
            return
        _cuda_sync_if_available()
        t_gen_e = time.perf_counter()
        
        # 5) 解碼
        t_dec_s = time.perf_counter()
        try:
            # Get generated tokens and decode them to text
            input_length = inputs['input_ids'].size(1)
            total_len = output.shape[1]
            new_tokens = max(0, total_len - input_length)
            generated_tokens = output[0, input_length:]
            generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # 後處理
            generated_text = postprocess_possible_downgrade(generated_text, replace_reason=True)
            
            print(f"生成的文本: {generated_text}")
            
            # === 這裡依需求把檔名一起輸出到訊息 ===
            if generated_text.strip():
                if include_filename:
                    content = f"**{os.path.basename(image_path)}**\n```json\n{generated_text.strip()}\n```"
                else:
                    content = generated_text.strip()
                await cl.Message(content=content).send()
            else:
                await cl.Message("模型沒有生成任何文本，請嘗試其他問題。").send()
                
        except Exception as e:
            print(f"解碼錯誤: {e}")
            await cl.Message(f"解碼錯誤: {str(e)}").send()
            return
        t_dec_e = time.perf_counter()
        overall_e = time.perf_counter()
        
        # 推論效能統計
        t_img = (t_img_e - t_img_s) * 1000.0
        t_proc = (t_proc_e - t_proc_s) * 1000.0
        t_move = (t_move_e - t_move_s) * 1000.0
        t_gen = (t_gen_e - t_gen_s)          # 秒
        t_dec = (t_dec_e - t_dec_s) * 1000.0
        t_total = (overall_e - overall_s)    # 秒
        toks_per_sec = (new_tokens / t_gen) if t_gen > 0 else float('nan')
        
        # 再送出效能摘要（獨立訊息，不影響上面的 JSON）
        perf_msg = (
            "⏱️ **推論效能摘要**\n"
            f"- 讀圖: {t_img:.1f} ms\n"
            f"- 前處理(processor): {t_proc:.1f} ms\n"
            f"- 搬到裝置: {t_move:.1f} ms\n"
            f"- 生成(推論): {t_gen*1000.0:.1f} ms\n"
            f"- 解碼: {t_dec:.1f} ms\n"
            f"- **端到端總時間**: {t_total:.3f} s\n"
            f"- 產生的新 tokens: {new_tokens}\n"
            f"- **吞吐量**: {toks_per_sec:.2f} tokens/s\n"
        )
        
        if send_perf:
            await cl.Message(content=perf_msg).send()

        return generated_text.strip()
            
    except Exception as e:
        import traceback
        print(f"總體錯誤: {e}")
        print(f"完整堆棧:\n{traceback.format_exc()}")
        await cl.Message(f"處理圖片時發生錯誤: {str(e)}").send()

if __name__ == "__main__":
    cl.run()
