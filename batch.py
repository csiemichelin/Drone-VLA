# 批次處理 images/ 資料夾中的所有圖片
from pathlib import Path
import os
import time
import chainlit as cl
from processing import process_image
from prompts import DEFAULT_PROMPT

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

    total_start = time.perf_counter()
    count_all = 0
    count_ok = 0
    time_ok_sum = 0.0
    for p in paths:
        t0 = time.perf_counter()
        try:
            # include_filename=True：訊息會把檔名和 JSON 一起顯示
            # send_perf=False：批次時不刷效能摘要，避免洗版
            text = await process_image(str(p), prompt, include_filename=True, send_perf=False)
            out = text if (text and text.strip()) else "<EMPTY>"
            ok = True
        except Exception as e:
            out = f"<ERROR: {e}>"
            ok = False
        finally:
            dt = time.perf_counter() - t0
            count_all += 1
            if ok:
                count_ok += 1
                time_ok_sum += dt

        # console 一行輸出：檔名 \t JSON
        print(f"{p.name}\t{out}")
    total_elapsed = time.perf_counter() - total_start

    avg_all = (total_elapsed / count_all) if count_all else 0.0
    avg_ok = (time_ok_sum / count_ok) if count_ok else 0.0

    print(f"Total elapsed: {total_elapsed:.3f}s for {count_all} items")
    print(f"Avg per item (all): {avg_all:.3f}s/item")
    print(f"Avg per item (success only): {avg_ok:.3f}s/item")