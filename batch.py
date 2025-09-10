# 批次處理 images/ 資料夾中的所有圖片
from pathlib import Path
import os
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