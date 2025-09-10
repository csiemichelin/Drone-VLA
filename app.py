# Chainlit 入口：啟動訊息、上傳互動、啟動批次掃描
import os
import chainlit as cl
from prompts import DEFAULT_PROMPT
from batch import batch_process_images
from processing import process_image

@cl.on_chat_start
async def start():
    await cl.Message("歡迎使用圖像分析應用！已自動掃描 `images/` 並執行情境真測。你也可以再上傳單張圖片和描述要求。").send()
    await batch_process_images(dir_path="images", prompt=DEFAULT_PROMPT, recursive=True)
    
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

if __name__ == "__main__":
    cl.run()
