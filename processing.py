# 單張圖片處理流程（讀圖→前處理→推論→解碼→後處理→回覆→效能統計）
import os
import time
from PIL import Image
import torch
import chainlit as cl
from transformers import GenerationConfig

from model_runtime import processor, model, _to_model_device_and_dtype
from postprocess import postprocess_possible_downgrade
from utils import _cuda_sync_if_available

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