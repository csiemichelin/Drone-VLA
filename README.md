Drone-VLA/
├─ app.py                  # Chainlit 入口：啟動訊息、上傳互動、啟動批次掃描
├─ processing.py           # 單張圖片處理流程（讀圖→前處理→推論→解碼→後處理→回覆→效能統計）
├─ batch.py                # 批次處理 images/ 資料夾中的所有圖片
├─ model_runtime.py        # 模型/處理器載入、預熱、裝置/資料型別搬運工具
├─ postprocess.py          # 後處理：含 "possible" 字樣的降級邏輯
├─ utils.py                # 通用工具（CUDA 同步、時間/效能輔助等）
├─ prompts.py              # 預設提示（DEFAULT_PROMPT）
├─ requirements.txt        # 相依套件
└─ images/                 #（可選）放要自動批次推論的影像