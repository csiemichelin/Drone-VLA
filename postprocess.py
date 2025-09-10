# 後處理：含 "possible" 字樣的降級邏輯
import json
import re

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
        data["scenario"] = "Forest",
        if replace_reason:
            data["reasons"] = "No event detected in the aerial image"

    # 確保必要鍵存在（避免下游取值報錯）
    for k in ("stage", "priority", "event_name", "reasons"):
        data.setdefault(k, "")

    return json.dumps(data, ensure_ascii=False)