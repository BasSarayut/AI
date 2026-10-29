import os
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

from calorie.estimator import CalorieReport, detections_from_result, estimate_calories
from calorie.nutrition import load_nutrition, missing_classes

ROOT = Path(__file__).parent
NUTRITION = load_nutrition(ROOT / "data" / "nutrition.csv")
DISCLAIMER = "ค่าที่ได้เป็นการประมาณจากขนาดจานมาตรฐาน ไม่ใช่คำแนะนำทางการแพทย์"


def load_model() -> YOLO:
    local = Path(os.environ.get("MODEL_PATH", ROOT / "weights" / "best.pt"))
    if local.exists():
        return YOLO(str(local))
    repo = os.environ.get("HF_MODEL_REPO")
    if not repo:
        raise RuntimeError("ไม่พบโมเดล: วางไฟล์ไว้ที่ weights/best.pt หรือตั้งค่า env HF_MODEL_REPO")
    from huggingface_hub import hf_hub_download

    return YOLO(hf_hub_download(repo_id=repo, filename="best.pt"))


MODEL = load_model()
if missing := missing_classes(MODEL.names.values(), NUTRITION):
    print(f"[warn] classes without nutrition data (add them to data/nutrition.csv): {missing}")


def summarize(report: CalorieReport) -> str:
    if not report.items:
        lines = ["### ไม่พบอาหารที่รู้จักในภาพ", "ลองลด confidence หรือถ่ายให้เห็นจานชัดขึ้น"]
    else:
        lines = [
            f"### รวมประมาณ {report.total_kcal:,.0f} kcal",
            f"ช่วงที่เป็นไปได้: {report.low_kcal:,.0f} - {report.high_kcal:,.0f} kcal",
        ]
    if report.unknown_labels:
        lines.append(f"ยังไม่มีข้อมูลโภชนาการของ: {', '.join(report.unknown_labels)}")
    lines.append(f"_{DISCLAIMER}_")
    return "\n\n".join(lines)


def analyze(image, min_confidence: float):
    if image is None:
        raise gr.Error("กรุณาอัปโหลดรูปอาหาร")
    result = MODEL.predict(image, conf=min_confidence, verbose=False)[0]
    report = estimate_calories(detections_from_result(result), NUTRITION, min_confidence=min_confidence)
    rows = [[i.name_th, i.label, f"{i.confidence:.0%}", round(i.kcal)] for i in report.items]
    return result.plot()[..., ::-1], rows, summarize(report)


with gr.Blocks(title="Thai Food Calorie Estimator") as demo:
    gr.Markdown("# Thai Food Calorie Estimator\nอัปโหลดรูปอาหารไทย ระบบจะตรวจจับเมนูด้วย YOLO แล้วประมาณแคลอรี่")
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="รูปอาหาร")
            confidence = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="Confidence ขั้นต่ำ")
            run = gr.Button("วิเคราะห์", variant="primary")
        with gr.Column():
            image_out = gr.Image(label="ผลการตรวจจับ")
            items = gr.Dataframe(headers=["เมนู", "class", "ความมั่นใจ", "kcal"], label="รายการอาหาร")
            summary = gr.Markdown()
    run.click(analyze, inputs=[image_in, confidence], outputs=[image_out, items, summary])

if __name__ == "__main__":
    demo.launch()
