# Object Detection Inference – Assignment 2

This repository contains my submission for **CVS Assignment 2: Object Detection Comparison**.  
The project evaluates multiple object detection models on the COCO 2017 validation set and provides inference results, evaluation metrics, and demo visualizations.  

---

## 📂 Repository Structure
object_detection_comparison/
│
├── run_inference.py # script to run inference with chosen model
├── evaluate.py # COCO evaluation (AP/AR metrics)
├── make_demo.py # saves demo images with predicted boxes
├── coco80.txt # list of COCO class names
├── results/ # all outputs saved here
│ ├── frcnn.json # Faster R-CNN predictions (full val2017)
│ ├── frcnn_50.json # Faster R-CNN predictions (50 images)
│ ├── detr.json # DETR predictions (full val2017)
│ ├── gdino.json # Grounding-DINO predictions (full val2017)
│ ├── eval_log.txt # evaluation logs for each model
│ ├── demo_frcnn/ # demo images (Faster R-CNN)
│ ├── demo_detr/ # demo images (DETR)
│ └── demo_gdino/ # demo images (Grounding-DINO)
└── logs/ # runtime logs

yaml
Copy code

---

## ✅ Models Covered
- **Faster R-CNN** (`torchvision.models.detection.fasterrcnn_resnet50_fpn`)
- **DETR** (`facebook/detr-resnet-50`)
- **Grounding-DINO** (`IDEA-Research/grounding-dino-base`)  
*(Note: Plain DINO could not be run due to missing public model checkpoint.)*

---

## ⚙️ Environment Setup
Create a conda environment and install dependencies:
```bash
conda create -n gpu_env python=3.9 -y
conda activate gpu_env
pip install -r requirements.txt
Key packages:

torch, torchvision, transformers>=4.40

pycocotools, datasets, evaluate, pillow

📊 Running Inference & Evaluation
1. Run inference (example: Faster R-CNN on 50 images)
powershell
Copy code
cd $ROOT
python run_inference.py --model fasterrcnn --coco_dir "$COCO" --split val2017 --max_images 50 --output results\frcnn_50.json --device cuda --log logs\run.txt
2. Evaluate results
powershell
Copy code
python evaluate.py --coco_dir "$COCO" --split val2017 --pred results\frcnn_50.json --out_log results\eval_log.txt
3. Generate demo images
powershell
Copy code
python make_demo.py --coco_dir "$COCO" --split val2017 --pred results\frcnn_50.json --out_dir results\demo_frcnn --score_thr 0.3
🚀 One-Line Command (all models + demos + zip)
To reproduce my submission in one step (except plain DINO):

powershell
Copy code
cd $ROOT; $env:TF_CPP_MIN_LOG_LEVEL="3"; $PROMPTS=(Get-Content ".\coco80.txt" -Raw);
python run_inference.py --model fasterrcnn --coco_dir "$COCO" --split val2017 --score_thr 0.00 --output results\frcnn.json --device cuda --log logs\run.txt;
python run_inference.py --model detr --coco_dir "$COCO" --split val2017 --score_thr 0.00 --output results\detr.json --device cuda --log logs\run.txt;
python run_inference.py --model groundingdino --coco_dir "$COCO" --split val2017 --text_prompts "$PROMPTS" --box_thr 0.25 --text_thr 0.20 --score_thr 0.00 --output results\gdino.json --device cuda --log logs\run.txt;
python evaluate.py --coco_dir "$COCO" --split val2017 --pred results\frcnn.json --out_log results\eval_log.txt;
python evaluate.py --coco_dir "$COCO" --split val2017 --pred results\detr.json --out_log results\eval_log.txt;
python evaluate.py --coco_dir "$COCO" --split val2017 --pred results\gdino.json --out_log results\eval_log.txt;
python make_demo.py --coco_dir "$COCO" --split val2017 --pred results\frcnn.json --out_dir results\demo_frcnn --score_thr 0.3;
python make_demo.py --coco_dir "$COCO" --split val2017 --pred results\detr.json --out_dir results\demo_detr --score_thr 0.3;
python make_demo.py --coco_dir "$COCO" --split val2017 --pred results\gdino.json --out_dir results\demo_gdino --score_thr 0.3;
Compress-Archive -Force -Path '.\run_inference.py','.\evaluate.py','.\make_demo.py','.\coco80.txt','.\logs\*','.\results\*.json','.\results\eval_log.txt' -DestinationPath '.\cvs_a2_submission.zip'
📌 Notes
COCO 2017 validation set (val2017 + annotations/instances_val2017.json) must be available under $COCO.

Logs (logs/run.txt) show runtime info like FPS and GPU memory.

Evaluation results (results/eval_log.txt) summarize AP/AR metrics for all models.

Demo images with bounding boxes are in results/demo_* folders.

This setup ensures everything needed for grading is centralized in results/ + scripts at repo root.

yaml
Copy code

---








Ask ChatGPT
