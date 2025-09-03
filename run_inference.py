#!/usr/bin/env python3
"""
Run inference on COCO2017 (val2017 by default) with:
- Faster R-CNN (torchvision)
- DETR (HF)
- DINO (HF)
- GroundingDINO (HF, zero-shot via text prompts)

Writes COCO-format predictions JSON and appends a simple runtime log.
"""

import os
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as T

from pycocotools.coco import COCO

from transformers import (
    DetrImageProcessor, DetrForObjectDetection,
    AutoProcessor, AutoModelForObjectDetection,
    AutoModelForZeroShotObjectDetection
)


# ----------------------- Utilities -----------------------

def set_deterministic(seed: int = 123):
    torch.manual_seed(seed)
    np.random.seed(seed)

def xyxy_to_xywh(box_xyxy: Union[List[float], np.ndarray]) -> List[float]:
    x1, y1, x2, y2 = map(float, box_xyxy)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

def normalize_label(name: str) -> str:
    return name.strip().lower().replace("_", " ")

def build_name_to_cid(coco_gt: COCO) -> Dict[str, int]:
    name2cid = {}
    for c in coco_gt.loadCats(coco_gt.getCatIds()):
        name2cid[normalize_label(c["name"])] = int(c["id"])
    # common aliases -> canonical COCO names
    aliases = {
        "motorbike": "motorcycle", "aeroplane": "airplane", "aircraft": "airplane",
        "tvmonitor": "tv", "tv": "tv", "traffic light": "traffic light",
        "diningtable": "dining table", "pottedplant": "potted plant",
        "cellphone": "cell phone", "mobile phone": "cell phone",
        "hand bag": "handbag", "wineglass": "wine glass", "hairdryer": "hair drier"
    }
    # map aliases directly
    for k, v in aliases.items():
        name2cid.setdefault(normalize_label(k), name2cid.get(normalize_label(v), -1))
    return name2cid

def load_split(coco_dir: str, split: str) -> Tuple[COCO, List[str], List[dict]]:
    ann_file = os.path.join(coco_dir, "annotations", f"instances_{split}.json")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"Missing annotations file: {ann_file}")
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)
    img_paths = [os.path.join(coco_dir, split, im["file_name"]) for im in imgs]
    return coco, img_paths, imgs

def get_device(pref: str) -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------- Model wrappers -----------------------

def build_fasterrcnn(device: torch.device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval().to(device)
    tfm = T.Compose([T.ToTensor()])

    # TorchVision's 91-slot label list (with background + "N/A" gaps)
    TV_CATS_91 = [
        "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
        "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    @torch.no_grad()
    def forward(pil_img: Image.Image):
        x = tfm(pil_img).to(device)
        out = model([x])[0]
        boxes = out["boxes"].detach().cpu().numpy()             # xyxy
        idxs  = out["labels"].detach().cpu().numpy().tolist()   # contiguous indices
        scores= out["scores"].detach().cpu().numpy().tolist()

        # map indices -> names (skip background/“N/A”)
        names = []
        for i in idxs:
            if 0 <= i < len(TV_CATS_91):
                n = TV_CATS_91[i]
                if n != "__background__" and n != "N/A":
                    names.append(n)
                else:
                    names.append(None)
            else:
                names.append(None)
        return boxes, names, scores

    # we now return NAMES, not COCO ids -> downstream will map name->category_id
    return forward, False


def build_detr(device: torch.device):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device).eval()
    id2label = model.config.id2label  # {idx: name}

    @torch.no_grad()
    def forward(pil_img: Image.Image):
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # target_sizes uses (h, w)
        target_sizes = torch.tensor([pil_img.size[::-1]], device=device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        boxes = results["boxes"].detach().cpu().numpy()               # xyxy
        labels_idx = results["labels"].detach().cpu().numpy().tolist()
        scores = results["scores"].detach().cpu().numpy().tolist()
        labels = [id2label[int(i)] for i in labels_idx]               # names
        return boxes, labels, scores

    return forward, False  # labels are names


def build_dino(device: torch.device):
    # IDEA-Research DINO via Transformers hub
    processor = AutoProcessor.from_pretrained("IDEA-Research/dino-resnet-50")
    model = AutoModelForObjectDetection.from_pretrained("IDEA-Research/dino-resnet-50").to(device).eval()
    id2label = model.config.id2label

    @torch.no_grad()
    def forward(pil_img: Image.Image):
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        outputs = model(**inputs)
        target_sizes = torch.tensor([pil_img.size[::-1]], device=device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        boxes = results["boxes"].detach().cpu().numpy()
        labels_idx = results["labels"].detach().cpu().numpy().tolist()
        scores = results["scores"].detach().cpu().numpy().tolist()
        labels = [id2label[int(i)] for i in labels_idx]
        return boxes, labels, scores

    return forward, False  # labels are names


def build_groundingdino(device: torch.device, text_prompts: str, box_thr: float, text_thr: float):
    if not text_prompts:
        # reasonable default coverage of frequent classes if user omitted prompts
        text_prompts = "person,car,dog,cat,bicycle,truck,bus,motorcycle,chair,bench,backpack,umbrella,handbag,cell phone,book,clock"
    queries = [t.strip() for t in text_prompts.split(",") if t.strip()]

    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device).eval()

    @torch.no_grad()
    def forward(pil_img: Image.Image):
        inputs = processor(images=pil_img, text=queries, return_tensors="pt").to(device)
        outputs = model(**inputs)

        H, W = pil_img.size[1], pil_img.size[0]
        target_sizes = torch.tensor([[H, W]], device=device)

        # Transformers versions vary; support both APIs
        if hasattr(processor, "post_process_grounded_object_detection"):
            res = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.get("input_ids"),
                target_sizes=target_sizes,
                box_threshold=box_thr,
                text_threshold=text_thr
            )[0]
            boxes = res["boxes"].detach().cpu().numpy()
            labels = res["labels"]  # strings from queries
            scores = res["scores"].detach().cpu().numpy().tolist()
        else:
            res = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
            boxes = res["boxes"].detach().cpu().numpy()
            scores = res["scores"].detach().cpu().numpy().tolist()
            # if labels are not produced, fallback to first query to avoid crash
            labels = [queries[0]] * len(boxes)

        return boxes, labels, scores

    return forward, False  # labels are names (strings from prompts)


# ----------------------- Main -----------------------

def main():
    parser = argparse.ArgumentParser(description="COCO2017 inference-only runner")
    parser.add_argument("--model", required=True, choices=["fasterrcnn", "detr", "dino", "groundingdino"])
    parser.add_argument("--coco_dir", required=True, help="Path to COCO root (contains val2017/ and annotations/)")
    parser.add_argument("--split", default="val2017", help="val2017 | train2017 | test2017 (eval expects val2017)")
    parser.add_argument("--output", required=True, help="Predictions JSON path")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_images", type=int, default=-1, help="Limit number of images (for quick tests)")
    parser.add_argument("--score_thr", type=float, default=0.3, help="Confidence threshold for saving detections")
    parser.add_argument("--text_prompts", type=str, default="", help="Comma-separated labels (GroundingDINO)")
    parser.add_argument("--box_thr", type=float, default=0.3, help="GroundingDINO box threshold")
    parser.add_argument("--text_thr", type=float, default=0.25, help="GroundingDINO text threshold")
    parser.add_argument("--log", type=str, default="logs/run_log.txt")
    args = parser.parse_args()

    set_deterministic(123)

    out_dir = Path(args.output).parent
    log_dir = Path(args.log).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    coco_gt, img_paths, img_metas = load_split(args.coco_dir, args.split)
    if args.max_images > 0:
        img_paths = img_paths[:args.max_images]
        img_metas = img_metas[:args.max_images]

    name2cid = build_name_to_cid(coco_gt)

    # Build chosen model
    if args.model == "fasterrcnn":
        forward, label_is_id = build_fasterrcnn(device)
    elif args.model == "detr":
        forward, label_is_id = build_detr(device)
    elif args.model == "dino":
        forward, label_is_id = build_dino(device)
    else:
        forward, label_is_id = build_groundingdino(device, args.text_prompts, args.box_thr, args.text_thr)

    results = []
    t0 = time.time()
    processed = 0

    torch.set_grad_enabled(False)

    for meta, img_path in zip(img_metas, img_paths):
        img_id = int(meta["id"])
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            warnings.warn(f"Skipping unreadable image: {img_path} ({e})")
            continue

        boxes, labels, scores = forward(pil)

        for b, lab, sc in zip(boxes, labels, scores):
            if sc < args.score_thr:
                continue

            x1, y1, x2, y2 = [float(v) for v in (b if isinstance(b, (list, tuple, np.ndarray)) else b.tolist())]
            coco_bbox = xyxy_to_xywh([x1, y1, x2, y2])

            if label_is_id:
                cat_id = int(lab)  # torchvision already returns COCO category ids
            else:
                # map label string -> COCO cat id
                if isinstance(lab, str):
                    key = normalize_label(lab)
                    # exact match
                    cat_id = name2cid.get(key, -1)
                    # simple alias fix if not found
                    if cat_id == -1:
                        alias_map = {
                            "bike": "bicycle", "motorbike": "motorcycle", "aeroplane": "airplane",
                            "wineglass": "wine glass", "tv": "tv", "cellphone": "cell phone",
                            "diningtable": "dining table", "pottedplant": "potted plant",
                        }
                        key2 = normalize_label(alias_map.get(key, key))
                        cat_id = name2cid.get(key2, -1)
                else:
                    cat_id = -1  # unsupported label type

            if cat_id == -1:
                # unknown label -> skip to avoid corrupting evaluation
                continue

            results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [float(coco_bbox[0]), float(coco_bbox[1]), float(coco_bbox[2]), float(coco_bbox[3])],
                "score": float(sc)
            })

        processed += 1
        if processed % 100 == 0:
            print(f"{processed} images processed...")

    elapsed = time.time() - t0
    fps = (len(img_paths) / elapsed) if elapsed > 0 else 0.0

    with open(args.output, "w") as f:
        json.dump(results, f)

    peak_mb = None
    if device.type == "cuda":
        try:
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        except Exception:
            peak_mb = None

    # Append a one-line run log for grading reproducibility
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"model={args.model} split={args.split} imgs={len(img_paths)} "
            f"device={device.type} thr={args.score_thr} fps={fps:.2f} "
            f"peakMB={(peak_mb and f'{peak_mb:.1f}') or 'NA'} out={args.output}\n"
        )

    print(f"Saved predictions: {args.output}")
    print(f"Images: {len(img_paths)} | Time: {elapsed:.1f}s | Avg FPS: {fps:.2f}")
    if peak_mb is not None:
        print(f"Peak GPU memory: {peak_mb:.1f} MB")


if __name__ == "__main__":
    main()
