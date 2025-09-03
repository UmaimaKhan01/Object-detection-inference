#!/usr/bin/env python3
"""
Create demo overlays (JPEGs) from a COCO-format predictions JSON.
"""

import os
import json
import random
import argparse
from pathlib import Path

import cv2
from pycocotools.coco import COCO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_dir", required=True)
    ap.add_argument("--split", default="val2017")
    ap.add_argument("--pred", required=True)         # predictions json
    ap.add_argument("--out_dir", default="results/demo")
    ap.add_argument("--num", type=int, default=20)   # how many images to render
    ap.add_argument("--score_thr", type=float, default=0.5)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    ann = f"{args.coco_dir}/annotations/instances_{args.split}.json"
    coco = COCO(ann)
    img_dir = f"{args.coco_dir}/{args.split}"

    with open(args.pred, "r", encoding="utf-8") as f:
        preds = json.load(f)

    # Group detections by image_id
    by_img = {}
    for p in preds:
        if p.get("score", 0.0) < args.score_thr:
            continue
        by_img.setdefault(int(p["image_id"]), []).append(p)

    img_ids = list(by_img.keys())
    if not img_ids:
        print("No detections above threshold to render.")
        return

    random.shuffle(img_ids)
    img_ids = img_ids[:args.num]

    cid2name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    for iid in img_ids:
        info = coco.loadImgs([iid])[0]
        src = os.path.join(img_dir, info["file_name"])
        im = cv2.imread(src)
        if im is None:
            print(f"Could not read {src}, skipping.")
            continue

        for det in by_img[iid]:
            x, y, w, h = det["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            label = cid2name.get(int(det["category_id"]), str(det["category_id"]))
            score = float(det["score"])

            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                im, f"{label} {score:.2f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        outp = os.path.join(args.out_dir, f"{iid}.jpg")
        cv2.imwrite(outp, im)
        print("wrote", outp)


if __name__ == "__main__":
    main()