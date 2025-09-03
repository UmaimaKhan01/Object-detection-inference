#!/usr/bin/env python3
"""
Evaluate COCO-format predictions JSON against COCO2017 annotations.
Writes a compact TSV-like log to results/eval_log.txt by default.
"""

import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_dir", required=True, help="COCO root (contains annotations/)")
    ap.add_argument("--split", default="val2017", help="Evaluation split: val2017")
    ap.add_argument("--pred", required=True, help="Predictions JSON path")
    ap.add_argument("--out_log", default="results/eval_log.txt", help="File to append results")
    args = ap.parse_args()

    ann = f"{args.coco_dir}/annotations/instances_{args.split}.json"
    coco_gt = COCO(ann)
    coco_dt = coco_gt.loadRes(args.pred)

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # COCOeval.stats (12 metrics, standard order)
    headers = [
        "AP","AP50","AP75","AP_small","AP_medium","AP_large",
        "AR1","AR10","AR100","AR_small","AR_medium","AR_large"
    ]
    with open(args.out_log, "a", encoding="utf-8") as f:
        f.write(f"\n=== EVAL {args.pred} ===\n")
        f.write("metric\tvalue\n")
        for name, val in zip(headers, list(ev.stats)):
            f.write(f"{name}\t{val:.4f}\n")


if __name__ == "__main__":
    main()
