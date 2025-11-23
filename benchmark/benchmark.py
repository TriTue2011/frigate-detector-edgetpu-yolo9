import cv2
import numpy as np
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import time
from frigate_shim import EdgeTpuTfl, EdgeTpuDetectorConfig, ModelConfig, ModelTypeEnum

# --- MAPPING ---
# Frigate YOLOv9 subset (0-16) -> COCO Original ID (1-90)
FRIGATE_TO_COCO = {
    0: 1,  # person
    1: 2,  # bicycle
    2: 3,  # car
    3: 4,  # motorcycle
    4: 5,  # airplane
    5: 6,  # bus
    6: 7,  # train
    7: 8,  # truck
    8: 9,  # boat
    9: 16, # bird
    10: 17, # cat
    11: 18, # dog
    12: 19, # horse
    13: 20, # sheep
    14: 21, # cow
    15: 22, # elephant
    16: 23  # bear
}
# List of COCO IDs we care about for the final score
TARGET_COCO_IDS = list(FRIGATE_TO_COCO.values())

def main(args):
    # 1. Load COCO Ground Truth
    print(f"Loading COCO annotations from {args.coco_ann}...")
    coco_gt = COCO(args.coco_ann)
    
    # Get all image IDs
    img_ids = coco_gt.getImgIds()
    if args.limit:
        img_ids = img_ids[:args.limit]
        print(f"Limiting to first {args.limit} images.")

    # 2. Initialize Model
    model_type = ModelTypeEnum.yologeneric if "yolo" in args.model.lower() else ModelTypeEnum.ssd
    
    config = EdgeTpuDetectorConfig(
        model=ModelConfig(
            path=args.model,
            width=args.size,
            height=args.size,
            model_type=model_type
        )
    )
    
    detector = EdgeTpuTfl(config)
    print(f"Model loaded. Type: {model_type}, Input Size: {args.size}x{args.size}")

    results = []
    inference_times = []
    postprocess_times = []

    print(f"Starting inference on {len(img_ids)} images...")
    
    for img_id in img_ids:
        # Load Image
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(args.coco_img_dir, img_info['file_name'])
        
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read {img_path}")
            continue

        original_h, original_w = frame.shape[:2]

        # --- PREPROCESSING ---
        # Frigate simply resizes the input region to model size.
        # Since we are passing the whole image as the "region", we simple-resize it.
        # This "squashes" the image if aspect ratio differs from model (usually square).
        input_frame = cv2.resize(frame, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB (Frigate does this as it receives YUV usually, but cv2 is BGR)
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension
        input_tensor = np.expand_dims(input_frame, axis=0)

        # --- INFERENCE ---
        # detector.detect_raw expects standard uint8 tensor (if quantized)
        detections = detector.detect_raw(input_tensor)
        
        # Record Timings
        inference_times.append(detector.last_inference_time * 1000)
        postprocess_times.append(detector.last_postprocess_time * 1000)

        # --- FORMATTING RESULTS ---
        # Frigate returns: [class_id, score, ymin, xmin, ymax, xmax] (normalized 0-1)
        for det in detections:
            class_id, score, ymin, xmin, ymax, xmax = det
            
            if score < 0.01: continue # Filter empty rows
            
            # Map Model Class ID to COCO Class ID
            if int(class_id) not in FRIGATE_TO_COCO:
                # If the model predicts something we don't map (unlikely with the subset), skip
                continue
                
            coco_category_id = FRIGATE_TO_COCO[int(class_id)]

            # Denormalize to Original Image Coordinates
            # Note: xmin is relative to model width (320). 
            # We multiply by original_w to get back to absolute coordinates, 
            # implicitly reversing the "squash".
            abs_x = xmin * original_w
            abs_y = ymin * original_h
            abs_w = (xmax - xmin) * original_w
            abs_h = (ymax - ymin) * original_h

            # COCO format: [x, y, width, height]
            result = {
                "image_id": img_id,
                "category_id": coco_category_id,
                "bbox": [abs_x, abs_y, abs_w, abs_h],
                "score": float(score)
            }
            results.append(result)

    print(f"\nInference Complete.")
    print(f"Avg TPU Inference Time: {np.mean(inference_times):.2f} ms")
    print(f"Avg CPU NMS Time:       {np.mean(postprocess_times):.2f} ms")
    print(f"Total Latency:          {np.mean(inference_times) + np.mean(postprocess_times):.2f} ms")

    # 3. Calculate mAP
    if not results:
        print("No detections made.")
        return

    print("Evaluating mAP...")
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # CRITICAL: Filter evaluation to ONLY the classes supported by the model
    coco_eval.params.catIds = TARGET_COCO_IDS
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--size", type=int, default=320, help="Model input size (e.g. 320, 512)")
    parser.add_argument("--coco_ann", required=True, help="Path to instances_val2017.json")
    parser.add_argument("--coco_img_dir", required=True, help="Path to val2017 image directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images (for quick testing)")
    args = parser.parse_args()
    main(args)
