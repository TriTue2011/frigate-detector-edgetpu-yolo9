import logging
import os
import numpy as np
import time
import cv2
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field

# --- MOCKING FRIGATE DEPENDENCIES ---
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class ModelTypeEnum(str, Enum):
    ssd = "ssd"
    yologeneric = "yologeneric"

class ModelConfig(BaseModel):
    path: str
    width: int
    height: int
    model_type: ModelTypeEnum

class EdgeTpuDetectorConfig(BaseModel):
    type: Literal["edgetpu"] = "edgetpu"
    device: Optional[str] = None
    model: ModelConfig

# Placeholder for the imported function if you aren't using the single-tensor model
def post_process_yolo(outputs, w, h):
    raise NotImplementedError("Single tensor path not implemented in benchmark shim")

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ModuleNotFoundError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter, load_delegate
    except ImportError:
        logger.error("Could not import tflite_runtime or tensorflow.lite")
        raise

# --- YOUR ORIGINAL CODE (Modified only to use mocks) ---

class EdgeTpuTfl:
    def __init__(self, detector_config: EdgeTpuDetectorConfig):
        logger.info(f"Initializing Detector: {detector_config.model.path}")
        
        device_config = {}
        if detector_config.device is not None:
            device_config = {"device": detector_config.device}

        edge_tpu_delegate = None
        try:
            # Force "auto" or specific device
            edge_tpu_delegate = load_delegate("libedgetpu.so.1.0", device_config)
            self.interpreter = Interpreter(
                model_path=detector_config.model.path,
                experimental_delegates=[edge_tpu_delegate],
            )
        except Exception as e:
            logger.error(f"Error loading EdgeTPU: {e}")
            raise

        self.interpreter.allocate_tensors()

        self.tensor_input_details = self.interpreter.get_input_details()
        self.tensor_output_details = self.interpreter.get_output_details()
        self.model_width = detector_config.model.width
        self.model_height = detector_config.model.height

        self.min_score = 0.4
        self.max_detections = 20

        model_type = detector_config.model.model_type
        self.yolo_model = model_type == ModelTypeEnum.yologeneric
        
        # METRICS STORAGE (Added for benchmarking)
        self.last_inference_time = 0
        self.last_postprocess_time = 0

        if self.yolo_model:
            logger.info(f"Preparing YOLO postprocessing for {len(self.tensor_output_details)}-tensor output")
            if len(self.tensor_output_details) in [2,3]:
                self._generate_anchors_and_strides()
                self.min_logit_value = np.log(self.min_score / (1 - self.min_score))
                self.reg_max = 16 
                self.project = np.arange(self.reg_max, dtype=np.float32)
                
                self.output_boxes_index = None
                self.output_classes_index = None
                self.output_max_scores_index = None
                
                for i, x in enumerate(self.tensor_output_details):
                    if len(x["shape"]) == 3 and x["shape"][2] == 64:
                        self.output_boxes_index = i
                    elif len(x["shape"]) == 3 and x["shape"][2] == 1:
                        self.output_max_scores_index = i
                    elif len(x["shape"]) == 3:
                        self.output_classes_index = i
                
                if self.output_boxes_index is None or self.output_classes_index is None:
                    # Fallback logic from your code
                    self.output_classes_index = 0 
                    self.output_boxes_index = 1 
        else:
            # SSD Logic
            for x in self.tensor_output_details:
                if len(x["shape"]) == 3:
                    self.output_boxes_index = x["index"]
                elif len(x["shape"]) == 1:
                    self.output_count_index = x["index"]
            self.output_class_ids_index = None
            self.output_class_scores_index = None

    def _generate_anchors_and_strides(self):
        all_anchors = []
        all_strides = []
        strides = (8, 16, 32) 

        for stride in strides:
            feat_h, feat_w = self.model_height // stride, self.model_width // stride
            grid_y, grid_x = np.meshgrid(
                np.arange(feat_h, dtype=np.float32),
                np.arange(feat_w, dtype=np.float32),
                indexing='ij'
            )
            grid_coords = np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)
            anchor_points = grid_coords + 0.5
            all_anchors.append(anchor_points)
            all_strides.append(np.full((feat_h * feat_w, 1), stride, dtype=np.float32))

        self.anchors = np.concatenate(all_anchors, axis=0)
        self.anchor_strides = np.concatenate(all_strides, axis=0)

    def determine_indexes_for_non_yolo_models(self):
        if self.output_class_ids_index is None or self.output_class_scores_index is None:
            for i in range(4):
                index = self.tensor_output_details[i]["index"]
                if index != self.output_boxes_index and index != self.output_count_index:
                    if np.mod(np.float32(self.interpreter.tensor(index)()[0][0]), 1) == 0.0:
                        self.output_class_ids_index = index
                    else:
                        self.output_scores_index = index

    def detect_raw(self, tensor_input):
        start_inference = time.perf_counter()
        
        if self.yolo_model:
            # Only implementing the multi-tensor path as requested
            if len(self.tensor_output_details) in [2,3]:
                self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
                self.interpreter.invoke()
                
                # End Inference / Start Post-Process
                end_inference = time.perf_counter()
                self.last_inference_time = end_inference - start_inference

                detections = np.zeros((self.max_detections, 6), np.float32) 

                scores_output_quantized = None
                
                if self.output_max_scores_index is not None:
                    max_scores_details = self.tensor_output_details[self.output_max_scores_index]
                    max_scores_scale, max_scores_zero_point = max_scores_details['quantization']
                    min_max_score_quantized = int((self.min_logit_value / max_scores_scale) + max_scores_zero_point)
                    max_scores_quantized = (self.interpreter.get_tensor(max_scores_details['index'])[0]).squeeze(-1)
                    mask = max_scores_quantized >= min_max_score_quantized
                    max_scores_filtered_plus4 = ((max_scores_quantized[mask] - max_scores_zero_point) * max_scores_scale) + 4.0
                else:
                    scores_details = self.tensor_output_details[self.output_classes_index]
                    scores_scale, scores_zero_point = scores_details['quantization']
                    min_score_quantized = int((self.min_logit_value / scores_scale) + scores_zero_point)
                    scores_output_quantized = self.interpreter.get_tensor(scores_details['index'])[0]
                    max_scores_quantized = np.max(scores_output_quantized, axis=1)
                    mask = max_scores_quantized >= min_score_quantized
                    max_scores_filtered_plus4 = ((max_scores_quantized[mask] - scores_zero_point) * scores_scale) + 4.0

                if not mask.any():
                    self.last_postprocess_time = time.perf_counter() - end_inference
                    return detections

                boxes_details = self.tensor_output_details[self.output_boxes_index]
                boxes_scale, boxes_zero_point = boxes_details['quantization']
                
                boxes_output_quantized_filtered = (self.interpreter.get_tensor(boxes_details['index'])[0])[mask]
                boxes_output_filtered = (boxes_output_quantized_filtered.astype(np.float32) - boxes_zero_point) * boxes_scale
                
                count_filtered_proposals, dfl_channels = boxes_output_filtered.shape
                dfl_distributions = boxes_output_filtered.reshape(count_filtered_proposals, 4, self.reg_max)
                dfl_exp = np.exp(dfl_distributions - np.max(dfl_distributions, axis=2, keepdims=True))
                dfl_probs = dfl_exp / np.sum(dfl_exp, axis=2, keepdims=True)
                distances = np.einsum('pcr,r->pc', dfl_probs, self.project)

                anchors_filtered = self.anchors[mask]
                anchor_strides_filtered = self.anchor_strides[mask]
                x1y1 = (anchors_filtered - distances[:, [0, 1]]) * anchor_strides_filtered
                x2y2 = (anchors_filtered + distances[:, [2, 3]]) * anchor_strides_filtered
                boxes_filtered_decoded = np.concatenate((x1y1, x2y2), axis=-1)

                indices = cv2.dnn.NMSBoxes(
                    bboxes=boxes_filtered_decoded,
                    scores=max_scores_filtered_plus4,
                    score_threshold=(self.min_logit_value + 4.0),
                    nms_threshold=0.4
                )

                if scores_output_quantized is None:
                    scores_details = self.tensor_output_details[self.output_classes_index]
                    scores_output_quantized = self.interpreter.get_tensor(scores_details['index'])[0]
                
                orig_row = np.where(mask)[0]
                if len(indices) > 0:
                    nms_indices = np.array(indices, dtype=np.int32).ravel()
                else:
                    nms_indices = np.array([], dtype=np.int32)
                
                kept_logits_quantized = scores_output_quantized[orig_row[nms_indices]]
                class_ids_post_nms = np.argmax(kept_logits_quantized, axis=1)

                for i, (bbox, logit_plus4, class_id) in enumerate(
                    zip(boxes_filtered_decoded[indices], max_scores_filtered_plus4[indices], class_ids_post_nms)
                ):
                    if i == self.max_detections:
                        break

                    detections[i] = [
                        class_id,
                        1 / (1 + np.exp(-1 * (logit_plus4 - 4.0))),
                        bbox[1] / self.model_height,
                        bbox[0] / self.model_width,
                        bbox[3] / self.model_height,
                        bbox[2] / self.model_width,
                    ]
                
                self.last_postprocess_time = time.perf_counter() - end_inference
                return detections
        else:
            # SSD Model
            self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
            self.interpreter.invoke()
            
            end_inference = time.perf_counter()
            self.last_inference_time = end_inference - start_inference

            self.determine_indexes_for_non_yolo_models()
            boxes = self.interpreter.tensor(self.tensor_output_details[0]["index"])()[0]
            class_ids = self.interpreter.tensor(self.tensor_output_details[1]["index"])()[0]
            scores = self.interpreter.tensor(self.tensor_output_details[2]["index"])()[0]
            count = int(self.interpreter.tensor(self.tensor_output_details[3]["index"])()[0])

            detections = np.zeros((self.max_detections, 6), np.float32)

            for i in range(count):
                if scores[i] < self.min_score:
                    break
                if i == self.max_detections:
                    break
                detections[i] = [
                    class_ids[i],
                    float(scores[i]),
                    boxes[i][0],
                    boxes[i][1],
                    boxes[i][2],
                    boxes[i][3],
                ]
            self.last_postprocess_time = time.perf_counter() - end_inference
            return detections
