import logging
import os

import numpy as np
from typing import Tuple
from pydantic import Field
from typing_extensions import Literal
import cv2

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum
from frigate.util.model import post_process_yolo

import time

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ModuleNotFoundError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate

logger = logging.getLogger(__name__)

DETECTOR_KEY = "edgetpu"

class EdgeTpuDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default=None, title="Device Type")
    # model_type inherited from BaseDetectorConfig, but can override default

class EdgeTpuTfl(DetectionApi):
    type_key = DETECTOR_KEY
    supported_models = [
        ModelTypeEnum.ssd,
        ModelTypeEnum.yologeneric,
    ]

    def __init__(self, detector_config: EdgeTpuDetectorConfig):
        logger.info(f"Initializing {DETECTOR_KEY} detector with multi-model support (YOLO single/dual tensor, SSD)")
        device_config = {}
        if detector_config.device is not None:
            device_config = {"device": detector_config.device}

        edge_tpu_delegate = None

        try:
            device_type = (
                device_config["device"] if "device" in device_config else "auto"
            )
            logger.info(f"Attempting to load TPU as {device_type}")
            edge_tpu_delegate = load_delegate("libedgetpu.so.1.0", device_config)
            logger.info("TPU found")
            self.interpreter = Interpreter(
                model_path=detector_config.model.path,
                experimental_delegates=[edge_tpu_delegate],
            )
        except ValueError:
            _, ext = os.path.splitext(detector_config.model.path)

            if ext and ext != ".tflite":
                logger.error(
                    "Incorrect model used with EdgeTPU. Only .tflite models can be used with a Coral EdgeTPU."
                )
            else:
                logger.error(
                    "No EdgeTPU was detected. If you do not have a Coral device yet, you must configure CPU detectors."
                )

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

        if self.yolo_model:
            logger.info(f"Preparing YOLO postprocessing for {len(self.tensor_output_details)}-tensor output")
            if len(self.tensor_output_details) == 2:
                self._generate_anchors_and_strides()
                self.min_logit_value = np.log(self.min_score / (1 - self.min_score))
                self.reg_max = 16 # = dfl_channels // 4  # 64 // 4 = 16 # YOLO standard
                self.project = np.arange(self.reg_max, dtype=np.float32)
                # determine YOLO 2-tensor model indices (boxes, class_scores)
                # the tensors have shapes (B, N, C) where N is the number of candidate detections (=2100 for 320x320)
                # this should work properly UNLESS the number of classes is exactly 60 and classes is at index 1
                self.output_boxes_index = None
                self.output_classes_index = None
                for i, x in enumerate(self.tensor_output_details):
                    #logger.info(f"tensor[{i}] found with nominal index {x['index']} and shape {x['shape']}")
                    # the nominal index seems to start at 1 instead of 0
                    if len(x["shape"]) == 3 and x["shape"][2] == 64:
                        if self.output_boxes_index is not None:
                            self.output_classes_index = self.output_boxes_index
                        self.output_boxes_index = i
                    elif len(x["shape"]) == 3:
                        self.output_classes_index = i
                if self.output_boxes_index is None or self.output_classes_index is None:
                    logger.warning(f"Unrecognized model output, unexpected tensor shapes.")
                    self.output_classes_index = 0
                    self.output_boxes_index = 1
                    logger.warning(f"Assuming classes : {self.output_classes_index} and boxes : {self.output_boxes_index}")
                #logger.info(f"Using output index for classes : {self.output_classes_index} and boxes : {self.output_boxes_index}")

        else:
            if model_type not in [ModelTypeEnum.ssd, None]:
                logger.warning(f"Unsupported model_type '{model_type}' for EdgeTPU detector, falling back to SSD")
            logger.info(f"Using SSD preprocessing/postprocessing")

            # SSD model indices (4 outputs: boxes, class_ids, scores, count)
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
        strides = (8, 16, 32) # YOLO standard

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
        """Legacy method for SSD models."""
        if self.output_class_ids_index is None or self.output_class_scores_index is None:
            for i in range(4):
                index = self.tensor_output_details[i]["index"]
                if index != self.output_boxes_index and index != self.output_count_index:
                    if np.mod(np.float32(self.interpreter.tensor(index)()[0][0]), 1) == 0.0:
                        self.output_class_ids_index = index
                    else:
                        self.output_scores_index = index

    def detect_raw(self, tensor_input):
        if self.yolo_model:
            if len(self.tensor_output_details) == 1:
                # Single-tensor YOLO model
                # model output is (1, NC+4, 2100) for 320x320 image size
                # boxes as xywh (normalized to [0,1]) followed by NC class probabilities (also [0,1])
                self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
                self.interpreter.invoke()

                outputs = []
                for output in self.tensor_output_details:
                    x = self.interpreter.get_tensor(output['index'])
                    scale, zero_point = output['quantization']
                    x = (x.astype(np.float32) - zero_point) * scale
                    # Denormalize xywh by image size
                    x[:, [0, 2]] *= self.model_width
                    x[:, [1, 3]] *= self.model_height
                    outputs.append(x)

                return post_process_yolo(outputs, self.model_width, self.model_height)

            if len(self.tensor_output_details) == 2:
                # Two-tensor YOLO model with (non-standard B(H*W)C output format)
                # model output tensor for classes is (1, 2100, NC) where NC is the count of classes, and the scores are logits
                # model output tensor for boxes is (1, 2100, 64) for image size 320x320, representing box dfl scores
                # recommend clamping logit values between -4,+4 to avoid wasting precision on values with probabilty ~zero or ~100%
                self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
                self.interpreter.invoke()

                detections = np.zeros((self.max_detections, 6), np.float32) # empty results as default

                # dequantize scores
                scores_details = self.tensor_output_details[self.output_classes_index]
                scores_scale, scores_zero_point = scores_details['quantization']
                scores_output = self.interpreter.get_tensor(scores_details['index'])
                scores_output = (scores_output.astype(np.float32) - scores_zero_point) * scores_scale
                #logger.info(f"dequantized scores output {scores_output.shape}")

                # 1. Get best class and confidence for each detection
                class_ids = np.argmax(scores_output[0], axis=1)  # (2100,)
                class_confs = np.max(scores_output[0], axis=1)  # (2100,)

                # 1b. Check if no scores above threshold
                mask = class_confs >= self.min_logit_value  # (2100,) this mask filters out low confidence candidates
                if not mask.any():
                    return detections # empty results
                class_confs_filtered = class_confs[mask]  # (N,) where N is likely to be much less than 2100
                class_ids_filtered = class_ids[mask]  # (N,)

                # dequantize boxes
                boxes_details = self.tensor_output_details[self.output_boxes_index]
                boxes_scale, boxes_zero_point = boxes_details['quantization']
                # remove candidates with probabilities < threshold
                boxes_output = (self.interpreter.get_tensor(boxes_details['index']))[:, mask, :]  # (1, N, 64)
                boxes_output = (boxes_output.astype(np.float32) - boxes_zero_point) * boxes_scale
                #logger.info(f"dequantized boxes output {boxes_output.shape}")
                bs, count_filtered_proposals, dfl_channels = boxes_output.shape

                # 2. Decode DFL to distances (ltrb)
                dfl_distributions = boxes_output.reshape(bs, count_filtered_proposals, 4, self.reg_max)  # (1, N, 4, 16)

                # Softmax over the 16 bins
                dfl_exp = np.exp(dfl_distributions - np.max(dfl_distributions, axis=3, keepdims=True))
                dfl_probs = dfl_exp / np.sum(dfl_exp, axis=3, keepdims=True)  # (1, N, 4, 16)

                # Weighted sum: (1, N, 4, 16) * (16,) -> (1, N, 4)
                distances = np.einsum('bpcr,r->bpc', dfl_probs, self.project)

                # 3. Convert distances to bounding boxes (xyxy)
                distances = distances[0]  # (N, 4)

                # Calculate box corners in pixel coordinates
                anchors_filtered = self.anchors[mask]
                anchor_strides_filtered = self.anchor_strides[mask]
                x1y1 = (anchors_filtered - distances[:, [0, 1]]) * anchor_strides_filtered  # (N, 2)
                x2y2 = (anchors_filtered + distances[:, [2, 3]]) * anchor_strides_filtered  # (N, 2)
                boxes_decoded = np.concatenate((x1y1, x2y2), axis=-1)  # (N, 4)

                scores_decoded = 1 / (1 + np.exp(-class_confs_filtered))  # (N,) # sigmoid

                # 9. Apply NMS
                indices = cv2.dnn.NMSBoxes(
                    bboxes=boxes_decoded,
                    scores=scores_decoded, # would prefer to use logits and do this transformation after NMS, but ...
                    score_threshold=self.min_score, # cv2 asserts the min_score must be > 0 which means >50% probability
                    nms_threshold=0.4,
                )

                #logger.info(f"NMS kept {indices.shape} boxes out of {np.sum(mask)}")

                for i, (bbox, confidence, class_id) in enumerate(
                    zip(boxes_decoded[indices], scores_decoded[indices], class_ids_filtered[indices])
                ):
                    if i == self.max_detections:
                        break

                    detections[i] = [
                        class_id,
                        confidence,
                        bbox[1] / self.model_height,
                        bbox[0] / self.model_width,
                        bbox[3] / self.model_height,
                        bbox[2] / self.model_width,
                    ]
                return detections

        else:
            # Default SSD model
            self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
            self.interpreter.invoke()

            self.determine_indexes_for_non_yolo_models()
            boxes = self.interpreter.tensor(self.tensor_output_details[0]["index"])()[0]
            class_ids = self.interpreter.tensor(self.tensor_output_details[1]["index"])()[0]
            scores = self.interpreter.tensor(self.tensor_output_details[2]["index"])()[0]
            count = int(
                self.interpreter.tensor(self.tensor_output_details[3]["index"])()[0]
            )

            detections = np.zeros((self.max_detections, 6), np.float32)

            for i in range(count):
                if scores[i] < self.min_score:
                    break
                if i == self.max_detections:
                    logger.info(f"Too many detections ({count})!")
                    break
                detections[i] = [
                    class_ids[i],
                    float(scores[i]),
                    boxes[i][0],
                    boxes[i][1],
                    boxes[i][2],
                    boxes[i][3],
                ]

            return detections
