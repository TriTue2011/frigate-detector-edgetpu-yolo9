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
        try:
            self.min_score = detector_config.model.min_score
        except AttributeError:
            pass

        self.max_detections = 20

        model_type = detector_config.model.model_type
        self.yolo_model = model_type == ModelTypeEnum.yologeneric

        if self.yolo_model:
            logger.info(f"Preparing YOLO postprocessing for {len(self.tensor_output_details)}-tensor output")
            if len(self.tensor_output_details) > 1: # expecting 2 or 3
                self.reg_max = 16 # = dfl_channels // 4  # 64 // 4 = 16 # YOLO standard
                self.min_logit_value = np.log(self.min_score / (1 - self.min_score)) # for filtering
                self._generate_anchors_and_strides() # for decoding bounding box DFL information
                self.project = np.arange(self.reg_max, dtype=np.float32) # for decoding bounding box DFL information

                # determine YOLO tensor indices for boxes and class_scores based on shape
                # the tensors have shapes (B, N, C) where N is the number of candidate detections (=2100 for 320x320)
                # this should work properly EXCEPT if the number of classes is exactly 64, then it might guess wrong
                self.output_boxes_index = None
                self.output_classes_index = None
                self.output_max_scores_index = None
                for i, x in enumerate(self.tensor_output_details):
                    #logger.info(f"tensor[{i}] found with nominal index {x['index']} and shape {x['shape']}")
                    # the nominal index seems to start at 1 instead of 0, so we don't use it
                    if len(x["shape"]) == 3 and x["shape"][2] == 64:
                        self.output_boxes_index = i
                    elif len(x["shape"]) == 3 and x["shape"][2] == 1:
                        # this tensor is optional. It can accelerate post-processing by about 2ms
                        self.output_max_scores_index = i
                    elif len(x["shape"]) == 3:
                        self.output_classes_index = i
                if self.output_boxes_index is None or self.output_classes_index is None:
                    logger.warning(f"Unrecognized model output, unexpected tensor shapes.")
                    self.output_classes_index = 0 if (self.output_boxes is None or self.output_classes_index == 1) else 1 # 0 is default guess
                    self.output_boxes_index = 1 if (self.output_boxes_index == 0) else 0
                classes_count = self.tensor_output_details[self.output_classes_index]["shape"][2]
                accel_text = "" if self.output_max_scores_index is None else f", {self.output_max_scores_index} for max scores"
                logger.info(f"Using tensor index {self.output_boxes_index} for boxes(DFL), {self.output_classes_index} for {classes_count} class scores{accel_text}")

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
        # for decoding the bounding box DFL information into xy coordinates
        all_anchors = []
        all_strides = []
        strides = (8, 16, 32) # YOLO standard for small, medium, large detection heads

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
            # TODO: check if data type of input and model match (uint8 or int8) and cast() if necessary
            if len(self.tensor_output_details) == 1:
                # Single-tensor YOLO model
                # model output is (1, NC+4, 2100) for 320x320 image size
                # boxes as xywh (normalized to [0,1]) followed by NC class probabilities (also [0,1])
                # BEWARE the tensor has only one quantization scale/zero_point, so it should be
                # assembled carefully to have a range of [0,1] for all channels
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

            else:
                # Multi-tensor YOLO model with (non-standard B(H*W)C output format).
                # (the comments indicate the shape of tensors, using "2100" as the anchor count
                # corresponding to an image size of 320x320, "NC" as number of classes,
                # "N" as the post-filtered count)
                # TENSOR A) class scores (1, 2100, NC) where NC is the count of classes, and the score values are logits
                # TENSOR B) max class scores (1, 2100, 1) is an optional tensor to accelerate post-processing by CPU
                #    For my system this reduces inference time from 12ms to 10ms
                # TENSOR C) box coordinates (1, 2100, 64) encoded as dfl scores
                # both (A) and (B) should be clamped to the range [-4,+4] to preserve precision because
                # we don't need details when probability <2% or >98% and NMS requires the min_score
                # parameter to be >= 0 (I don't know why this requirement is necessary for NMS).
                self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
                self.interpreter.invoke()

                detections = np.zeros((self.max_detections, 6), np.float32) # empty results as default

                # get the quantization scaling info to translate thresholds into quantized amounts
                # but don't dequantize scores data yet, wait until the low-confidence candidates
                # are filtered out from the overall result set.
                # this method works with raw quantized numbers when possible, which relies on the
                # value of the scale factor to be >0. This speeds up max and argmax operations.
                scores_output_quantized = None # defer this until later if possible
                if self.output_max_scores_index is not None:
                    # accelerated path
                    # the maximum logit score across all classes has already been calculated by the TPU for each candidate
                    # so we can use that to create the mask and filter the data immediately. saves 1 or 2 milliseconds
                    max_scores_details = self.tensor_output_details[self.output_max_scores_index]
                    max_scores_scale, max_scores_zero_point = max_scores_details['quantization']
                    min_max_score_quantized = int((self.min_logit_value / max_scores_scale) + max_scores_zero_point) # constant
                    max_scores_quantized = (self.interpreter.get_tensor(max_scores_details['index'])[0]).squeeze(-1) # (2100, 1)
                    mask = max_scores_quantized >= min_max_score_quantized # (2100,) this mask filters out low confidence candidates
                    # add 4.0 to (clamped) logit values to get values to be non-negative, which cv2 requires for the input to NMS (why?)
                    # it allows us to reduce the number of sigmoid operations by doing them after NMS
                    # need to convert into float before sending to NMS
                    max_scores_filtered_plus4 = ((max_scores_quantized[mask] - max_scores_zero_point) * max_scores_scale) + 4.0  # (N,1)
                else:
                    # regular path. need to calculate the max value of class scores for each candidate.
                    scores_details = self.tensor_output_details[self.output_classes_index]
                    scores_scale, scores_zero_point = scores_details['quantization']
                    min_score_quantized = int((self.min_logit_value / scores_scale) + scores_zero_point) # constant
                    scores_output_quantized = self.interpreter.get_tensor(scores_details['index'])[0] # (2100, NC)
                    # Get max confidence for each detection. this max() operation over a large tensor takes approx 2ms for 320x320
                    max_scores_quantized = np.max(scores_output_quantized, axis=1)  # (2100,)
                    mask = max_scores_quantized >= min_score_quantized  # (2100,) this mask filters out low confidence candidates
                    max_scores_filtered_plus4 = ((max_scores_quantized[mask] - scores_zero_point) * scores_scale) + 4.0  # (N,1)

                if not mask.any():
                    return detections # empty results

                # dequantize boxes. NMS needs them to be in float format
                boxes_details = self.tensor_output_details[self.output_boxes_index]
                boxes_scale, boxes_zero_point = boxes_details['quantization']
                # remove candidates with probabilities < threshold
                boxes_output_quantized_filtered = (self.interpreter.get_tensor(boxes_details['index'])[0])[mask]  # (N, 64)
                boxes_output_filtered = (boxes_output_quantized_filtered.astype(np.float32) - boxes_zero_point) * boxes_scale
                count_filtered_proposals, dfl_channels = boxes_output_filtered.shape # (N, 16)

                # 2. Decode DFL to distances (ltrb)
                dfl_distributions = boxes_output_filtered.reshape(count_filtered_proposals, 4, self.reg_max)  # (N, 4, 16)

                # Softmax over the 16 bins
                dfl_exp = np.exp(dfl_distributions - np.max(dfl_distributions, axis=2, keepdims=True))
                dfl_probs = dfl_exp / np.sum(dfl_exp, axis=2, keepdims=True)  # (N, 4, 16)

                # Weighted sum: (N, 4, 16) * (16,) -> (N, 4)
                distances = np.einsum('pcr,r->pc', dfl_probs, self.project)

                # Calculate box corners in pixel coordinates
                anchors_filtered = self.anchors[mask]
                anchor_strides_filtered = self.anchor_strides[mask]
                x1y1 = (anchors_filtered - distances[:, [0, 1]]) * anchor_strides_filtered  # (N, 2)
                x2y2 = (anchors_filtered + distances[:, [2, 3]]) * anchor_strides_filtered  # (N, 2)
                boxes_filtered_decoded = np.concatenate((x1y1, x2y2), axis=-1)  # (N, 4)

                # 9. Apply NMS
                indices = cv2.dnn.NMSBoxes(
                    bboxes=boxes_filtered_decoded,
                    scores=max_scores_filtered_plus4, # logit scores shifted to be non-negative to allow min_score <50% or logit<0
                    score_threshold=(self.min_logit_value + 4.0), # cv2 asserts the min_score must be > 0 which means >50% probability
                    nms_threshold=0.4 # should this be a model config setting?
                )
                #logger.info(f"NMS kept {indices.shape} boxes out of {np.sum(mask)}")

                # get the filtered, post-NMS list of class ids
                if scores_output_quantized is None:
                    # leave the scores as quantized values because we will use them to get the class id (not the score itself)
                    scores_details = self.tensor_output_details[self.output_classes_index]
                    scores_output_quantized = self.interpreter.get_tensor(scores_details['index'])[0] # (2100, NC)
                # create a mapping from the index in the nms output to the original index in the model output
                orig_row = np.where(mask)[0]
                if len(indices) > 0:
                    nms_indices = np.array(indices, dtype=np.int32).ravel()  # or .flatten()
                else:
                    nms_indices = np.array([], dtype=np.int32)
                kept_logits_quantized = scores_output_quantized[orig_row[nms_indices]]
                class_ids_post_nms = np.argmax(kept_logits_quantized, axis=1)

                # package the results for Frigate
                for i, (bbox, logit_plus4, class_id) in enumerate(
                    zip(boxes_filtered_decoded[indices], max_scores_filtered_plus4[indices], class_ids_post_nms)
                ):
                    if i == self.max_detections:
                        break

                    detections[i] = [
                        class_id,
                        1 / (1 + np.exp(-1 * (logit_plus4 - 4.0))),  # sigmoid to transform logit score to probability
                        bbox[1] / self.model_height,
                        bbox[0] / self.model_width,
                        bbox[3] / self.model_height,
                        bbox[2] / self.model_width,
                    ]
                if False:
                    # debug statement to indicate if detections are happening at small/med/large head
                    top_i = np.argmax(max_scores) # highest probability detection overall
                    head = "small" if top_i < 1600 else "med" if top_i < 2000 else "large" # these are for 320x320 images
                    logger.info(f"Top detect # {top_i} ({head}) class {class_ids_post_nms[0]} with logit score {(max_scores_filtered_plus4[indices][0] - 4.0):.3f}")
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
