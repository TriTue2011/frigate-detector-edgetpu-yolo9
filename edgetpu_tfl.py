import logging
import os

import numpy as np
from pydantic import Field
from typing_extensions import Literal

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

    def __init__(self, detector_config: EdgeTpuDetectorConfig):
        logger.info(f"Initializing {DETECTOR_KEY} detector with support for model_type ssd and yolo-generic")
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
        self.model_width = detector_config.model.width # used for YOLO
        self.model_height = detector_config.model.height # used for YOLO
        self.min_score = 0.4 # used for SSD
        self.max_detections = 20 # used for SSD

        model_type = detector_config.model.model_type
        self.yolo_model = model_type == ModelTypeEnum.yologeneric # config YAML sets model_type: yolo-generic

        if self.yolo_model:
            logger.info(f"Using YOLO model type")
        else:
            if model_type not in [ModelTypeEnum.ssd, None]:
                logger.warning(f"Unsupported model_type '{model_type}' for EdgeTPU detector, falling back to SSD")
            logger.info(f"Using SSD model type")

            # SSD model indices (4 outputs: boxes, class_ids, scores, count)
            for x in self.tensor_output_details:
                if len(x["shape"]) == 3:
                    self.output_boxes_index = x["index"]
                elif len(x["shape"]) == 1:
                    self.output_count_index = x["index"]

            self.output_class_ids_index = None
            self.output_class_scores_index = None

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
            # Single-tensor YOLO model with shape (84, N)
            # first four numbers in each detection are xywh pixel coordinates (normalized to [0,1])
            # followed by 80 COCO class probabilities (also [0,1])
            self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
            self.interpreter.invoke()

            model_width = self.model_width
            model_height = self.model_height
            outputs = []
            for output in self.tensor_output_details:
                x = self.interpreter.get_tensor(output['index'])
                scale, zero_point = output['quantization']
                x = (x.astype(np.float32) - zero_point) * scale
                # Denormalize xywh by image size
                x[:, [0, 2]] *= model_width
                x[:, [1, 3]] *= model_height
                outputs.append(x)

            return post_process_yolo(outputs, model_width, model_height)
 
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
