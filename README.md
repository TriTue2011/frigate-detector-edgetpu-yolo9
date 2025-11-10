# frigate-detector-edgetpu-yolo9

This repository provides a custom detector plugin for Frigate, specifically designed to enable the use of **YOLOv9** models with Google Coral Edge TPUs. This plugin handles the necessary post-processing for YOLO model outputs, making them compatible with Frigate's detection pipeline without modifying Frigate's core source code.

## Why YOLOv9 and this Plugin?

Frigate, an open-source NVR system, supports various detector hardware, including the Google Coral Edge TPU. While the default SSD MobileDet model works well in many case, in some cases YOLOv9 models can often offer improved detection accuracy and reduced false positives.

YOLOv9 can run almost entirely on the TPU of the Google Coral device. This detection method is both high-speed and low-energy, handling approximately 90 detections per second and using only a few Watts of power, while leaving the CPU available for other tasks. Another consideration is minimizing heat produced by the CPU, which can challenge my system's small fan.

There are other YOLO models with incompatible software license terms (eg Ultralytics is licensed with the AGPL). This approach is meant to be used to start with the GPL-licensed YOLOv9 model, which IS compatible with Frigate (also MIT-licensed). For more information, see [this discussion in Frigate's issues](https://github.com/blakeblackshear/frigate/discussions/15630#discussioncomment-11639733).

## Features

*   **Improved Accuracy:** Fewer false positive detections than the default SSD MobileDet object detection model.
*   **YOLO Compatibility:** Processes output from YOLO models exported for Edge TPU.
*   **Google Coral Edge TPU Support:** Optimized for efficient inference on Coral devices. ~10ms inference time (vs 7ms for the MobileDet model)
*   **Simple Integration:** Adds as a Frigate plugin via a Docker volume mount, no core Frigate code changes needed.

## Caveat

The plugin code does some post-processing on the CPU, so this approach will use the CPU more than the default SSD MobileDet model. However with careful selection of the YOLO model generation, type, and resolution the amount of CPU involved does not appear to be a bottleneck (on a very old machine, Intel 3rd generation i7).

## Example System Performance

* Google Coral mini-PCIe card
* average 25 detections per second
* zero skipped frames
* 13ms detection speed
* Detector CPU usage varies between 10% and 15%
* CPU is Intel 3rd Generation i7, 8GB RAM, circa 2012


## Prerequisites

Before you begin, ensure you have:

* **A running Frigate installation:** This plugin has been tested with Frigate **v0.15 and v0.16**.
* **Google Coral Edge TPU:** Properly configured and accessible by your Frigate Docker container. Should work with any variation of Google Coral hardware (USB, M2, mPCIe)

## Installation

Follow these steps to integrate the replacement `edgetpu_tfl.py` plugin into your Frigate setup.

### 1. Get a YOLO Model File for Edge TPU

* Download the model weights here: [https://github.com/user-attachments/files/23448296/yolov9-s-relu6-10epoch-17class_320_int8_edgetpu.zip](https://github.com/user-attachments/files/23448296/yolov9-s-relu6-10epoch-17class_320_int8_edgetpu.zip)

* Unzip the file and copy it to your docker host, to somewhere like /opt/frigate-plugins/

### 2. Download the Plugin File and Class Label File

Create a directory on your host system to store the plugin file. For example, you might create `/opt/frigate-plugins/`.

```bash
sudo mkdir -p /opt/frigate-plugins
cd /opt/frigate-plugins
sudo wget https://raw.githubusercontent.com/dbro/frigate-detector-edgetpu-yolo9/main/edgetpu_tfl.py
# Or, if you cloned the repo:
# sudo cp path/to/cloned/repo/edgetpu_tfl.py /opt/frigate-plugins/
```

### 3. Update docker-compose.yml

You need to add a volume mount to your Frigate service in your docker-compose.yml file. This mounts the plugin file into Frigate's detector plugins directory.

Locate your Frigate service definition and add the following two lines under the volumes: section. Adjust /opt/frigate-plugins/edgetpu_tfl.py if you stored the file elsewhere.

The second line to add will make your YOLO model accessible.

```yaml
# ... other services ...
frigate:
  # ... other frigate configurations ...
  volumes:
    # ... existing volumes ...
    - /opt/frigate-plugins/edgetpu_tfl.py:/opt/frigate/frigate/detectors/plugins/edgetpu_tfl.py:ro
    - /opt/frigate-plugins/yolov9-s-relu6-10epoch-17class_320_int8_edgetpu.tflite:/opt/frigate/models/yolov9-s-relu6-10epoch-17class_320_int8_edgetpu.tflite:ro
  # ... rest of frigate service ...
```

After modifying docker-compose.yml, restart your Frigate container:

```bash
docker-compose down
docker-compose up -d
```

### 4. Configure Frigate's config.yml

Now, you need to tell Frigate to use this new detector plugin and your YOLO model.

In your config.yml, under the detectors: section, add the model_type parameter as shown below and update the model path to point to your YOLO model file that you mounted in the previous step.

```yaml
detectors:
  coral:
    type: edgetpu
    # ... other detector settings ...
  model:
      model_type: yolo-generic
      labelmap_path: /labelmap/coco-80.txt
      path: /opt/frigate-plugins/yolov9-s-relu6-10epoch-17class_320_int8_edgetpu.tflite # Update this to your model's path
      # Optionally specify the model dimensions (these are the same as Frigate's default 320x320)
      width: 320
      height: 320
```

### 5. Restart Frigate and Check Performance

Save the Frigate configuration and rexstart Frigate.

* Are there any error messages in Frigate logs during startup?
* What is the inference speed reported in Frigate's System Metrics page?
* What is the level of Detector CPU Usage?
* Are there any skipped frames?
* How many detection errors do you observe? Look for both false positives and false negatives.

## How it Works

This modified version of edgetpu_tfl.py replaces the standard version of the plugin and acts as an intermediary. When Frigate requests a detection:

1. Frigate's standard Edge TPU handler passes the image frame to the plugin.
2. The plugin loads your specified YOLO model.
3. It performs inference on the Edge TPU.
4. It then post-processes the raw output from the YOLO model to transform it into the format expected by Frigate (e.g., converting bounding box coordinates, handling confidence scores).
5. The processed detections are returned to Frigate.

## Contributing

This project is open-source. If you find issues or have improvements, feel free to open an issue or submit a pull request.

## References

* [https://github.com/blakeblackshear/frigate](https://github.com/blakeblackshear/frigate)
* [https://github.com/WongKinYiu/yolov9/tree/main](https://github.com/WongKinYiu/yolov9/tree/main)

## License
This project is licensed under the MIT License. See the LICENSE file for details.
