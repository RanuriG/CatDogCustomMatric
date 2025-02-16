# YOLO v5 Custom Metric Setup & Training Instructions

This repository contains the setup and training instructions for YOLO v5 with a custom metric designed to address Scale Sensitivity in bounding box predictions.

## 0. YOLO Setup/Architecture

YOLO v5 is an open-source object detection architecture that is fast, accurate, and highly efficient. In this setup, the YOLO v5 model is modified to incorporate a custom loss function with an emphasis on Scale Sensitivity when calculating Intersection over Union (IoU) for bounding boxes.

### Key Modifications:
1. **Loss Function**: In the `utils/loss.py` file, update the IoU calculation to include `CIoU=True` and `ScaleSensitivity=True` at line 159:

    ```python
    iou = bbox_iou(pbox, tbox[i], CIoU=True, ScaleSensitivity=True).squeeze()  # iou(prediction, target)
    ```

2. **Scale Sensitivity**: The `utils/matrices.py` file is modified to include the custom Scale Sensitivity metric, which is used to evaluate the closeness of the predicted bounding box area to the ground truth bounding box area. This metric is crucial for improving the accuracy of predictions in variable-sized objects (e.g., pedestrians and vehicles). Replace `bbox_iou` with below.

   ```python
    def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False,ScaleSensitivity=False, eps=1e-7):
    """
    Calculates IoU, GIoU, DIoU, or CIoU between two boxes, supporting xywh/xyxy formats.

    Input shapes are box1(1,4) to box2(n,4).
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    
    # Compute areas
    area1 = w1 * h1
    area2 = w2 * h2

    
    # Compute Scale Sensitivity
    if ScaleSensitivity:
        scale_sensitivity = 1 - torch.abs(area1 - area2) / torch.max(area1, area2)

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return scale_sensitivity + (iou - (rho2 / c2 + v * alpha))  # CIoU and Scale Sensitivity
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU
    ```

## 1. Custom Metric Definition

### Scale Sensitivity:
This metric measures how closely the area of the predicted bounding box matches the ground truth bounding box area. It is defined as:

\[
Scale Sensitivity = 1 - \frac{|area_{pred} - area_{gt}|}{\max(area_{pred}, area_{gt})}
\]

Where:
- `area_pred` = Area of the predicted bounding box
- `area_gt` = Area of the ground truth bounding box

This provides a similarity score between 0 and 1, where 1 indicates perfect matching of the bounding box areas, and 0 indicates complete mismatch. This metric helps reduce false positives and negatives, especially in varying object sizes.


## 2. Instructions for Training & Evaluation

Follow these steps to set up the YOLO v5 model with the custom metric for training and evaluation:

### Prerequisites:
- Google Colab or local setup with GPU support.
- Install necessary dependencies.

### Setup on Google Colab:
1. **Clone the YOLO v5 repository**:

    ```bash
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    ```

2. **Install dependencies**:

    ```bash
    pip install -U -r requirements.txt
    ```

3. **Set up your dataset** by uploading it to Google Drive or use a direct dataset link. [Dataset](https://drive.google.com/drive/folders/1WQKvLgxxZmGJHD84mTzdQkpCbXoPaZxn?usp=sharing)


4. **Modify the `utils/loss.py` file** as above.

5. **Modify `utils/matrices.py`** as above.

6. **Train the model** using the custom loss:

    ```bash
    python train.py --data <dataset> --cfg <yolov5_config> --weights yolov5s.pt --batch-size 16
    ```

### Evaluation:
Once training is completed, you can evaluate the model’s performance using the following command:

```bash
python val.py --data <dataset> --weights runs/train/exp/weights/best.pt --img-size 640 --batch-size 16
 ```
