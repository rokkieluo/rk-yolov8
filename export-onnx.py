import sys

from ultralytics import YOLO

# 选择类型：det, seg, pose, obb
model_type = sys.argv[1]
# 权重位置
input_model = sys.argv[2]

if model_type == "det":
    # 转为onnx：注意需要修改为对应的模型配置文件
    model = YOLO("./weights/yolov8n.pt")
    results = model(
        task="detect", mode="predict", source="./test.jpg", imgsz=640, line_width=3, show=True, save=True, device="cpu"
    )
elif model_type == "seg":
    model = YOLO("./weights/yolov8n-seg.pt")
    results = model(
        task="detect", mode="predict", source="./test.jpg", imgsz=640, line_width=3, show=True, save=True, device="cpu"
    )
elif model_type == "pose":
    model = YOLO("./weights/yolov8n-pose.pt")
    results = model(
        task="detect", mode="predict", source="./test.jpg", imgsz=640, line_width=3, show=True, save=True, device="cpu"
    )
elif model_type == "obb":
    model = YOLO("./weights/yolov8n-obb.pt")
    results = model(
        task="detect", mode="predict", source="./test.jpg", imgsz=640, line_width=3, show=True, save=True, device="cpu"
    )
