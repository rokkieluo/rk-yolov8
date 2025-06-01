# RK3588的yolov8模型转换(pt->onnx)

本教程适用于6个输出头的onnx目标检测/10个输出头的分割/9个输出头的姿态/9个输出头的旋转模型，如需9个输出头，请参考rk官方的示例：[airockchip/ultralytics_yolov8: NEW - YOLOv8 🚀 in PyTorch > ONNX > CoreML > TFLite](https://github.com/airockchip/ultralytics_yolov8)

第一步：获取本仓库代码

```bash
git clone https://github.com/rokkieluo/rk-yolov8.git
```

第二步：更改model.py

以检测模型为例

该文件位于ultralytics-main/ultralytics/engine/model.py

找到第313行，将导出路径torch.onnx.export设置为你自己的导出路径和导出文件名

```python
if model == 'det':
   print("===========  det-onnx start =========== ")
   input_names = ["data"]
   output_names = ["reg1", "cls1", "reg2", "cls2", "reg3", "cls3"]
   torch.onnx.export(self.model, dummy_input, "./weights/yolov8n.onnx", verbose=False, input_names=input_names, output_names=output_names, opset_version=11)
    print("======================== convert det-onnx Finished! .... ")
```

其他模型的导出同理

第三步：更改export-onnx.py

```python
from ultralytics import YOLO
import sys
# 选择类型：det, seg, pose, obb
model_type= sys.argv[1]
# 权重位置
input_model = sys.argv[2]

if model_type == 'det':
    # 转为onnx：注意需要修改为对应的模型配置文件
    model = YOLO("./weights/yolov8n.pt")
    results = model(task='detect', mode='predict', source='./test.jpg', imgsz=640, line_width=3, show=True, save=True, device='cpu')
elif model_type == 'seg':
    model = YOLO("./weights/yolov8n-seg.pt")
    results = model(task='detect', mode='predict', source='./test.jpg', imgsz=640, line_width=3, show=True, save=True, device='cpu')
elif model_type == 'pose':
    model = YOLO("./weights/yolov8n-pose.pt")
    results = model(task='detect', mode='predict', source='./test.jpg', imgsz=640, line_width=3, show=True, save=True, device='cpu')
elif model_type == 'obb':
    model = YOLO("./weights/yolov8n-obb.pt")
    results = model(task='detect', mode='predict', source='./test.jpg', imgsz=640, line_width=3, show=True, save=True, device='cpu')

```

以检测为例

将代码中的

```python
model = YOLO("./weights/yolov8n.pt")
```

改为你自己对应的模型文件

第四步：运行

如：检测模型

```bash
python export-onnx.py det ./weights/yolov8n.pt
```

其他模型依次类推

运行可能会报错，但是不影响onnx模型生成，只要能生成onnx模型，不必理会报错

至此就可以得到6个输出头的onnx目标检测/10个输出头的分割/9个输出头的姿态/9个输出头的旋转的onnx模型，可在netron中自行检查