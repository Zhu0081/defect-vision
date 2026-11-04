# src/cpp —— C++ 工具与端侧推理

这里的代码有两个目的：

1. **工具**：检查数据集标注文件（对应 Python 的 `src\data\check_labels.py`）
2. **端侧推理**：用 ONNX Runtime 的 C 接口跑模型，不经过 Python

面试时能讲的话：**同一个检测算法，我用 Python 训练、用 C++ 部署，两条路的结果对得上。**

---

## 环境

| 组件 | 版本 | 位置 |
|---|---|---|
| 编译器 | MinGW-w64 g++ **16.2.0** | `F:\tools\mingw64\bin` |
| ONNX Runtime | **1.31.1**（win-x64 CPU） | `F:\tools\onnxruntime-win-x64-1.31.1` |

编译前先把这两条路径加到 PATH：

```powershell
$env:PATH = "F:\tools\mingw64\bin;F:\tools\onnxruntime-win-x64-1.31.1\lib;" + $env:PATH
```

---

## 1. hello.cpp —— 最小可运行程序

```powershell
g++ hello.cpp -o hello.exe -std=c++17
.\hello.exe
```

## 2. count_label.cpp —— 读文件、数缺陷

```powershell
g++ count_label.cpp -o count_label.exe -std=c++17
.\count_label.exe ..\..\data\yolo\labels\test\card_p022_scratch_002.txt
```

输出：

```
file  : ..\..\data\yolo\labels\test\card_p022_scratch_002.txt
lines : 2  (defects)
  #1  points = 8
  #2  points = 8
```

## 3. check_label.cpp —— 四道格式校验

检查：类别号 0~3、至少 3 个点、坐标个数为偶数、坐标在 0~1。

```powershell
g++ check_label.cpp -o check_label.exe -std=c++17
.\check_label.exe ..\..\data\yolo\labels\test\card_p022_scratch_002.txt
```

合格返回 `0` 并打印 `RESULT: PASS`；有问题返回 `1` 并逐行指出。

## 4. ort_detect.cpp —— ONNX Runtime C++ 推理

### 第 1 步：生成输入张量

ONNX Runtime 的 C++ 接口只吃裸 `float32` 数组，不做图片解码和缩放。所以先用 Python 生成：

```powershell
cd F:\defect-vision
python src\cpp\make_input.py data\captured\light_side\card\card_p022_scratch_002.jpg
```

产出 `src\cpp\input_1x3x1280x1280.bin`（18.75 MB）与 `input_meta.txt`。

### 第 2 步：编译

```powershell
cd F:\defect-vision\src\cpp
g++ ort_detect.cpp -o ort_detect.exe -std=c++17 -O2 -static-libgcc -static-libstdc++ `
    "-IF:\tools\onnxruntime-win-x64-1.31.1\include" `
    "-LF:\tools\onnxruntime-win-x64-1.31.1\lib" -lonnxruntime
```

⚠ **必须把 `onnxruntime.dll` 拷到 exe 旁边**。否则 Windows 会去 PATH 里找，
可能撞上别的版本，报 `The requested API version [31] is not available, only [1, 17]`。

```powershell
Copy-Item F:\tools\onnxruntime-win-x64-1.31.1\lib\onnxruntime*.dll .
```

### 第 3 步：运行

```powershell
.\ort_detect.exe ..\..\models\best.onnx input_1x3x1280x1280.bin input_meta.txt
```

实测输出：

```
ONNX Runtime 版本: 1.31.1
模型: ..\..\models\best.onnx  11.7 MB
会话创建成功
输入: images   (共 1 个)
输出[0]: output0
输出[1]: output1
输入张量: 4915200 个 float  (18.75 MB)
推理耗时: 178.2 ms
输出形状: [1, 300, 38]
候选框: 300 行 × 38 列  (总计 11400 个数)
  #0  scratch  conf=0.946  box=(728.1, 576.9, 905.1, 731.1)
  #1  scratch  conf=0.792  box=(394.4, 451.7, 702.7, 879.5)
阈值 0.25 下命中 2 个
```

**和 Python 服务端对同一张图的结果完全一致**（scratch 0.946 / 0.792）—— 这是最重要的一条：
说明预处理、模型、后处理三处都没有搞错。

---

## 关键点（面试可讲）

**为什么用 C 接口而不是 C++ 接口？**
`onnxruntime_cxx_api.h` 会把 `std::string`、`std::vector` 传进 dll。ONNX Runtime 的 dll 是 MSVC 编的，
我们用 MinGW 编 —— 两边的 `std::` 内部布局不一样。C 接口只传裸指针和整数，没有 ABI 问题。

**输出 `[1, 300, 38]` 是什么？**
300 个候选框，每行 38 个数：`x1, y1, x2, y2, score, class_id` + 32 个分割 mask 系数（本项目暂时丢掉）。

**框坐标在哪个空间？**
在 letterbox 后 1280×1280 的画布上，不是原图像素。要还原得用 `input_meta.txt` 里的
`r / left / top`：`原图 = (框坐标 - padding) / r`。

**为什么要有 `make_input.py`？**
因为解码 JPEG、等比缩放、归一化这些事，C++ 侧要么引 OpenCV（体积大），
要么手写（工作量大）。先把「推理」这一段单独打通、和 Python 结果对上，
再去补解码 —— 这是把一个大问题拆成两个可验证的小问题的做法。

---

## 已知限制

- `ort_detect.cpp` 里的输入形状写死为 `1x3x1280x1280`，换模型要改源码（下一步从模型元信息读）
- 只取前 6 列，分割 mask 没有还原成掩膜
- 没有做 letterbox 反变换，打印的是画布坐标
- 单线程跑单张图，没有 batching
