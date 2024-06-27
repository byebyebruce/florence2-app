# Florence2 App

> 本项目是将[Microsoft Florence2](https://huggingface.co/microsoft/Florence-2-large)([Paper](https://arxiv.org/abs/2311.06242) | [Model](https://huggingface.co/microsoft/Florence-2-large) | [Notebook](https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb)) 模型封装成 Flask 的 API 和 CLI 工具 。它可以完成多种[视觉任务](#支持视觉任务列表)，如生成描述，执行目标检测、OCR 等。


## 功能特点
- **API 模式**：提供一个API Server来处理图像任务请求。
- **CLI 模式**：用户通过命令行直接输入任务和图像路径。

## 安装

1. **克隆仓库**：
    ```sh
    git clone https://github.com/byebyebruce/florence2-app.git
    cd florence2-app
    ```

2. **创建虚拟环境**：
    ```sh
	conda create -n florence2-app python=3.11
	conda activate florence2-app
    ```

3. **安装依赖**：
    ```sh
    pip install -r requirements.txt
    ```

## 使用方法

### 运行 API Server

```sh
python3 main.py api
```

#### 请求示例：
1. 目标检测
```sh
curl -X POST http://localhost:5000/api/predict \
    -F "image=@testdata/car.jpg" \
    -F "task=OD"
```

2. 图片描述
```sh
curl -X POST http://localhost:5000/api/predict \
    -F "image=@testdata/car.jpg" \
    -F "task=CAPTION"
```

3. 特定区域分割
```sh
curl -X POST http://localhost:5000/api/predict \
    -F "image=@testdata/car.jpg" \
    -F "task=REFERRING_EXPRESSION_SEGMENTATION" \
    -F "prompt=the car"
```

### 运行 CLI 工具

```sh
python3 main.py cli OD ./testdata/car.jpg
```

## 支持视觉任务列表
| Task Prompt                          | Description                           |
|--------------------------------------|---------------------------------------|
| DETAILED_CAPTION                     | 详细图像描述                              |
| MORE_DETAILED_CAPTION                | 更详细图像描述                             |
| CAPTION_TO_PHRASE_GROUNDING          | 图像描述到定语                              |
| DENSE_REGION_CAPTION                 | 密集区域描述                               |
| OD                                   | 目标检测                                 |
| REGION_PROPOSAL                      | 候选区域                                 |
| OCR                                  | 字符识别                                 |
| OCR_WITH_REGION                      | 区域字符识别                               |
| REFERRING_EXPRESSION_SEGMENTATION    | REFERRING_EXPRESSION_SEGMENTATION     |
| REGION_TO_SEGMENTATION               | 区域分割                                 |
| OPEN_VOCABULARY_DETECTION            | 开放词汇下的目标检测                         |
| REGION_TO_CATEGORY                   | 区域类别                                 |
| REGION_TO_DESCRIPTION                | 区域描述                                 |
