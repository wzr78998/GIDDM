

# GIDDM: Generating Labels with Diffusion Model to Promote Cross-domain Open-set Image Recognition

<div align="center">

*Implementation of the paper GIDDM: **G**enerating Labels w**i**th **D**iffusion **M**odel to Promote Cross-domain Open-set Image Recognition*

</div>

---

## 📋 项目概览 (Project Overview)

本项目是论文 **GIDDM** 的官方实现：
**GIDDM: Generating Labels with Diffusion Model to Promote Cross-domain Open-set Image Recognition**

## ✨ 主要特性 (Main Features)

- 🚀 **多源数据集支持**: 全面支持 Houston、Pavia 等多种高光谱数据集。
- 🎯 **专用处理流程**: 针对高光谱图像特性，优化了数据处理、加载和增强流程。
- 📈 **完整框架**: 提供从数据准备到模型训练、评估和结果分析的完整工作流。

## 🛠️ 环境要求 (Environment Requirements)

| 依赖项 (Dependency) | 版本要求 (Version) |
| :------------------ | :------------------- |
| **Python**          | `3.6+`               |
| **PyTorch**         | `1.7+`               |
| **CUDA**            | `10.0+` (可选, 用于 GPU 加速) |
| **其他依赖**        | `requirements.txt`   |

## 📦 安装指南 (Installation Instructions)

### 1. 克隆项目 (Clone the Project)

```bash
git clone [project_url]
cd GIDDM
```

### 2. 安装依赖 (Install Dependencies)

```bash
pip install -r requirements.txt
```

### 3. 准备数据集 (Prepare the Dataset)

- 将数据集文件放置在 `data/` 目录下。
- 确保数据格式符合要求（`.mat` 文件）。

---

## 📚 数据集准备 (Dataset Preparation)

### 支持的数据集 (Supported Datasets)

#### 🏙️ Houston 数据集

| 项目 (Item)         | 详情 (Details)         |
| :------------------ | :----------------------- |
| **源域 (Source)**   | `Houston13.mat`          |
| **目标域 (Target)**   | `Houston18.mat`          |
| **训练脚本**        | `train_HS.py`            |

#### 🏛️ Pavia 数据集

| 项目 (Item)         | 详情 (Details)         |
| :------------------ | :----------------------- |
| **源域 (Source)**   | `paviaC.mat`             |
| **目标域 (Target)**   | `paviaU.mat`             |
| **训练脚本**        | `train-Pavia.py`         |

### 格式要求 (Data Format Requirements)

- ✅ 数据文件应为 `.mat` 格式。
- ✅ 标签文件需包含对应的 Ground Truth 信息。
- ✅ 数据应按照指定的目录结构进行组织。

---

## 🚀 使用方法 (Usage)

### 基础训练命令 (Basic Training Commands)

#### 训练 Houston 数据集
```bash
python train_HS.py
```

#### 训练 Pavia 数据集
```bash
python train-Pavia.py
```

### 配置参数 (Configuration Parameters)

主要配置参数定义在 `config_HSI_PCPU.py` 文件中：

#### 📊 数据级参数

| 参数 (Parameter)        | 描述 (Description)         | 示例值 (Example)        |
| :-------------------- | :------------------------- | :---------------------- |
| `--dataset`           | 数据集名称                 | `Houston`, `Pavia`      |
| `--target_domain`     | 目标域名称                 | `Houston18`, `paviaU`   |

### 模型训练工作流 (Model Training Workflow)

1.  **📥 数据加载**: 从指定路径加载源域和目标域数据。
2.  **🔧 数据预处理**: 执行标签映射和数据增强。
3.  **🏗️ 模型构建**: 初始化 GIDDM 模型。
4.  **⚙️ 训练初始化**: 设置训练参数和优化器。
5.  **🚀 模型训练**: 执行域适应训练。
6.  **💾 保存结果**: 保存训练日志、模型权重和评估结果。

---

## 📁 项目结构 (Project Structure)

```
GIDDM/
├── 📄 config_HSI_PCPU.py      # 配置文件
├── 🐍 train_HS.py             # Houston 数据集训练脚本
├── 🐍 train-Pavia.py          # Pavia 数据集训练脚本
├── 📖 README.md               # 项目文档
│
├── 🧠 models/                 # 模型定义
│   ├── model_PCPU_HSI.py     # GIDDM 模型实现
│   ├── model_HSI_HS.py       # HSI 模型实现
│   ├── basenet.py            # 基础网络
│   └── function.py            # 工具函数
│
├── 📊 data_loader/            # 数据加载器
│   ├── get_loader.py         # 数据加载接口
│   ├── mydataset.py          # 数据集类
│   └── base.py               # 基础数据集类
│
├── 📁 data/                   # 数据集目录
│   ├── Houston/
│   ├── Pavia/
│   └── ...
│
├── 📈 results/                # 结果保存目录
└── 🛠️ utils/                  # 工具函数
```

---

## 📚 引用 (Citation)

如果您的研究使用了本项目，请引用相关论文：

```bibtex
[在此处添加相关论文的 BibTeX 引用]
```

## 📄 许可证 (License)

本项目基于 [MIT License](LICENSE) 发布。详情请参阅 `LICENSE` 文件。



## 📞 联系方式 (Contact)

如果您有任何问题，欢迎通过以下方式联系我们：

- **📧 邮箱**: [wanghaoyucumt@163.com]


---

<div align="center">

**⭐ 如果您觉得这个项目对您有帮助，请给我们一个 Star！ ⭐**

</div>
