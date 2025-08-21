
# GIDDM: Generating Labels with Diffusion Model to Promote Cross-domain Open-set Image Recognition

## ✨ Main Features

- 🚀 **Multi-Dataset Support**: Comprehensive support for multiple hyperspectral datasets, including Houston, Pavia, etc.
- 🎯 **Specialized Pipeline**: An optimized data processing, loading, and augmentation pipeline tailored for the characteristics of hyperspectral images.
- 📈 **Complete Framework**: A full workflow from data preparation to model training, evaluation, and results analysis.

## 🛠️ Environment Requirements

| Dependency         | Version Requirement |
| :----------------- | :------------------ |
| **Python**         | `3.6+`              |
| **PyTorch**        | `1.7+`              |
| **CUDA**           | `10.0+` (Optional, for GPU acceleration) |
| **Other Packages** | See `requirements.txt` |

## 📦 Installation Instructions

### 1. Clone the Project

```bash
git clone [project_url]
cd GIDDM
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

-   Place the dataset files in the `data/` directory.
-   Ensure the data format meets the requirements (`.mat` files).
-   the data set can be got at:  https://pan.baidu.com/s/1WHzeZkE1zhG8M-54gdyW6A  code: 6az7 

---

## 📚 Dataset Preparation

### Supported Datasets

#### 🏙️ Houston Dataset

| Item            | Details           |
| :-------------- | :---------------- |
| **Source**      | `Houston13.mat`   |
| **Target**      | `Houston18.mat`   |
| **Training Script** | `train_HS.py` |

#### 🏛️ Pavia Dataset

| Item            | Details            |
| :-------------- | :----------------- |
| **Source**      | `paviaC.mat`       |
| **Target**      | `paviaU.mat`       |
| **Training Script** | `train-Pavia.py` |

### Data Format Requirements

-   ✅ Data files must be in `.mat` format.
-   ✅ Label files must contain the corresponding ground truth information.
-   ✅ Data should be organized according to the specified directory structure.

---

## 🚀 Usage

### Basic Training Commands

#### Train on Houston Dataset

```bash
python train_HS.py
```

#### Train on Pavia Dataset

```bash
python train-Pavia.py
```

### Configuration Parameters

Key configuration parameters are defined in the `config_HSI_PCPU.py` file:

#### 📊 Data-level Parameters

| Parameter         | Description        | Example Value           |
| :---------------- | :----------------- | :---------------------- |
| `--dataset`       | Dataset name       | `Houston`, `Pavia`      |
| `--target_domain` | Target domain name | `Houston18`, `paviaU`   |

### Model Training Workflow

1.  **📥 Data Loading**: Load source and target domain data from the specified path.
2.  **🔧 Data Preprocessing**: Perform label mapping and data augmentation.
3.  **🏗️ Model Construction**: Initialize the GIDDM model.
4.  **⚙️ Training Setup**: Configure training parameters and the optimizer.
5.  **🚀 Model Training**: Execute the domain adaptation training process.
6.  **💾 Save Results**: Save training logs, model weights, and evaluation results.

---

## 📁 Project Structure

```
GIDDM/
├── 📄 config_HSI_PCPU.py      # Configuration file
├── 🐍 train_HS.py             # Training script for Houston dataset
├── 🐍 train-Pavia.py          # Training script for Pavia dataset
├── 📖 README.md               # Project documentation
│
├── 🧠 models/                 # Model definitions
│   ├── model_PCPU_HSI.py     # GIDDM model implementation
│   ├── model_HSI_HS.py       # HSI model implementation
│   ├── basenet.py            # Base network
│   └── function.py            # Utility functions
│
├── 📊 data_loader/            # Data loaders
│   ├── get_loader.py         # Data loading interface
│   ├── mydataset.py          # Custom dataset class
│   └── base.py               # Base dataset class
│
├── 📁 data/                   # Dataset directory
│   ├── Houston/
│   ├── Pavia/
│   └── ...
│
├── 📈 results/                # Directory for saving results
└── 🛠️ utils/                  # Utility functions
```

---

## 📚 Citation

If you use this project in your research, please cite the related paper:

```bibtex
[BibTeX citation for the paper here]
```

## 📄 License

This project is released under the [MIT License](LICENSE). See the `LICENSE` file for more details.



## 📞 Contact

If you have any questions, feel free to reach out:

-   **📧 Email**: [wanghaoyucumt@163.com]


---

<div align="center">

**⭐ If you find this project useful, please give us a star! ⭐**

</div>
