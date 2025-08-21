

# GIDDM: Hyperspectral Image Domain Adaptation Project  

## 📋 Project Overview  

This is the implementation of the paper **GIDDM**:  
**GIDDM: Generating Labels with Diffusion Model to Promote Cross-domain Open-set Image Recognition**  

## ✨ Main Features  

- 🚀 Supports multiple hyperspectral datasets (Houston, Pavia, etc.)  
- 🎯 Optimized data processing pipeline for the characteristics of hyperspectral images  
- 📈 Complete training and evaluation framework  

## 🛠️ Environment Requirements  

| Dependency | Version Requirement |
|------------|---------------------|
| Python     | 3.6+                |
| PyTorch    | 1.7+                |
| CUDA       | 10.0+ (Optional, for GPU acceleration) |
| Other dependencies | See `requirements.txt` |

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
- Place dataset files in the `data/` directory  
- Ensure the data format meets the requirements (`.mat` files)  

## 📚 Dataset Preparation  

### Supported Datasets  

#### 🏙️ Houston Dataset  
| Item         | Details              |
|--------------|----------------------|
| **Source**   | `Houston13.mat`      |
| **Target**   | `Houston18.mat`      |
| **Training Script** | `train_HS.py` |

#### 🏛️ Pavia Dataset  
| Item         | Details              |
|--------------|----------------------|
| **Source**   | `paviaC.mat`         |
| **Target**   | `paviaU.mat`         |
| **Training Script** | `train-Pavia.py` |

### 📋 Data Format Requirements  

- ✅ Data files should be in `.mat` format  
- ✅ Label files should contain the corresponding ground truth information  
- ✅ Data should be organized according to the specified directory structure  

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

The main configuration parameters are defined in the `config_HSI_PCPU.py` file:  

#### 📊 Data-level Parameters  
| Parameter         | Description                | Example Value           |
|-------------------|----------------------------|-------------------------|
| `--dataset`       | Dataset name               | Houston, Pavia          |
| `--target_domain` | Target domain name          | Houston18, paviaU       |  

### Training Examples  

#### 🎯 Standard Training  
```bash
# Pavia Dataset
python train-Pavia.py

# Houston Dataset
python train_HS.py
```

### 🔄 Model Training Workflow  

```mermaid
1. **📥 Data Loading**: Load source domain and target domain data from the specified path  
2. **🔧 Data Preprocessing**: Perform label mapping and data augmentation  
3. **🏗️ Model Construction**: Initialize the GIDDM model  
4. **⚙️ Training Initialization**: Set training parameters and optimizer  
5. **🚀 Model Training**: Execute domain adaptation training  
6. **💾 Save Results**: Save training results and models  
```

## 📁 Project Structure  

```
GIDDM/
├── 📄 config_HSI_PCPU.py          # Configuration file
├── 🐍 train_HS.py                 # Houston dataset training script
├── 🐍 train-Pavia.py              # Pavia dataset training script
├── 🧠 models/                     # Model definitions
│   ├── model_PCPU_HSI.py         # GIDDM model implementation
│   ├── model_HSI_HS.py           # HSI model implementation
│   ├── basenet.py                # Base network
│   └── function.py                # Utility functions
├── 📊 data_loader/                # Data loaders
│   ├── get_loader.py             # Data loading interface
│   ├── mydataset.py              # Dataset class
│   └── base.py                   # Base dataset class
├── 📁 data/                       # Dataset directory
│   ├── Houston/                  # Houston dataset
│   ├── Pavia/                    # Pavia dataset
│   ├── KSC/                      # KSC dataset
│   └── office/                   # Office dataset
├── 🛠️ utils/                      # Utility functions
├── 📈 results/                    # Results saving directory
└── 📖 README.md                   # Project documentation
```

## 📚 Citation  

If you use this project in your research, please cite the related paper:  

```bibtex
[Related paper citation]
```

## 📄 License  

This project is released under the MIT License. See the LICENSE file for details.  

## 🤝 Contributing  

Contributions are welcome via Issues and Pull Requests to improve the project!  

- 🐛 Report bugs  
- 💡 Suggest new features  
- 📝 Improve documentation  
- 🔧 Contribute code  

## 📞 Contact  

If you have any questions, please contact:  

- 📧 Email: [wanghaoyucumt@163.com]  
- 🐙 GitHub Issues: [project_url]/issues  

---

<div align="center">

**⭐ If you find this project useful, please give us a star! ⭐**

</div>  
