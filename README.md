## KAI-BERT: An Educational BERT Implementation 📖

Welcome to KAI-BERT, a clear and academic-friendly implementation of BERT, designed to help researchers and engineers understand its building blocks without unnecessary complexity.

This project follows the classic BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al.) and focuses on clarity over speed, making it perfect for those who want to learn, experiment, and extend. 🚀

### 🏗️ What’s Inside?
- 🧩 Modeling BERT from scratch: Check out modeling.py to see the architecture unfold.
- 🔍 Masked Language Modeling (MLM) training setup.
- 📜 Config-driven design: Hyperparameters and model settings live in configuration.py.
- ⚡ PyTorch Lightning for structured and modular training.

### 📂 Project Structure
```
📂 kai-bert  
├── data/  
│   ├── config.json  # Model & training configs  
│   ├── model.safetensors  # Pretrained weights  
├── scripts/  # Utility scripts (training, evaluation, etc.)  
├── src/  
│   ├── bert.py  # Core BERT model  
│   ├── configuration.py  # Model hyperparams  
│   ├── modeling.py  # Educational breakdown of BERT’s components  
│   ├── modeling_outputs.py  # Utilities for handling model outputs  
│   ├── __init__.py  # Python module  
├── train.py  # Step-by-step training script for MLM 🏋️  
├── playbook.ipynb  # Interactive notebook to experiment 🎭  
├── playbook.py  # Script version of the notebook  
├── .gitignore  # Keep things tidy  
├── README.md  # You're reading this! 📖  
```
## 🚀 Installation & Setup
### 1️⃣ Clone the repo
```bash
git clone https://github.com/abidikhairi/kai-bert.git  
cd kai-bert  
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt  
```

### 3️⃣ Train your own BERT (step-by-step)

```bash
python train.py  
```


## ✅ To-Do List
### 📌 Educational Enhancements:
- [ ] Add diagrams to explain BERT's computational graph 🖼️
- [ ] Visualize Linear LR Scheduler 📉

### 📌 Training & Experimentation:
- [ ] Add train.py for Masked Language Modeling (MLM) 🏋️
- [ ] Implement custom data collator for efficient batching 📦
- [ ] Experiment with different optimizers & schedulers ⚡

### 📌 General Improvements:
- [ ] Add inference script for testing 🏎️
- [ ] Improve documentation with more explanations 📝

## 🎯 Who Is This For?
- ✅ Students & Researchers learning how BERT works under the hood.
- ✅ Engineers who want a structured, readable codebase for experimentation.
- ✅ Anyone curious about transformer architectures without the usual complexity.

This is not optimized for speed but instead prioritizes clarity and educational value.


## 🤖 Contributions
Got a cool idea? Found a bug? PRs and issues are always welcome! 🎉

