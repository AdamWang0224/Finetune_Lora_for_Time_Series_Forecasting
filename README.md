# M2 Coursework: LoRA Fine-Tuning for Time Series Forecasting

This project explores fine-tuning the Qwen2.5-0.5B-Instruct model using Low-Rank Adaptation (LoRA) for forecasting predator-prey time series modeled by Lotka–Volterra dynamics.

## Environment Setup
To create the environment with conda, run the following commands in the terminal:

```bash
conda create -n m2 python=3.10
conda activate m2
pip install torch>=2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

(Optional) To use this conda environment in Jupyter Notebook, run the following command in the terminal:

```bash
pip install ipykernel
python -m ipykernel install --user --name m2 --display-name "Python (m2)"
```

> ⚠️ **Recommendation:** We strongly recommend installing the environment via **Conda** for reproducible installation, especially with GPU support. 

## Data Preparation
You should use the following command to create a data folder under the root dir and put the data in it:

```bash
mkdir data
```

And the structure of the data folder is as following:

```bash
data/
├── lotka_volterra_data.h5
```

Then you should run the following code in your terminal to create training and evaluation set:

```bash
python -m src.data_processing.split_train_eval_set
```

Now your data folder should be like:

```bash
data/
├── lotka_volterra_data.h5
├── lotka_volterra_train.h5
├── lotka_volterra_test.h5
```

Two example sequences, their preprocessed forms, and tokenized results can be found in **m2_preprocess_and_flops.ipynb**.

## LoRA Finetuning

Use the following scripts to run fine-tuning experiments.

🔧 Basic LoRA fine-tuning:
```bash
./scripts/finetune_lora.sh
```

🔬 Ablation Studies:

(1) Varying learning rate:
```bash
./scripts/lora_ablation_lr.sh
```
(2) Varying LoRA rank:
```bash
./scripts/lora_ablation_rank.sh
```
(3) Varying context length:
```bash
./scripts/lora_ablation_context.sh
```

🏆 Best hyperparameter setting (final training run):
```bash
./scripts/finetune_lora_best_model.sh
```

## Model Evaluation

🔍 Evaluation with default setup:
```bash
./scripts/evaluate.sh
```

🌡️ Evaluation with temperature sampling (optional):
```bash
./scripts/evaluate_temperature.sh
```

## Calculate FLOPS

For FLOPs Calculation, simply run:

```bash
python src/flops.py
```

You can also modify the experiment settings in the script if you want to change the model architecture. The breakdown of the FLOPs is as follows:

### Training FLOPs Usage Breakdown

| Experiment            | Steps | L   | r | log₁₀(LoRA FLOPs) | log₁₀(Qwen FLOPs) | log₁₀(Total FLOPs) |
|-----------------------|-------|-----|---|--------------------|--------------------|----------------------|
| Basic Setting         | 2000  | 512 | 4 | 12.822             | 16.131             | 16.131               |
| Learning Rate = 5e-5  | 2000  | 512 | 4 | 12.822             | 16.131             | 16.131               |
| Learning Rate = 1e-4  | 2000  | 512 | 4 | 12.822             | 16.131             | 16.131               |
| LoRA Rank = 2         | 2000  | 512 | 2 | 12.521             | 16.131             | 16.131               |
| LoRA Rank = 8         | 2000  | 512 | 8 | 13.123             | 16.131             | 16.131               |
| Context = 128         | 1000  | 128 | 8 | 12.220             | 15.214             | 15.215               |
| Context = 512         | 1000  | 512 | 8 | 12.822             | 15.830             | 15.830               |
| Context = 768         | 1000  | 768 | 8 | 12.999             | 16.014             | 16.015               |
| Best Setting          | 2000  | 512 | 8 | 13.123             | 16.131             | 16.131               |

> Total training cost across all experiments is approximately **$10^{16.9995}$ FLOPs**.


To use calflops to calculate FLOPS, run the following code:

```bash
python src/utils/calflops_FLOPS_validation.py
```

Note that you may need to modify whether to use Lora and its rank in the script.

## Declaration of Using Generative AI Tools

During the development of this coursework, generative AI tools (such as ChatGPT) were used to assist with debugging and to improve the clarity of the code and aesthetics of plotting. After using these tools, the author reviewed and edited the content as needed and takes full responsibility for the final submission.