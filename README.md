# Summarisation Fine Tuning
## Introduction
The aim of this project is to create a model that is capable of performing abstractive summarisation through a fine tuned model. Trying to achieve this by fine-tuning a T5 model on publicly available datasets and then measuring its performance on summarisation metrics such as BERTScore. The explanation and background information can be found in this [blog post.](https://dev.to/sri_harikarthik_909342ac/fine-tuning-a-language-model-for-summarisation-using-lora-3lg)

## Setup
The main execution occurs in the `main.ipynb` jupyter notebook and every process is commented and documented extensively.

Please create a fresh conda or venv environment on python 3.12 (this is important as some dependency libraries need that version) to install the dependencies in `requirements.txt`.

## 📊 Results

Evaluation was performed using **BERTScore F1**, which measures semantic similarity between generated and reference summaries.

| Model            | BERTScore F1 |
|------------------|--------------|
| Vanilla T5-small | **0.8594**   |
| LoRA Fine-tuned  | **0.8665**   |

- Even with **only one epoch of training** and updating just **0.48% of model parameters**, LoRA-based fine-tuning achieved a measurable performance gain.  
- Training was done on the **CNN/DailyMail dataset** using Hugging Face Transformers.  
- Further improvement is possible with additional epochs, larger model variants (e.g., T5-base), or domain-specific datasets.  

---

### Limitations
- Fine-tuned for only one epoch (budget-constrained).  
- Dataset not domain-specific (CNN/DailyMail, which overlaps with T5’s pretraining).  
- Training limited to 512 input tokens and 128 output tokens for efficiency.  
- Not optimised for deployment (focus was on adapter-based fine-tuning process).  

---

### Conclusion
This project demonstrates how **LoRA adapters** enable efficient fine-tuning of large language models with minimal compute and cost. Even under constrained conditions, LoRA fine-tuning produced a **+0.71% improvement in BERTScore F1** over the vanilla model.
