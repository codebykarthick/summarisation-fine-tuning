# Summarisation Fine Tuning
## Introduction
The aim of this project is to create a model that is capable of performing abstractive summarisation through a fine tuned model. Trying to achieve this by fine-tuning a T5 model on publicly available datasets and then measuring its performance on summarisation metrics such as BERTScore. The explanation and background information can be found in this [blog post.](https://dev.to/sri_harikarthik_909342ac/fine-tuning-a-language-model-for-summarisation-using-lora-3lg)

## Setup
The main execution occurs in the `main.ipynb` jupyter notebook and every process is commented and documented extensively.

Please create a fresh conda or venv environment on python 3.12 (this is important as some dependency libraries need that version) to install the dependencies in `requirements.txt`.
