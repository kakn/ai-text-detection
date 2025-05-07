# Hybrid Feature-Embedding Models for Robust AI Text Detection

This project contains the code for a publication on 'AI Text Detection', supervised by Christian Hardmeier. 
- 'src' contains folders 'llm' and 'feature', which detail the LLM and feature-based approaches used in the thesis
- 'data' contains all the figures, datasets and metrics used in the thesis, including the generated evasive texts. However, some files are very large, so they are excluded from Git. You may need to run some code to generate them, or you can contact me for the data (kasper.knudsen@gmail.com)

## Resource requirements
Experiments were conducted on university HPC resources, utilizing multi-core CPUs (up to 64 cores) for feature extraction and XGBoost training, and NVIDIA A100 GPUs (40-80GB VRAM) for DistilBERT fine-tuning and Hybrid FFNN training. Loading pre-computed embeddings required substantial system RAM (approaching 1000GB).
