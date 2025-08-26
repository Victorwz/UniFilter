# UniFilter

Offcial code for our work [Train a Unified Multimodal Data Quality Classifier with Synthetic Data]


## Release
<!-- - [3/31/2025] 🔥 We released all pre-training data in webdataset format at [Open-Qwen2VL-Data](https://huggingface.co/datasets/weizhiwang/Open-Qwen2VL-Data).
- [3/31/2025] 🔥 We released the technical report for [**Open-Qwen2VL**](https://arxiv.org/abs/2504.00595). -->
- [8/25/2025] 🔥 We released UniFilter model at [UniFilter-Qwen2.5-1.5B](https://huggingface.co/weizhiwang/UniFilter-Qwen2.5-1.5B). Empowered by a strong 1.5B LLM backbone, the UniFilter model achieves best inference speed on quality score generation and the classification accuracy.


## Introduction
UniFilter is a Unified Multimodal Data Quality Classifier for High-Quality Multimodal Data Filtering, which can generate quality scores for both image-text caption and interleaved document data. Such quality scores can be further used for high-quality data filtering to significantly strengthen the capability of pre-trained MLLMs.

This repo supports
 - synthetic data generation
 - UniFilter training-
 - quality score generation with [UniFilter-Qwen2.5-1.5B](https://huggingface.co/weizhiwang/UniFilter-Qwen2.5-1.5B).

## Installation
If you just require the quality score generation, please install the customized LLaVA package only.

```Shell
conda create -n unifilter python=3.10
conda activate unifilter
pip install -e LLaVA
pip install flash-attn==2.5.2 --no-build-isolation
```

## Synthetic Data Generation for UniFilter Training
We instruct Claude-3 or Claude-3.5 to generate the desired (multimodal data example, quality score) pairs across 4 designated quality levels.
The synthetic data generation scrips are:
 - [claude_sonnet_caption_data_generation.py](data_prepare/caption_data_scripts/claude_sonnet_caption_data_generation.py)
 - [claude_sonnet_interleaved_data_generation.py](data_prepare/interleaved_data_scripts/claude_sonnet_interleaved_data_generation.py)

## Data Preparation for UniFilter Training
UniFilter is trained a large-scale set of (multimodal data example, quality score) pairs, which contains both caption data and interleaved document data. The synthetic multimodal example-score paired data are available at [UniFilter-Post-Train-Data]().

## UniFilter Training
We develop the UniFilter training and scoring codebase based on [LLaVA-Unified]() repo, which is adapted from LLaVA with the support for recent LLMs and Vision Encoders. 
<!-- An additional [LlavaPhi3Classifier](LLaVA/llava/model/language_model/llava_phi3.py#235) class is customized as the model class for UniFilter. -->

The architectural design of UniFilter contains three modules, the vision encoder, the visual projector, and the LLM Backbone. Different from a MLLM, the LLM Backbone does not have a language modeling head and we replace it with a score generation head. All these module parameters are specified with:
- `--mm_projector_type`: visual projector, i.e. aapool_mlp representing average pooling vision projector with 144 tokens for one image
- `--vision_tower`: vision encoder, i.e. SigLIP-SO-400M with 384px resolution
- `--model_name_or_path`: LLM Backbone, i.e. Qwen2.5-0.5B-Instruct


### Visual Projector Pre-Training (Stage 1)

Please download the 558K subset of the LLAVA-Pretrain caption dataset [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

Training script with DeepSpeed ZeRO-2: [`pretrain.sh`](scripts/v1_5/pretrain.sh).


### UniFilter Classifier Training (Stage 2)


Training script with DeepSpeed ZeRO-3: [`train_classifier.sh`](scripts/v1_5/train_classifier.sh).

Our training script will upload the metrics to wandb. The best UniFilter model is saved based on the best quality classification accuracy on the validation sets.


## Quality Score Generation

## Caption Data Quality Scoring
```Shell
python data_scoring/data_quality_classifier_caption_scoring.py \
    --model-path weizhiwang/UniFilter-Qwen2.5-1.5B \
    --tar-file-path data/datacomp/medium_vanilla_filter\ 
    --gpu-id 0 \
    --batch-size 4 \
    --tars-per-gpu 256 \
```

## Interleaved Data Quality Scoring
```Shell
python data_scoring/data_quality_classifier_interleaved_scoring.py \
    --model-path weizhiwang/UniFilter-Qwen2.5-1.5B \
    --tar-file-path data/OBELICS/obelics_webdataset\ 
    --gpu-id 0 \
    --batch-size 1 \
    --tars-per-gpu 128 \
```

Parameters to note:
- `--gpu-id`: for large-scale score generation using multi-machines, specify the index of machines
- `--model-path`: path to the UniFilter model checkpoint
- `--tar-file-path`: path to the webdataset image-text caption data or interleaved document data tars
- `--tars-per-gpu`: the number of webdataset tars for a single-gpu to inference on


## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon for UniFilter training.

