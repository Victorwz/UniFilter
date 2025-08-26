from transformers import AutoTokenizer, TextGenerationPipeline
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "/fsx-Training/shopqa-training-fsx-prod-us-east-1/home/wzwang/checkpoints/model_arch_ablations_v3_40k_data/data_quality_classifier_llava_phi3_mini_caption_only_mse_loss_siglip_384_mmtoken_144_class_4"

quant_path = "data_quality_classifier_llava_phi3_mini_caption_only_mse_loss_siglip_384_mmtoken_144_200k_data-4bit-128g"


quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)#, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# logging.basicConfig(
#     format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
# )

"""
Download https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-preview to local
Make following edits to the config.json
LlavaLlamaForCausalLM -> LlamaForCausalLM
"model_type": "llava" -> "llama"
"""


# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
# examples = [
#     tokenizer(
#         "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
#     )
# ]

# quantize_config = BaseQuantizeConfig(
#     bits=4,  # quantize model to 4-bit
#     group_size=128,  # it is recommended to set the value to 128
#     desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad 
# )
# print("start to load")
# # load un-quantized model, by default, the model will always be loaded into CPU memory
# model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
# print("start to quantize")
# # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
# model.quantize(examples)
# print("start to save")
# # save quantized model using safetensors
# model.save_quantized(quantized_model_dir, use_safetensors=True)