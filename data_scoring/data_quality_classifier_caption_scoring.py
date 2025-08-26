from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import LlavaPhi3Classifier, LlavaGemmaClassifier, LlavaLlamaClassifier, LlavaQwenClassifier


import requests
import argparse
from PIL import Image
from io import BytesIO
from transformers import AutoTokenizer, BitsAndBytesConfig
import pandas as pd

from tqdm import tqdm
from typing import List, Tuple
import logging
import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
import webdataset as wds
from dataclasses import dataclass, field


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("evaluation test")


@dataclass
class DataCollatorForImagePreprocessing(object):
    def __init__(self, tokenizer, image_processor, max_len): 
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        self.max_len = max_len

    def format_text(self, text: str):
        text = "<image>\n" + text.replace("<image>", "image")
        return text

    # @staticmethod
    def pad_sequence(self, sequence, padding_value=0):
        """Pad a sequence to the desired max length."""
        if self.tokenizer.padding_side == "left":
            sequence = [torch.flip(_input_ids, [0]) for _input_ids in sequence]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequence,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)[:, :self.max_len]
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return input_ids, attention_mask

    def __call__(self,
                 batch: Tuple[List, List, List]) -> Tuple[torch.Tensor, torch.Tensor, list]:
        images, txts, infos = batch
        prompt = [self.format_text(text) for text in txts]
        # inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        # batch_input_ids, batch_attention_mask = inputs["input_ids"][:, :self.max_len], inputs["attention_mask"][:, :self.max_len].type(torch.bool)
        batch_input_ids = [tokenizer_image_token(sequence, self.tokenizer, return_tensors='pt') for sequence in prompt]
        if batch_input_ids[0][0] != self.tokenizer.bos_token_id:
            batch_input_ids = [torch.cat([torch.LongTensor([self.tokenizer.bos_token_id]), sequence]) for sequence in batch_input_ids]
        if batch_input_ids[0][-1] != self.tokenizer.eos_token_id:
            batch_input_ids = [torch.cat([sequence, torch.LongTensor([self.tokenizer.eos_token_id])]) for sequence in batch_input_ids]
        
        batch_input_ids, batch_attention_mask = self.pad_sequence(batch_input_ids)
        
        batch_image_tensor = self.image_processor(images, return_tensors="pt")["pixel_values"]

        return (batch_image_tensor, batch_input_ids, batch_attention_mask, infos)


def filter_small_images(sample):
    # Check if image exists and has valid dimensions
    if "image" in sample and sample["image"] is not None:
        # Get image dimensions
        width, height = sample["image"].size
        # Filter out images with width or height less than 1
        return width >= 1 and height >= 1
    return False

def main(args, gpu_id=0):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"device_map": device}
    kwargs['torch_dtype'] = torch.float16
    # kwargs['attn_implementation'] = 'flash_attention_2'

    logger.info(f"Model class {model_name} already loaded")

    if args.load_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif args.load_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        quantization_config = None

    if 'phi' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = LlavaPhi3Classifier.from_pretrained(args.model_path, quantization_config=quantization_config, attn_implementation="flash_attention_2", **kwargs)
    elif 'gemma' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = LlavaGemmaClassifier.from_pretrained(args.model_path, quantization_config=quantization_config, **kwargs)
    elif 'qwen' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = LlavaQwenClassifier.from_pretrained(args.model_path, quantization_config=quantization_config, attn_implementation="flash_attention_2", **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        model = LlavaLlamaClassifier.from_pretrained(args.model_path, quantization_config=quantization_config, **kwargs)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
        vision_tower.to(dtype=torch.float16, device=model.device)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 4096
    
    # set padding side to `left` for batch text generation
    model.config.tokenizer_padding_side = tokenizer.padding_side = "right"

    logger.info(f"Model loading finished for CUDA {torch.cuda.current_device()}")
        
    collator = DataCollatorForImagePreprocessing(tokenizer, image_processor, 75)
    
    for tar_id in list(range(100000))[gpu_id * args.tars_per_gpu: (gpu_id + 1) * args.tars_per_gpu]:
        logger.info(f"Start processing tar {tar_id}")
        
        if not os.path.exists(os.path.join(args.tar_file_path, f"{str(tar_id).zfill(8)}.tar")):
            continue
        if os.path.exists(os.path.join(args.tar_file_path, f"{str(tar_id).zfill(8)}_quality_score.parquet")):
            logger.info(f"Tar {tar_id} already processed")
            continue
        shard_path = args.tar_file_path + "/{:08d}.tar".format(tar_id)
        pipeline = [
            wds.SimpleShardList(shard_path),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt", json="json"),
            wds.to_tuple("image", "text", "json"),
            wds.batched(args.batch_size, partial=True),
        ]
        dataset = wds.DataPipeline(*pipeline)
        
        dataloader = wds.WebLoader(
            dataset,
            collate_fn=collator,
            batch_size=None,
            shuffle=False,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
        )
        
        final_data = []
        for batch_image_tensor, batch_input_ids, batch_attention_mask, info in tqdm(dataloader):
            with torch.inference_mode():
                
                logits = model(input_ids=batch_input_ids.to(model.device),
                                attention_mask=batch_attention_mask.to(model.device),
                                images=batch_image_tensor.half().to(model.device)).logits
            
            if model.num_labels >= 2:
                scores = torch.softmax(logits, dim=-1).cpu().numpy()
                quality_class = torch.argmax(logits, dim=-1).cpu().numpy()
                
                for i in range(batch_input_ids.shape[0]):
                    info[i]["quality_score"] = scores[i][1] # We only use the normalized logits on the positive class
                    info[i]["quality_class"] = quality_class[i]
            elif model.num_labels == 1:
                scores = logits.view(-1).cpu().numpy()
                
                for i in range(batch_input_ids.shape[0]):
                    info[i]["quality_score"] = scores[i]
            else:
                pass

            final_data += info
        
        df = pd.DataFrame(final_data)
        df.to_parquet(f"{args.tar_file_path}/{str(tar_id).zfill(8)}_quality_score.parquet")
        logger.info(f"Tar {tar_id} finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="weizhiwang/UniFilter-Qwen2.5-1.5B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--tar-file-path", type=str, default="datacomp/medium_vanilla_filter/")
    parser.add_argument("--num-gpus", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tars-per-gpu", type=int, default=128)
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-len", type=int, default=2048)
    args = parser.parse_args()
    logger.info(args)
    main(args, gpu_id=args.gpu_id)