import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np
import re

from libra.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from libra.conversation import conv_templates, SeparatorStyle
from libra.model.builder import load_pretrained_model
from libra.utils import disable_torch_init
from libra.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

import math
import pydicom
from PIL import Image
from io import BytesIO
from pydicom.pixel_data_handlers.util import apply_voi_lut

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_images(image_file):
    """
    Load an image from a local file, a URL, or a DICOM file.

    Args:
        image_file (str): The path or URL of the image file to load.

    Returns:
        PIL.Image.Image: The loaded image in RGB format.

    Raises:
        ValueError: If the DICOM file does not contain image data.
        TypeError: If the input is neither a valid file path nor a URL.
    """
    if isinstance(image_file, str):
        # Case 1: Load from URL
        if image_file.startswith(('http://', 'https://')):
            try:
                response = requests.get(image_file)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
            except Exception as e:
                raise ValueError(f"Error loading image from URL: {image_file}\n{e}")

        # Case 2: Load from DICOM file
        elif image_file.lower().endswith('.dcm'):
            try:
                dicom = pydicom.dcmread(image_file)
                if 'PixelData' in dicom:
                    data = apply_voi_lut(dicom.pixel_array, dicom)

                    # Handle MONOCHROME1 images
                    if dicom.PhotometricInterpretation == "MONOCHROME1":
                        data = np.max(data) - data

                    # Normalize the image data
                    data = data - np.min(data)
                    data = data / np.max(data)
                    data = (data * 255).astype(np.uint8)

                    # Convert to 3-channel RGB if necessary
                    if data.ndim == 2:
                        data = np.stack([data] * 3, axis=-1)

                    image = Image.fromarray(data).convert('RGB')
                else:
                    raise ValueError("DICOM file does not contain image data")
            except Exception as e:
                raise ValueError(f"Error loading DICOM file: {image_file}\n{e}")

        # Case 3: Load standard image files (e.g., PNG, JPG)
        else:
            try:
                image = Image.open(image_file).convert('RGB')
            except Exception as e:
                raise ValueError(f"Error loading standard image file: {image_file}\n{e}")

    else:
        raise TypeError("image_file must be a string representing a file path or URL")

    return image

def get_image_tensors(image_file, image_folder, image_processor, model, device='cuda'):
    # Load and preprocess the images
    if isinstance(image_file, str):
        image = []
        image_path = os.path.join(image_folder, image_file)
        img = load_images(image_path)
        image.append(img)
    elif isinstance(image_file, (list, tuple)):
        image = []
        image_paths = [os.path.join(image_folder, file_name) for file_name in image_file]
        for path in image_paths:
            img = load_images(path)
            image.append(img)
    else:
        raise TypeError("image_file must be a string or a str/list/tuple of strings")

    # Ensure two images are present
    if len(image) != 2:
        image.append(image[0])  
        if model.config.mm_projector_type == "TAC":
            print("Contains only current image. Adding a dummy prior image for TAC.")

    # Process each image
    processed_images = []
    for img_data in image:
        image_temp = process_images([img_data], image_processor, model.config)[0]
        image_temp = image_temp.to(device=device, non_blocking=True)
        processed_images.append(image_temp)

    # Separate current and prior images
    cur_images = [processed_images[0]]
    prior_images = [processed_images[1]]

    # Stack and return as batched tensor
    batch_images = torch.stack([torch.stack(cur_images), torch.stack(prior_images)])

    return batch_images

def eval_model(args):
    """
    Evaluate a pre-trained model on a set of questions and images.
    Args:
        args (Namespace): A namespace object containing the following attributes:
            - model_path (str): Path to the pre-trained model.
            - model_base (str): Base model name.
            - question_file (str): Path to the JSON file containing questions.
            - num_chunks (int): Number of chunks to split the questions into.
            - chunk_idx (int): Index of the chunk to process.
            - answers_file (str): Path to the file where answers will be saved.
            - image_folder (str): Folder containing the images.
            - conv_mode (str): Conversation mode to use.
            - temperature (float): Sampling temperature for generation.
            - top_p (float): Top-p sampling parameter.
            - num_beams (int): Number of beams for beam search.
            - max_new_tokens (int): Maximum number of new tokens to generate.
            - length_penalty (float): Length penalty for beam search.
            - num_return_sequences (int): Number of sequences to return.
    Raises:
        TypeError: If `image_file` is neither a string nor a list/tuple of strings.
    Returns:
        None
    """
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        pad_token_id = tokenizer.pad_token_id
        
        image_tensors = get_image_tensors(image_file, args.image_folder, image_processor, model)
        
        stop_str = conv.sep if conv.sep_style not in {SeparatorStyle.TWO, SeparatorStyle.LLAMA_3, SeparatorStyle.MISTRAL} else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            torch.cuda.empty_cache() 
            if args.num_beams > 1:
                output_ids = model.generate(
                    input_ids=input_ids,
                    images=image_tensors,
                    do_sample=False,
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    no_repeat_ngram_size=3,
                    max_new_tokens=args.max_new_tokens,
                    stopping_criteria=[stopping_criteria],
                    use_cache=True,
                    length_penalty=args.length_penalty, 
                    # output_scores=True,  
                    num_return_sequences = args.num_return_sequences,
                    attention_mask=attention_mask, 
                    pad_token_id=pad_token_id)
            else:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    no_repeat_ngram_size=3,
                    max_new_tokens=args.max_new_tokens,
                    stopping_criteria=[stopping_criteria],
                    use_cache=True,
                    length_penalty=args.length_penalty,
                    # output_scores=True, 
                    num_return_sequences = args.num_return_sequences,
                    attention_mask=attention_mask, 
                    pad_token_id=pad_token_id)
            
        torch.cuda.empty_cache() 
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="libra")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="libra_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)