import argparse
import torch
from libra.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from libra.conversation import conv_templates, SeparatorStyle
from libra.model.builder import load_pretrained_model
from libra.utils import disable_torch_init
from libra.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

import requests
import pydicom
from PIL import Image
from io import BytesIO
from pydicom.pixel_data_handlers.util import apply_voi_lut
import datetime

def load_model(model_path, model_base=None):
    """
    Load the model and return its components.

    Args:
        model_path (str): Path to the model.
        model_base (str): Base model, if any.

    Returns:
        tuple: (tokenizer, model, image_processor, context_len)
    """
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    return tokenizer, model, image_processor, context_len

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

def get_image_tensors(image_path, image_processor, model, device='cuda'):
    # Load and preprocess the images
    if isinstance(image_path, str):
        image = []
        img = load_images(image_path)
        image.append(img)
    elif isinstance(image_path, (list, tuple)):
        image = []
        for path in image_path:
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

def libra_eval(
    model_path=None,
    model_base=None,
    image_file=None,
    query=None,
    conv_mode=None,
    temperature=0.2,
    top_p=None,
    num_beams=1,
    num_return_sequences=None,
    length_penalty=1.0,
    max_new_tokens=128,
    libra_model=None
):
    # Model
    disable_torch_init()

    if libra_model is not None:
        tokenizer, model, image_processor, context_len = libra_model
        model_name = model.config._name_or_path
    else:
        tokenizer, model, image_processor, context_len = load_model(model_path, model_base)
        model_name = get_model_name_from_path(model_path)

    qs = query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'libra-v1.0-3b' in model_name.lower():
        mode_conv = "libra_llama_3"
    elif 'cxrgen' in model_name.lower():
        mode_conv = "libra_v0"
    elif 'mistral' in model_name.lower():
        mode_conv = "mistral_instruct"
    else:
        mode_conv = "libra_v1"

    if conv_mode is not None and mode_conv != conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(mode_conv, conv_mode, conv_mode))
    else:
        conv_mode = mode_conv

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    pad_token_id = tokenizer.pad_token_id

    image_tensor = get_image_tensors(image_file, image_processor, model)

    stop_str = conv.sep if conv.sep_style not in {SeparatorStyle.TWO, SeparatorStyle.LLAMA_3, SeparatorStyle.MISTRAL} else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        torch.cuda.empty_cache()
        if num_beams > 1:
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor,
                do_sample=False,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                max_new_tokens=max_new_tokens,
                stopping_criteria=[stopping_criteria],
                use_cache=True,
                length_penalty=length_penalty, 
                output_scores=True,  
                attention_mask=attention_mask, 
                pad_token_id=pad_token_id,
                num_return_sequences = num_return_sequences)
        else:
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample= True,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                max_new_tokens=max_new_tokens,
                attention_mask=attention_mask, 
                pad_token_id=pad_token_id,
                stopping_criteria=[stopping_criteria],
                use_cache=True)
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()

    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="X-iZhang/libra-v1.0-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="libra_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=None)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    libra_eval(**vars(args))