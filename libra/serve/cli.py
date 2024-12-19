import argparse
import torch

from libra.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from libra.conversation import conv_templates, SeparatorStyle
from libra.model.builder import load_pretrained_model
from libra.utils import disable_torch_init
from libra.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import requests
import pydicom
from PIL import Image
from io import BytesIO
from pydicom.pixel_data_handlers.util import apply_voi_lut
from transformers import TextStreamer


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

def main(args):
    """
    Main function to load a pretrained model, process images, and interact with the user through a conversation loop.
    Args:
        args (Namespace): A namespace object containing the following attributes:
            model_path (str): Path to the pretrained model.
            model_base (str): Base model name.
            load_8bit (bool): Flag to load the model in 8-bit precision.
            load_4bit (bool): Flag to load the model in 4-bit precision.
            device (str): Device to load the model on (e.g., 'cuda', 'cpu').
            conv_mode (str, optional): Conversation mode to use. If None, it will be inferred from the model name.
            image_file (list): List of paths to image files to be processed.
            temperature (float): Sampling temperature for text generation.
            max_new_tokens (int): Maximum number of new tokens to generate.
            debug (bool): Flag to enable debug mode for additional output.
    Raises:
        EOFError: If an EOFError is encountered during user input, the loop will exit.
    """
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'libra' in model_name.lower():
        conv_mode = "libra_v1"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    roles = conv.roles
    
    image=[]
    for path in args.image_file:
        img = load_images(path)
        image.append(img)

    # set dummy prior image
    if len(image) == 1:
        print("Contains only current image. Adding a dummy prior image.")
        image.append(image[0])

    processed_images = []
    for img_data in image:
        image_temp = process_images([img_data], image_processor, model.config)[0]
        image_temp = image_temp.to(device='cuda',non_blocking=True) 
        processed_images.append(image_temp)
        
    cur_images = [processed_images[0]]
    prior_images = [processed_images[1]]
    image_tensor = torch.stack([torch.stack(cur_images), torch.stack(prior_images)])
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            image = None

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        pad_token_id = tokenizer.pad_token_id
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                attention_mask=attention_mask, 
                pad_token_id=pad_token_id,
                stopping_criteria=[stopping_criteria])
        
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:],skip_special_tokens=True).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, nargs="+", required=True, help="List of image files to process.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default="libra_v1")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    main(args)