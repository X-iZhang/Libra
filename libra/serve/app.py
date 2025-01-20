#    Copyright 2025 Xi Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
            
import os
import gradio as gr
import argparse
from libra.eval import libra_eval
from libra.eval.run_libra import load_model

# =========================================
# Global/Default Configuration
# =========================================
DEFAULT_MODEL_PATH = "X-iZhang/libra-v1.0-7b"

def get_model_short_name(model_path: str) -> str:
    """
    Extract the part after the last '/' in the model path to use as the name displayed in the dropdown menu.
    For example: "X-iZhang/libra-v1.0-7b" -> "libra-v1.0-7b"
    """
    return model_path.rstrip("/").split("/")[-1]


loaded_models = {}  # {model_key: reuse_model_object}

def generate_radiology_description(
    selected_model_name: str,
    current_img_data,
    prior_img_data,
    use_no_prior: bool,
    prompt: str,
    temperature: float,
    top_p: float,
    num_beams: int,
    max_new_tokens: int,
    model_paths_dict: dict
) -> str:
    """
    Perform radiology report inference:
    1) Find the actual model_path based on the selected model name from the dropdown.
    2) Ensure the user has selected both Current & Prior images.
    3) Call libra_eval for inference.
    """
    real_model_path = model_paths_dict[selected_model_name]

    if not current_img_data:
        return "Error: Please select or upload the Current Image."

    if use_no_prior:
        prior_img_data = current_img_data
    else:
        if not prior_img_data:
            return "Error: Please select or upload the Prior Image, or check 'Without Prior Image'."

    if selected_model_name in loaded_models:
        reuse_model = loaded_models[selected_model_name]
    else:
        reuse_model = load_model(real_model_path)
        loaded_models[selected_model_name] = reuse_model
        
    try:
        output = libra_eval(
            libra_model=reuse_model,
            image_file=[current_img_data, prior_img_data],
            query=prompt,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            length_penalty=1.0,
            num_return_sequences=1,
            conv_mode="libra_v1",
            max_new_tokens=max_new_tokens
        )
        return output
    except Exception as e:
        return f"An error occurred during model inference: {str(e)}"

def main():

    cur_dir = os.path.abspath(os.path.dirname(__file__))

    example_curent_path = os.path.join(cur_dir, "..", "..", "assets", "example_curent.jpg")
    example_curent_path = os.path.abspath(example_curent_path)

    example_prior_path = os.path.join(cur_dir, "..", "..", "assets", "example_prior.jpg")
    example_prior_path = os.path.abspath(example_prior_path)

    IMAGE_EXAMPLES = [
        [example_curent_path],
        [example_prior_path]
    ]

    parser = argparse.ArgumentParser(description="Demo for Radiology Image Description Generator (Local Examples)")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
         help="User-specified model path. If not provided, only default model is shown."
    )
    args = parser.parse_args()
    cmd_model_path = args.model_path
    
    model_paths_dict = {}
    user_key = get_model_short_name(cmd_model_path)
    model_paths_dict[user_key] = cmd_model_path

    if cmd_model_path != DEFAULT_MODEL_PATH:
        default_key = get_model_short_name(DEFAULT_MODEL_PATH)
        model_paths_dict[default_key] = DEFAULT_MODEL_PATH

    
    # ========== Gradio ==========
    with gr.Blocks(title="Libra: Radiology Analysis with Direct URL Examples") as demo:
        gr.Markdown("""
        ## 🩻 Libra: Leveraging Temporal Images for Biomedical Radiology Analysis
        [Project Page](https://x-izhang.github.io/Libra_v1.0/) | [Paper](https://arxiv.org/abs/2411.19378) | [Code](https://github.com/X-iZhang/Libra) | [Model](https://huggingface.co/X-iZhang/libra-v1.0-7b)

        **Requires a GPU to run effectively!**
        """)

        model_dropdown = gr.Dropdown(
            label="Select Model",
            choices=list(model_paths_dict.keys()),
            value=user_key,
            interactive=True
        )
        
        prompt_input = gr.Textbox(
            label="Clinical Prompt",
            value="Provide a detailed description of the findings in the radiology image.",
            lines=2,
            info=(
                "If clinical instructions are available, include them after the default prompt. "
                "For example: “Provide a detailed description of the findings in the radiology image. "
                "Following clinical context: Indication: chest pain, History: ...”"
            )
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Current Image")
                current_img = gr.Image(
                    label="Drop Or Upload Current Image",
                    type="filepath",
                    interactive=True
                )
                
                gr.Examples(
                    examples=IMAGE_EXAMPLES,
                    inputs=current_img,
                    label="Example Current Images"
                )

            with gr.Column():
                gr.Markdown("### Prior Image")
                prior_img = gr.Image(
                    label="Drop Or Upload Prior Image",
                    type="filepath",
                    interactive=True
                )
                
                with gr.Row():
                    gr.Examples(
                        examples=IMAGE_EXAMPLES,
                        inputs=prior_img,
                        label="Example Prior Images"
                    )
                    without_prior_checkbox = gr.Checkbox(
                        label="Without Prior Image",
                        value=False,  
                        info="If checked, the current image will be used as the dummy prior image in the Libra model."
                    )
                          
        
        with gr.Accordion("Parameters Settings", open=False):
            temperature_slider = gr.Slider(
                label="Temperature", 
                minimum=0.1, maximum=1.0, step=0.1, value=0.9
            )
            top_p_slider = gr.Slider(
                label="Top P", 
                minimum=0.1, maximum=1.0, step=0.1, value=0.8
            )
            num_beams_slider = gr.Slider(
                label="Number of Beams", 
                minimum=1, maximum=20, step=1, value=1
            )
            max_tokens_slider = gr.Slider(
                label="Max output tokens",
                minimum=10, maximum=4096, step=10, value=128
            )

        output_text = gr.Textbox(
            label="Generated Findings Section",
            lines=5
        )

        generate_button = gr.Button("Generate Findings Description")
        generate_button.click(
            fn=lambda model_name, c_img, p_img, no_prior, prompt, temp, top_p, beams, tokens: generate_radiology_description(
                model_name,
                c_img,
                p_img,
                no_prior,
                prompt,
                temp,
                top_p,
                beams,
                tokens,
                model_paths_dict
            ),
            inputs=[
                model_dropdown,    # model_name
                current_img,       # c_img
                prior_img,         # p_img
                without_prior_checkbox,  # no_prior
                prompt_input,      # prompt
                temperature_slider,# temp
                top_p_slider,      # top_p
                num_beams_slider,  # beams
                max_tokens_slider  # tokens
            ],
            outputs=output_text
        )
        
        gr.Markdown("""
        ### Terms of Use

        The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA.
        
        By accessing or using this demo, you acknowledge and agree to the following:

        - **Research & Non-Commercial Purposes**: This demo is provided solely for research and demonstration. It must not be used for commercial activities or profit-driven endeavors.
        - **Not Medical Advice**: All generated content is experimental and must not replace professional medical judgment.
        - **Content Moderationt**: While we apply basic safety checks, the system may still produce inaccurate or offensive outputs.
        - **Responsible Use**: Do not use this demo for any illegal, harmful, hateful, violent, or sexual purposes.

        By continuing to use this service, you confirm your acceptance of these terms. If you do not agree, please discontinue use immediately.
        """)
    demo.launch(share=True)

if __name__ == "__main__":
    main()
