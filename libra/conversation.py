import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    MISTRAL = auto()
    GEMMA = auto()
    PHI3 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
                    
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
                    
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
                    
        elif self.sep_style == SeparatorStyle.LLAMA_2 or self.sep_style == SeparatorStyle.MISTRAL:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: 
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
                    
            ret = ret.lstrip(self.sep)
            
        elif self.sep_style == SeparatorStyle.LLAMA_3:
            wrap_sys = lambda msg: f"<|start_header_id|>system<|end_header_id|>\n\n{msg}{self.sep2}" if len(msg) > 0 else ""
            wrap_role = lambda role, msg: f"<|start_header_id|>{role}<|end_header_id|>\n\n{msg}{self.sep2}\n"

            ret = "" # "<|begin_of_text|>" 

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                    
                if message:  
                    if isinstance(message, tuple):
                        message, _, _ = message  

                    
                    if i == 0:
                        ret += wrap_sys(self.system)  
                        ret += wrap_role("user", message)  
                    else:
                        role_name = "user" if role == self.roles[0] else "assistant"
                        ret += wrap_role(role_name, message)  
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    
            ret = ret.strip()
            
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
                    
        elif self.sep_style == SeparatorStyle.GEMMA:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
                    
        elif self.sep_style == SeparatorStyle.PHI3:
            ret = self.system + self.sep
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += self.roles[i % 2] + message + self.sep
                else:
                    ret += self.roles[i % 2]
                    
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])
        
    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
        if image_process_mode == "Pad":
            def expand2square(pil_img, background_color=(0, 0, 0)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((518, 518))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
        if max(image.size) > max_len:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str
        
    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    image = self.process_image(image, image_process_mode, return_pil=return_pil)
                    images.append(image)
        return images
    

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_gemma = Conversation(
    system="""""",
    roles=("<start_of_turn>user\n", "<start_of_turn>model\n"),
    version="gemma",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.GEMMA,
    sep="<end_of_turn>\n",
)

conv_phi3 = Conversation(
    system="""<|system|>\nYou are a helpful AI assistant.""",
    roles=("\n<|user|>\n", "\n<|assistant|>\n"),
    version="phi3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PHI3,
    sep="<|end|>",
)

conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MISTRAL,
    sep="",
    sep2="</s>",
)

conv_mistral_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llama_3 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_3,
    sep="<|begin_of_text|>",
    sep2="<|eot_id|>",
)

conv_libra_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_libra_llama_3 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language."
           "The assistant specialized in comparing Chest X-ray images, identifying differences, and noting temporal changes.",
    roles=("USER", "ASSISTANT"),
    version="llama_v3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_3,
    sep=" ",
    sep2="<|eot_id|>",
)

conv_libra_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_libra_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_libra_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_libra_v1 = Conversation(
    system="The assistant specialized in comparing Chest X-ray images, identifying differences, and noting temporal changes.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


conv_libra_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

llava_med_conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MISTRAL,
    sep="",
    sep2="</s>",
)

llava_med_conv_mistral_instruct_v0 = Conversation(
    system="You are LLaVA-Med, a large language and vision assistant trained by a group of researchers at Microsoft, based on the general domain LLaVA architecture.\n"
            "You are able to understand the visual content that the user provides, and assist the user with a variety of medical and clinical research tasks using natural language.\n"
            "Follow the instructions carefully and explain your answers in detail.\n",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(
        ("USER", "Hi!"),
        ("ASSISTANT", "Hi there!  How can I help you today?\n")
    ),
    offset=0,
    sep_style=SeparatorStyle.MISTRAL,
    sep="",
    sep2="</s>",
)

llava_med_conv_mistral_instruct_v1 = Conversation(
    system="You are LLaVA-Med, a large language and vision assistant trained by a group of researchers at Microsoft, based on the general domain LLaVA architecture.\n"
           "You are able to understand the visual content that the user provides, and assist the user with a variety of medical and clinical research tasks using natural language.\n"
           "Follow the instructions carefully and explain your answers in detail when necessary. However, if a question is simple and can be answered with a single word (e.g., yes or no), "
           "please provide only a succinct response without additional explanation.\n",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MISTRAL,
    sep="",
    sep2="</s>",
)

default_conversation = conv_vicuna_v1
conv_templates = {
    "default": conv_libra_v1,
    
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    
    "plain": conv_libra_plain,
    "libra_v0": conv_libra_v0,
    "libra_v1": conv_libra_v1,
    
    "libra_v0_mmtag": conv_libra_v0_mmtag,
    "libra_v1_mmtag": conv_libra_v1_mmtag,
    
    "llama_2": conv_llama_2,
    "libra_llama_2": conv_libra_llama_2,
    
    "llama_3": conv_llama_3,
    "libra_llama_3": conv_libra_llama_3,
    
    "mpt": conv_mpt,
    "conv_gemma": conv_gemma,
    "conv_phi3": conv_phi3,

    "mistral_instruct": conv_mistral_instruct,
    "mistral_direct": conv_mistral_direct,

    # another moder's conversation templates
    "llava_med_v1.5_mistral_7b": llava_med_conv_mistral_instruct,
    "llava_med_v1.5_mistral_7b_v0": llava_med_conv_mistral_instruct_v0,
    "llava_med_v1.5_mistral_7b_v1": llava_med_conv_mistral_instruct_v1,
    
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())