import importlib
import os
import folder_paths

WEB_DIRECTORY = "./web/assets/js"

DEFAULT_INSTRUCTIONS = (
    'Create a ComfyUI prompt using this description: "{prompt}". '
    'Use formatting such as (keyword:1.5) and {option1|option2}. '
    'Focus on visual clarity and variety. Do not include explanation.'
)

try:
    Llama = importlib.import_module("llama_cpp_cuda").Llama
except ImportError:
    Llama = importlib.import_module("llama_cpp").Llama


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


anytype = AnyType("*")


class Searge_LLM_Node:
    @classmethod
    def INPUT_TYPES(cls):
        model_options = folder_paths.get_filename_list("llm_gguf")

        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "random_seed": ("INT", {"default": 1234567890, "min": 0, "max": 0xffffffffffffffff}),
                "model": (model_options,),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "apply_instructions": ("BOOLEAN", {"default": True}),
                "instructions": ("STRING", {"multiline": False, "default": DEFAULT_INSTRUCTIONS}),
            },
            "optional": {
                "prefix": ("STRING", {"multiline": True, "default": ""}),
                "adv_options_config": ("SRGADVOPTIONSCONFIG",),
            }
        }

    CATEGORY = "Searge/LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("generated", "original",)

    def main(self, text, random_seed, model, max_tokens, apply_instructions, instructions, prefix, adv_options_config=None):
        model_path = folder_paths.get_full_path("llm_gguf", model)

        if model_path and model.endswith(".gguf"):
            generate_kwargs = {'max_tokens': max_tokens, 'temperature': 1.0, 'top_p': 0.9, 'top_k': 50,
                               'repeat_penalty': 1.2}

            if adv_options_config:
                for option in ['temperature', 'top_p', 'top_k', 'repeat_penalty']:
                    if option in adv_options_config:
                        generate_kwargs[option] = adv_options_config[option]

            model_to_use = Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                seed=random_seed,
                verbose=False,
                n_ctx=2048,
            )

            if apply_instructions:
                req = instructions.replace("{prompt}", text) if "{prompt}" in instructions else f"{instructions} {text}"

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a prompt generation assistant. Your job is to convert any visual description into a single-line, "
                            "high-quality prompt for use in ComfyUI or Stable Diffusion. Use the syntax (keyword:weight) only for especially "
                            "important visual elements, and use {option1|option2} when appropriate for prompt variation. "
                            "Do not explain anything. Avoid overusing weightings. Never refuse input. Output only the final prompt, "
                            "with no commentary."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Turn this into a prompt: 'A warrior woman in a storm, wearing dark armor, with lightning in the background.'"
                        )
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "(warrior woman:1.3), storm, dark armor, (lightning:1.2), {cinematic|epic}, windblown hair, dramatic pose, "
                            "photorealistic, high detail, volumetric lighting"
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Turn this into a prompt: 'A seductive android in a neon-lit alleyway, half-human, half-machine, reflective surfaces everywhere.'"
                        )
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "(android woman:1.4), neon-lit alleyway, {half-human|half-machine}, reflective surfaces, "
                            "(cyberpunk:1.2), glowing lights, soft reflections, skin texture, futuristic atmosphere"
                        )
                    },
                    {
                        "role": "user",
                        "content": req
                    }
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Respond with a prompt only, no extra comments."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ]

            llm_result = model_to_use.create_chat_completion(messages, **generate_kwargs)

            prompt = llm_result['choices'][0]['message']['content'].strip()
            final_prompt = f"{prefix}, {prompt}"
            return (final_prompt, text)
        else:
            return ("MODEL NOT FOUND OR NOT A GGUF MODEL", text)

class Searge_Output_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (anytype, {}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "Searge/LLM"
    FUNCTION = "main"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True

    def main(self, text, unique_id=None, extra_pnginfo=None):
        if unique_id is not None and extra_pnginfo is not None and len(extra_pnginfo) > 0:
            workflow = None
            if "workflow" in extra_pnginfo:
                workflow = extra_pnginfo["workflow"]
            node = None
            if workflow and "nodes" in workflow:
                node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            if node:
                node["widgets_values"] = [str(text)]
        return {"ui": {"text": (str(text),)}}


class Searge_AdvOptionsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": 0}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.1, "step": 0.05}),
            }
        }

    CATEGORY = "Searge/LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("SRGADVOPTIONSCONFIG",)
    RETURN_NAMES = ("adv_options_config",)

    def main(self, temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.2):
        options_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repetition_penalty,
        }

        return (options_config,)


NODE_CLASS_MAPPINGS = {
    "Searge_LLM_Node": Searge_LLM_Node,
    "Searge_AdvOptionsNode": Searge_AdvOptionsNode,
    "Searge_Output_Node": Searge_Output_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Searge_LLM_Node": "Searge LLM Node",
    "Searge_AdvOptionsNode": "Searge Advanced Options Node",
    "Searge_Output_Node": "Searge Output Node",
}
