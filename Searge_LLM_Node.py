import importlib
import os

import folder_paths

GLOBAL_MODELS_DIR = os.path.join(folder_paths.models_dir, "llm_gguf")

WEB_DIRECTORY = "./web/assets/js"

DEFAULT_INSTRUCTIONS = 'Generate a prompt from "{prompt}"'

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
        model_options = []
        if os.path.isdir(GLOBAL_MODELS_DIR):
            gguf_files = [file for file in os.listdir(GLOBAL_MODELS_DIR) if file.endswith(".gguf")]
            model_options.extend(gguf_files)

        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "random_seed": ("INT", {"default": 1234567890, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "model": (model_options,),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "apply_instructions": ("BOOLEAN", {"default": True}),
                "instructions": ("STRING", {"multiline": False, "default": DEFAULT_INSTRUCTIONS}),
            },
            "optional": {
                "adv_options_config": ("SRGADVOPTIONSCONFIG",),
            },
        }

    CATEGORY = "Searge/LLM"
    FUNCTION = "main"
    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "generated",
        "original",
    )

    def main(self, text, random_seed, model, max_tokens, apply_instructions, instructions, adv_options_config=None):
        model_path = os.path.join(GLOBAL_MODELS_DIR, model)

        if model.endswith(".gguf"):
            generate_kwargs = {
                "max_tokens": max_tokens,
                "temperature": 1.0,
                "top_p": 0.9,
                "top_k": 50,
                "repeat_penalty": 1.2,
            }

            n_ctx = 2048
            if adv_options_config:
                for option in ["temperature", "top_p", "top_k", "repeat_penalty"]:
                    if option in adv_options_config:
                        generate_kwargs[option] = adv_options_config[option]
                if "n_ctx" in adv_options_config:
                    n_ctx = adv_options_config["n_ctx"]

            # n_gpu_layers=0: force CPU to avoid VRAM conflict when NVFP4 diffusion
            # models are loaded alongside llama-cpp. llama-cpp and PyTorch compete
            # for VRAM and the resulting cast triggers a segfault on sm_120.
            model_to_use = Llama(
                model_path=model_path,
                n_gpu_layers=0,
                seed=random_seed,
                verbose=False,
                n_ctx=n_ctx,
            )

            if apply_instructions:
                req = instructions.replace("{prompt}", text) if "{prompt}" in instructions else f"{instructions} {text}"
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "An image generation prompt is a single paragraph summary to describe the subject and "
                        "style of an image. It includes a description of the kind of image, the subject of "
                        "the image, and some description of the image medium and style in the form of short "
                        'keyword.\n\nCreate an image generation prompt for the subject "a creepy creature '
                        'shadow in the dark in a dimly lit tunnel" in the style "digital art illustration '
                        'with intricate details".',
                    },
                    {
                        "role": "assistant",
                        "content": "Image Description: A digitally crafted illustration portrays a chilling scene within "
                        "a dimly lit, cavernous tunnel. The dominant subject of the image is a mysterious "
                        "creature, its form subtly discernible only as a menacing shadow on the walls of the "
                        "tunnel. Its elongated silhouette suggests a serpentine creature with sharp "
                        "protrusions and an ominous aura. The creature's eyes, a pair of glowing orbs, appear "
                        "eerily human-like yet alien. The tunnel is riddled with intricate details that "
                        "enhance the eerie atmosphere: dust particles floating lazily in the feeble light, "
                        "ancient and crumbling stone, water droplets sparkling on the damp walls, and a "
                        "hauntingly beautiful, bioluminescent fungi growing in the shadows. The dimly lit "
                        "environment is highlighted by strategically placed light sources that create "
                        "dramatic chiaroscuro effects, casting an unsettling and atmospheric glow on the "
                        "scene. Digital Art Illustration with Intricate Details (Dark, Atmospheric, "
                        "Suspenseful)",
                    },
                    {
                        "role": "user",
                        "content": "Now compile the description and style into a single paragraph summary",
                    },
                    {
                        "role": "assistant",
                        "content": "Digital art illustration featuring a dark, atmospheric, and suspenseful scene within "
                        "a dimly lit, cavernous tunnel. The subject of the image is a mysterious creature, "
                        "depicted only as a menacing shadow on the walls, with elongated silhouette and sharp "
                        "protrusions. The creature's eyes, a pair of glowing orbs, appear eerily human-like "
                        "yet alien. The tunnel is adorned with intricate details, such as dust particles, "
                        "ancient crumbling stone, water droplets, and a hauntingly beautiful bioluminescent "
                        "fungi growing in the shadows. Dramatic chiaroscuro effects are created through "
                        "strategically placed light sources, casting an unsettling and atmospheric glow on "
                        "the scene.",
                    },
                    {"role": "user", "content": "create a detailed summary without the title or style"},
                    {
                        "role": "assistant",
                        "content": "A dimly lit, cavernous tunnel is the setting for this chilling digital illustration. "
                        "A mysterious creature lurks in the shadows, its elongated silhouette suggestive of a "
                        "serpentine creature with sharp protrusions and an ominous aura. The creature's eyes, "
                        "a pair of glowing orbs, appear eerily human-like yet alien. The tunnel is riddled "
                        "with intricate details that enhance the eerie atmosphere: dust particles floating "
                        "lazily in the feeble light, ancient and crumbling stone, water droplets sparkling on "
                        "the damp walls, and a hauntingly beautiful, bioluminescent fungi growing in the "
                        "shadows. Dramatic chiaroscuro effects are created through strategically placed light "
                        "sources, casting an unsettling and atmospheric glow on the scene.",
                    },
                    {
                        "role": "user",
                        "content": 'Generate a prompt from "magical garden, sitting on a blue_bench, Disney Princess in '
                        "pink_dress, blonde_hair, portrait, Pixar, Disney style, photorealistic, high "
                        'details, detailed skin, pores, freckles"',
                    },
                    {
                        "role": "assistant",
                        "content": "In a captivating, Pixar-inspired, Disney-style, photorealistic portrait, a magical "
                        "garden unfolds as a breathtaking backdrop. The subject of the image is a Disney "
                        "Princess, dressed elegantly in a pink gown with flowing fabric, sitting serenely on "
                        "a blue bench. The princess boasts long, blonde hair and a radiant smile. The garden "
                        "is meticulously detailed, with vibrant, enchanted flora and fauna, as well as "
                        "whimsical details like sparkling fairy lights and a picturesque waterfall. The "
                        "princess is highlighted against the lush, detailed foliage, with a focus on the "
                        "high-definition details of her porcelain skin, visible freckles, and the intricacies "
                        "of her elegant gown. The image is rendered in the captivating, photorealistic style "
                        "that exemplifies both the Disney and Pixar brands, capturing the princess's timeless "
                        "beauty and the magic of her enchanting surroundings.",
                    },
                    {"role": "user", "content": req},
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Try your best to give the best response possible to "
                        "the user.",
                    },
                    {
                        "role": "user",
                        "content": f"Create a detailed visually descriptive caption of this description, which will be "
                        f"used as a prompt for a text to image AI system (caption only, no instructions like "
                        f'"create an image").Remove any mention of digital artwork or artwork style. Give '
                        f"detailed visual descriptions of the character(s), including ethnicity, skin tone, "
                        f"expression etc. Imagine using keywords for a still for someone who has aphantasia. "
                        f"Describe the image style, e.g. any photographic or art styles / techniques utilized. "
                        f"Make sure to fully describe all aspects of the cinematography, with abundant "
                        f"technical details and visual descriptions. If there is more than one image, combine "
                        f"the elements and characters from all of the images creatively into a single "
                        f"cohesive composition with a single background, inventing an interaction between the "
                        f"characters. Be creative in combining the characters into a single cohesive scene. "
                        f"Focus on two primary characters (or one) and describe an interesting interaction "
                        f"between them, such as a hug, a kiss, a fight, giving an object, an emotional "
                        f"reaction / interaction. If there is more than one background in the images, pick the "
                        f"most appropriate one. Your output is only the caption itself, no comments or extra "
                        f"formatting. The caption is in a single long paragraph. If you feel the images are "
                        f"inappropriate, invent a new scene / characters inspired by these. Additionally, "
                        f"incorporate a specific movie director's visual style and describe the lighting setup "
                        f"in detail, including the type, color, and placement of light sources to create the "
                        f"desired mood and atmosphere. Always frame the scene, including details about the "
                        f"film grain, color grading, and any artifacts or characteristics specific. "
                        f"Compress the output to be concise while retaining key visual details. MAX OUTPUT "
                        f"SIZE no more than 250 characters."
                        f"\nDescription : {text}",
                    },
                ]

            llm_result = model_to_use.create_chat_completion(messages, **generate_kwargs)
            result_text = llm_result["choices"][0]["message"]["content"].strip()

            # Free GPU memory before returning so PyTorch models can load cleanly.
            del model_to_use
            try:
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

            return (result_text, text)
        else:
            return ("NOT A GGUF MODEL", text)


class Searge_Output_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (anytype, {}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
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
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "step": 0.05,
                        "tooltip": "Controls randomness. Lower values (0.1–0.5) produce focused, deterministic output. Higher values (1.0+) increase creativity and variation.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.1,
                        "step": 0.05,
                        "tooltip": "Nucleus sampling threshold. The model only considers tokens whose cumulative probability reaches this value. Lower values (0.5–0.7) make output more conservative; 0.9 is a safe default.",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "tooltip": "Limits the token candidates to the top K most likely at each step. Lower values (10–20) produce more predictable output. Set to 0 to disable.",
                    },
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.2,
                        "min": 0.1,
                        "step": 0.05,
                        "tooltip": "Penalises tokens that have already appeared. Values above 1.0 reduce repetition. Increase to 1.3–1.5 if the model loops or repeats phrases.",
                    },
                ),
                "n_ctx": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 2048,
                        "max": 131072,
                        "step": 2048,
                        "tooltip": "Context window size in tokens. Increase if you get 'Requested tokens exceed context window' errors.",
                    },
                ),
            }
        }

    CATEGORY = "Searge/LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("SRGADVOPTIONSCONFIG",)
    RETURN_NAMES = ("adv_options_config",)

    def main(self, temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.2, n_ctx=2048):
        options_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repetition_penalty,
            "n_ctx": n_ctx,
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
