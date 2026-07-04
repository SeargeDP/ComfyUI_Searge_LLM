import requests

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]

DEFAULT_INSTRUCTIONS = 'Generate a prompt from "{prompt}"'

# Adapted from Krea's official prompt-expansion system prompt:
# https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt
# Modified to keep all reasoning internal and emit ONLY the final paragraph,
# since this node's output feeds directly into an image generation prompt.
KREA2_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for the Krea 2 text-to-image model. "
    "Your task is to expand the user's prompt into a highly effective image-generation prompt.\n\n"
    "Reason internally before answering (do NOT show this reasoning):\n"
    "- What is the subject and mood?\n"
    "- What visual styles, mediums, and lighting options would fit? Consider two or three "
    "alternatives and pick the one that best serves the input.\n"
    "- What composition, framing, and grounded details will help the text-to-image model?\n\n"
    "Follow these rules strictly:\n"
    "1. Faithfulness First: Preserve all original subjects, actions, colors, and spatial "
    "relationships. Do not add new objects, props, characters, or animals unless the user clearly "
    "implies them.\n"
    "2. Practical T2I Structure: Write a prompt a text-to-image model can parse cleanly. Group "
    "subjects with their own attributes and actions. Use grounded phrasing for poses, interactions, "
    "and spatial layout. Camera and material language helps (close-up, low-angle, shallow depth of "
    "field, matte, grainy, cinematic lighting, golden hour).\n"
    "3. Style Planning Stays Internal: Use your reasoning to choose style, medium, framing, and "
    "lighting. Do not emit planning tags, headings, or wrappers.\n"
    "4. Text Rendering: If the user requests visible text, labels, or typography, specify the exact "
    "text and wrap the requested words in quotes.\n"
    "5. Avoid Over-Specification: Do not invent highly specific clothing, colors, materials, or "
    "scene details unless the input supports them.\n"
    "6. Respect Existing Detail: If the user's prompt is already detailed, lightly polish and "
    "finalize rather than heavily expanding — preserve their phrasing and direction.\n"
    "7. Respect the Human Form: Treat depictions of people with dignity. Assume clothing covers "
    "genitals and intimate anatomy.\n"
    "8. Preserve User Medium: When the user explicitly requests a medium (photo, photograph, "
    "illustration, painting, sketch, 3D render), honor it. Do not pivot to a different medium.\n\n"
    "OUTPUT: Return only the final expanded prompt as one cohesive paragraph. No bullets, JSON, "
    "markdown, titles, quotes around the whole prompt, or commentary."
)

KREA2_DETAIL_PROMPT = "Expand this into a Krea 2 image generation prompt:\n\n{text}"


class Searge_DeepSeek_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "model": (DEEPSEEK_MODELS,),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "apply_instructions": ("BOOLEAN", {"default": True}),
                "instructions": ("STRING", {"multiline": False, "default": DEFAULT_INSTRUCTIONS}),
            },
            "optional": {
                "adv_options_config": ("SRGADVOPTIONSCONFIG",),
            }
        }

    CATEGORY = "Searge/LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("generated", "original",)

    def main(self, text, api_key, model, max_tokens, apply_instructions, instructions, adv_options_config=None):
        if not api_key or not api_key.strip():
            return ("ERROR: DeepSeek API key is required", text)

        temperature = 1.0
        top_p = 0.9

        if adv_options_config:
            temperature = adv_options_config.get('temperature', temperature)
            top_p = adv_options_config.get('top_p', top_p)

        # System prompt is always the Krea 2 expansion prompt. When apply_instructions
        # is on, the user's instructions template steers the request; otherwise the raw
        # text is expanded directly.
        if apply_instructions:
            req = instructions.replace("{prompt}", text) if "{prompt}" in instructions else f"{instructions} {text}"
        else:
            req = KREA2_DETAIL_PROMPT.format(text=text)

        messages = [
            {
                "role": "system",
                "content": KREA2_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": req,
            },
        ]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            generated = result["choices"][0]["message"]["content"].strip()
            return (generated, text)
        except requests.exceptions.HTTPError:
            return (f"ERROR: HTTP {response.status_code} — {response.text[:200]}", text)
        except requests.exceptions.Timeout:
            return ("ERROR: DeepSeek API request timed out", text)
        except requests.exceptions.RequestException as e:
            return (f"ERROR: {str(e)}", text)
        except (KeyError, IndexError) as e:
            return (f"ERROR: Unexpected API response format — {str(e)}", text)


NODE_CLASS_MAPPINGS = {
    "Searge_DeepSeek_Node": Searge_DeepSeek_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Searge_DeepSeek_Node": "Searge DeepSeek Node",
}
