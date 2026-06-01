"""
code source: https://github.com/geronimi73/3090_shorts/blob/main/captioning/captioning.ipynb
"""
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
DTYPE = torch.bfloat16


def get_VLM(model="SmolVLM2"):
    if model == "SmolVLM2":
        # Load VLM: SmolVLM2
        repo = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        processor = AutoProcessor.from_pretrained(repo)
        model = AutoModelForImageTextToText.from_pretrained(
            repo, 
            dtype=DTYPE,
            _attn_implementation="sdpa", # the author used _attn_implementation="flash_attention_2", but it didn't work
        ).to(DEVICE)

    return processor, model


def caption(processor, model, img, prompt):
    """
    single image input 
    """
    # Construct a conversation
    conversation = [ 
        dict(
            role="user", 
            content=[
                dict(type="image", image=img),
                dict(type="text", text=prompt),
            ]) 
    ]

    # Tokenize input
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=DTYPE)

    output_ids = model.generate(**inputs, max_new_tokens=128)   # Query model
    output_ids = output_ids[:, inputs["input_ids"].size(1):]    # Remove input tokens from response
    outputs = processor.batch_decode(output_ids, skip_special_tokens=True)  # Decode tokens to text

    return outputs[0].strip()


def batch_caption(processor, model, images, prompts):
    """
    batch image input 
    """
    # Construct a list of conversations, each with one image and prompt
    conversations = [ 
        [dict(
            role="user",
            content=[
                dict(type="image", image=img),
                dict(type="text", text=prompt)
            ]
        )]
        for img, prompt in zip(images, prompts)
    ]

    # Preprocess inputs
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        padding=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding_side="left", # without this config, there will empty string on the left
    ).to(model.device, dtype=DTYPE)
    
    # Query model and remove input tokens
    output_ids = model.generate(**inputs, max_new_tokens=128)
    output_ids=[ tok_out[len(tok_in):] for tok_in, tok_out in zip(inputs["input_ids"], output_ids) ]     

    generated_texts = processor.batch_decode(
        output_ids, 
        skip_special_tokens=True
    )
    
    return [t.strip() for t in generated_texts]
