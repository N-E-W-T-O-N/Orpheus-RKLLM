
#https: // huggingface.co/onnx-community/snac_24khz-ONNX/tree/main

#Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys

tokenizer = AutoTokenizer.from_pretrained("Prince-1/orpheus_3b_0.1_ft")

def InputTokenizer(input_str:str, voice ="jess"):

    prompt =[input_str]
    chosen_voice  = voice
    prompts = [(f"{chosen_voice}: " + p) if chosen_voice else "" for p in prompt]

    all_input_ids = []

    for prompt in prompts:
      input_ids = tokenizer(prompt, return_tensors="pt").input_ids
      all_input_ids.append(input_ids)

    #print(all_input_ids)

    start_token = torch.tensor([[ 128259]], dtype=torch.int64) # Start of human
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64) # End of the text, End of human

    all_modified_input_ids = []
    for input_ids in all_input_ids:
      modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1) # SOH SOT Text EOT EOH
      all_modified_input_ids.append(modified_input_ids)
    #print(all_modified_input_ids)
    all_padded_tensors = []
    all_attention_masks = []
    max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
    #print("max_length", max_length)
     

    for modified_input_ids in all_modified_input_ids:
      padding = max_length - modified_input_ids.shape[1]
      padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
      attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
      all_padded_tensors.append(padded_tensor)
      all_attention_masks.append(attention_mask)

    all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
    all_attention_masks = torch.cat(all_attention_masks, dim=0)

    input_ids = all_padded_tensors.to("cpu")
    attention_mask = all_attention_masks.to("cpu")

    print(",".join(map(str,input_ids[0].tolist())))

if __name__ == "__main__":
    input:str = sys.argv[1]
    #input =                                                                       \
    "Hey there my name is EDISON, <giggles> and I'm a speech generation model that can sound like a person.I Am a badass person"
    InputTokenizer(input)
