import torch
import os
from transformers import AutoTokenizer, pipeline, AutoConfig, AutoModelForCausalLM



if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    name = './models/mosaicml/mpt-7b'

    tokenizer = AutoTokenizer.from_pretrained('./models/EleutherAI/gpt-neox-20b')

    config = AutoConfig.from_pretrained(name, trust_remote_code=True)
    config.attn_config['attn_impl'] = 'triton'
    config.init_device = 'cuda:0' # For fast initialization directly on GPU!

    model = AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        torch_dtype=torch.bfloat16, # Load model weights in bfloat16
        trust_remote_code=True
    )

    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')

    with torch.autocast('cuda', dtype=torch.bfloat16):
        print(
            pipe('Here is a recipe for vegan banana bread:\n',
                max_new_tokens=100,
                do_sample=True,
                use_cache=True))

