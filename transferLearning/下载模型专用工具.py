import torch
import os
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM

# 保存模型到本地，和下载模型都是一个道理

if __name__ == "__main__":

    # bert-base-uncased
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained("bert-base-uncased")

    # tokenizer.save_pretrained("./models/bert-base-uncased")
    # model.save_pretrained("./models/bert-base-uncased")

    # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # model = BertModel.from_pretrained("bert-base-cased")

    # tokenizer.save_pretrained("./models/bert-base-cased")
    # model.save_pretrained("./models/bert-base-cased")

    # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True)
    # model = AutoModel.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True) # .half().cuda()
    # tokenizer.save_pretrained("./models/THUDM/chatglm2-6b-int4")
    # model.save_pretrained("./models/THUDM/chatglm2-6b-int4")



    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:96"
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = '1'

    # torch.cuda.empty_cache()
    # tokenizer = AutoTokenizer.from_pretrained("THUDM/codegeex2-6b-int4", trust_remote_code=True)
    # # dtype = torch.bfloat16
    # model = AutoModel.from_pretrained("THUDM/codegeex2-6b-int4", trust_remote_code=True)

    # tokenizer.save_pretrained("./models/THUDM/codegeex2-6b-int4")
    # model.save_pretrained("./models/THUDM/codegeex2-6b-int4")

    name = 'mosaicml/mpt-7b'

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    config = AutoConfig.from_pretrained(name, trust_remote_code=True)
    config.attn_config['attn_impl'] = 'triton'
    config.init_device = 'cuda:0' # For fast initialization directly on GPU!

    model = AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        torch_dtype=torch.bfloat16, # Load model weights in bfloat16
        trust_remote_code=True
    )
    tokenizer.save_pretrained("./models/EleutherAI/gpt-neox-20b")
    model.save_pretrained("./models/mosaicml/mpt-7b")
