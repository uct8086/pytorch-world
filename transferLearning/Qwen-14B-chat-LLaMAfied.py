
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


if __name__ == '__main__':

    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

    # tokenizer = AutoTokenizer.from_pretrained("./hiyouga/Qwen-14B-Chat-LLaMAfied", trust_remote_code=True)
    # model = AutoModel.from_pretrained("./hiyouga/Qwen-14B-Chat-LLaMAfied", trust_remote_code=True).half().cuda()

    # model = model.eval()
    # response, history = model.chat(tokenizer, "你好", history=[])
    # print(response)

    # prompt = ''
    # line_text = ''

    # while True:
    #     line_text = input()
    #     if line_text == 'begin':
    #         print('开始预测：\n')
    #         response, history = model.chat(tokenizer, prompt, history=history)
    #         print(response)
    #         break
    #     prompt += line_text




    tokenizer = AutoTokenizer.from_pretrained("./hiyouga/Qwen-14B-Chat-LLaMAfied")
    model = AutoModelForCausalLM.from_pretrained("./hiyouga/Qwen-14B-Chat-LLaMAfied", torch_dtype="auto", device_map="auto")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = [
        {"role": "user", "content": "Who are you?"}
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    generate_ids = model.generate(inputs, streamer=streamer)

