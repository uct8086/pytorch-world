import sys
from transformers import AutoTokenizer, AutoModel

if __name__ == '__main__':

    model_name = './models/THUDM/codegeex2-6b-int4'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    model = model.eval()

    # print(response)
    # language: python
    # write a bubble sort function

    print('请输入指令：\n\n')
    prompt = '# language: python\n '
    line_text = ''

    while True:
        line_text = input()
        if line_text == '':
            # remember adding a language tag for better performance
            # prompt = "# language: python\n# write a bubble sort function\n"
            print('当前Prompt： \n', prompt)
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(inputs, max_length=256, top_k=1)
            response = tokenizer.decode(outputs[0])
            print(response)
            break
        prompt += '# ' + line_text + '\n'


