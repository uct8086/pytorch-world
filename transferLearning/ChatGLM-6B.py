"""
ChatGLM2-6B 是开源中英双语对话模型 ChatGLM-6B 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM2-6B 引入了如下新特性：

更强大的性能：基于 ChatGLM 初代模型的开发经验，我们全面升级了 ChatGLM2-6B 的基座模型。ChatGLM2-6B 使用了 GLM 的混合目标函数，经过了 1.4T 中英标识符的预训练与人类偏好对齐训练，评测结果显示，相比于初代模型，ChatGLM2-6B 在 MMLU（+23%）、CEval（+33%）、GSM8K（+571%） 、BBH（+60%）等数据集上的性能取得了大幅度的提升，在同尺寸开源模型中具有较强的竞争力。
更长的上下文：基于 FlashAttention 技术，我们将基座模型的上下文长度（Context Length）由 ChatGLM-6B 的 2K 扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练，允许更多轮次的对话。但当前版本的 ChatGLM2-6B 对单轮超长文档的理解能力有限，我们会在后续迭代升级中着重进行优化。
更高效的推理：基于 Multi-Query Attention 技术，ChatGLM2-6B 有更高效的推理速度和更低的显存占用：在官方的模型实现下，推理速度相比初代提升了 42%，INT4 量化下，6G 显存支持的对话长度由 1K 提升到了 8K。
更开放的协议：ChatGLM2-6B 权重对学术研究完全开放，在填写问卷进行登记后亦允许免费商业使用。

"""
import sys
import os
import time
from transformers import AutoTokenizer, AutoModel


if __name__ == '__main__':

    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    tokenizer = AutoTokenizer.from_pretrained("./models/THUDM/chatglm2-6b-int4", trust_remote_code=True)
    model = AutoModel.from_pretrained("./models/THUDM/chatglm2-6b-int4", trust_remote_code=True).half().cuda()

    model = model.eval()
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)

    prompt = ''
    line_text = ''

    while True:
        line_text = input()
        if line_text == 'begin':
            start_time = time.time()
            response, history = model.chat(tokenizer, prompt, history=history)
            end_time = time.time()
            print(f"\n预测耗时 {end_time - start_time:.2f} 秒.\n")
            print(response)
            print("\n你可以继续向我提问\n")
            prompt = ''
        elif line_text == 'end':
            break
        else:
            prompt += line_text

