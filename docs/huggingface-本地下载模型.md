从本地加载模型
以上代码会由 transformers 自动下载模型实现和参数。完整的模型实现在 Hugging Face Hub。如果你的网络环境较差，下载模型参数可能会花费较长时间甚至失败。此时可以先将模型下载到本地，然后从本地加载。

从 Hugging Face Hub 下载模型需要先安装Git LFS，然后运行

git clone https://huggingface.co/THUDM/chatglm2-6b
如果你从 Hugging Face Hub 上下载 checkpoint 的速度较慢，可以只下载模型实现

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm2-6b
然后从这里手动下载模型参数文件，并将下载的文件替换到本地的 chatglm2-6b 目录下。

将模型下载到本地之后，将以上代码中的 THUDM/chatglm2-6b 替换为你本地的 chatglm2-6b 文件夹的路径，即可从本地加载模型。

模型的实现仍然处在变动中。如果希望固定使用的模型实现以保证兼容性，可以在 from_pretrained 的调用中增加 revision="v1.0" 参数。v1.0 是当前最新的版本号，完整的版本列表参见 Change Log。

See Detail : https://github.com/THUDM/ChatGLM2-6B#%E4%BB%8E%E6%9C%AC%E5%9C%B0%E5%8A%A0%E8%BD%BD%E6%A8%A1%E5%9E%8B
http://www.wqrd.cn/news/61921.html