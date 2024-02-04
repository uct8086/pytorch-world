# Dive into deep learning

基于《动手学深度学习》的教材，实现命令行可运行的代码项目

文件按时间先后顺序命名，越大，越代表后面的内容

剔除了Jupyter，无需安装d2l, 可以直接在VScode命令行跑

# 2023-07-13 更新

已完结，系统地了解了深度学习这个领域

# 下载模型

HF_ENDPOINT=https://hf-mirror.com python ./transferLearning/下载模型专用工具.py 


### hiyouga/Qwen-14B-Chat-LLaMAfied


 huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen-14B-Chat  --local-dir models/Qwen-14B-Chat