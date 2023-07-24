import torch
from transformers import BertConfig, BertModel, BertTokenizer



if __name__ == '__main__':

    model_name = 'hfl/chinese-roberta-wwm-ext'
    config = BertConfig.from_pretrained(model_name)	# 这个方法会自动从官方的s3数据库下载模型配置、参数等信息（代码中已配置好位置）
    tokenizer = BertTokenizer.from_pretrained(model_name)		 # 这个方法会自动从官方的s3数据库读取文件下的vocab.txt文件
    model = BertModel.from_pretrained(model_name)		# 这个方法会自动从官方的s3数据库下载模型信息


    s_a, s_b = "李白拿了个锤子", "锤子？"
    # 分词是tokenizer.tokenize, 分词并转化为id是tokenier.encode
    # 简单调用一下, 不作任何处理经过transformer
    input_id = tokenizer.encode(s_a)
    input_id = torch.tensor([input_id])  # 输入数据是tensor且batch形式的
    sequence_output, pooled_output = model(input_id) # 输出形状分别是[1, 9, 768], [1, 768]
    # 但是输入BertModel的还需要指示前后句子的信息的token type, 以及遮掉PAD部分的attention mask
    inputs = tokenizer.encode_plus(s_a, text_pair=s_b, return_tensors="pt")  # 还有些常用的可选参数max_length, pad_to_max_length等
    print(inputs.keys())  # 返回的是一个包含id, mask信息的字典
    # dict_keys(['input_ids', 'token_type_ids', 'attention_mask']
    sequence_output, pooled_output = model(**inputs)
