import torch
from transformers import BertConfig, BertModel, BertTokenizer



if __name__ == '__main__':

    model_name = './models/bert'
    config = BertConfig.from_pretrained(model_name)	# 这个方法会自动从官方的s3数据库下载模型配置、参数等信息（代码中已配置好位置）
    tokenizer = BertTokenizer.from_pretrained(model_name)		 # 这个方法会自动从官方的s3数据库读取文件下的vocab.txt文件
    model = BertModel.from_pretrained(model_name)		# 这个方法会自动从官方的s3数据库下载模型信息

    # 保存模型
    # tokenizer.save_pretrained("./bert")
    # model.save_pretrained("./bert")


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

    # encode仅返回input_ids
    tokenizer.encode("我爱你")
    # Out : [101, 2769, 4263, 872, 102]

    # encode_plus返回所有编码信息
    input_id = tokenizer.encode_plus("我爱你", "你也爱我")
    # Out : 
    #     {'input_ids': [101, 2769, 4263, 872, 102, 872, 738, 4263, 2769, 102],
    #     'token_type_ids': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    #     'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    # 添加batch维度并转化为tensor
    input_ids = torch.tensor(input_id['input_ids'])
    token_type_ids = torch.tensor(input_id['token_type_ids'])
    attention_mask_ids=torch.tensor(input_id['attention_mask'])

    # 将模型转化为eval模式
    model.eval()
    # 将模型和数据转移到cuda, 若无cuda,可更换为cpu
    device = 'cuda'
    tokens_tensor = input_ids.to(device).unsqueeze(0)
    segments_tensors = token_type_ids.to(device).unsqueeze(0)
    attention_mask_ids_tensors = attention_mask_ids.to(device).unsqueeze(0)
    model.to(device)

    # 进行编码
    with torch.no_grad():
        # See the models docstrings for the detail of the inputs
        outputs = model(tokens_tensor, segments_tensors, attention_mask_ids_tensors)
        # Transformers models always output tuples.
        # See the models docstrings for the detail of all the outputs
        # In our case, the first element is the hidden state of the last layer of the Bert model
        encoded_layers = outputs
        print(outputs)
    # 得到最终的编码结果encoded_layers


