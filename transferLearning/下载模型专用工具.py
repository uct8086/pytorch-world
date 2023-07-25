from transformers import BertTokenizer, BertModel

# 保存模型到本地，和下载模型都是一个道理

# bert-base-uncased
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")

# tokenizer.save_pretrained("./models/bert-base-uncased")
# model.save_pretrained("./models/bert-base-uncased")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased")

tokenizer.save_pretrained("./models/bert-base-cased")
model.save_pretrained("./models/bert-base-cased")