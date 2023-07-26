from transformers import pipeline, BertTokenizer, BertModel

model_name = './models/bert'

unmasker = pipeline('fill-mask', model=model_name)
print(unmasker("Hello I'm a [MASK] model."))


tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)
