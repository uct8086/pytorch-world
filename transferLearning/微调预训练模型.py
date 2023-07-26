import torch
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, TrainingArguments, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# 调用计算指标来计算预测的准确性。在将预测传递给计算之前，您需要将预测转换为 logits
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == '__main__':

    model_name = './models/bert-base-cased'
    # dataset = load_dataset("yelp_review_full", cache_dir='./data/yelp_review_full')
    dataset = load_dataset('./data/yelp_review_full')
    print(dataset["train"][100])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # 然后创建数据集的较小子集（如前所示）以加快微调速度
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    # 为您的训练和测试数据集创建一个DataLoader，以便您可以迭代批量数据
    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)  

    # 使用预期标签数量加载模型
    dtype = torch.bfloat16
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5, torch_dtype=dtype)

    # 创建优化器和学习率调度器来微调模型。让我们使用AdamWPyTorch 的优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # 从Trainer创建默认学习率调度程序
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # 指定device使用 GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # 循环训练
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # 评估
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print('最终准确度是： ', metric.compute())
    # 最终准确度是：  {'accuracy': 0.592}

    # 保存模型
    tokenizer.save_pretrained("./models/local-train-model")
    model.save_pretrained("./models/local-train-model")

    # 清空模型
    del model
    torch.cuda.empty_cache()


    # 在pytorch 中训练

    # model_name = './models/bert-base-cased'
    # dataset = load_dataset("yelp_review_full", cache_dir='./data/yelp_review_full')
    # print(dataset["train"][100])

    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # # 训练
    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)


    # training_args = TrainingArguments(output_dir="test_trainer")

    # # 评估
    # # Trainer 在训练期间不会自动评估模型性能。您需要向 Trainer 传递一个函数来计算和报告指标, Evaluate 库提供了一个简单的精度函数，您可以使用valuate.load（有关更多信息，请参阅此快速教程）函数加载
    # metric = evaluate.load("accuracy")

    # training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    # # trainer

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=small_train_dataset,
    #     eval_dataset=small_eval_dataset,
    #     compute_metrics=compute_metrics,
    # )

    # trainer.train()