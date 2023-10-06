import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)
from trl import RewardTrainer
from utils import charge_dataset, charge_qlora, formatting_func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prepared_dataset = charge_dataset()

qlora = True
if qlora:
    model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    # model_name = "sileod/deberta-v3-large-tasksource-rlhf-reward-model"
    model, tokenizer, dataset, peft_config = charge_qlora(prepared_dataset, model_name)

else:
    model_name = "gpt2"
    # model_name = "distilroberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        dataset = prepared_dataset.map(formatting_func)

    dataset = prepared_dataset.map(formatting_func)


formatted_dataset = prepared_dataset.map(formatting_func)

learning_rate = 8e-5

training_args = TrainingArguments(
    output_dir=f"./reward_model/{model_name}",
    per_device_train_batch_size=2,
    evaluation_strategy="steps",
    learning_rate=learning_rate,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=200,
    save_total_limit=1,
    report_to=None,
)

if qlora:
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],
        peft_config=peft_config,
        max_length=1024,
    )
else:
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],
        max_length=1024,
    )


trainer.train()

model.save_pretrained(f"./reward_model/{model_name}_{learning_rate}")
