import json

import pandas as pd
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def charge_dataset(data_train="train_reward", data_test="test_reward"):
    """
    Dataset charging

    Args:
        data_train (str): name of the train dataset
        data_test (str): name of the test dataset

    Returns:
        prepared_dataset (DatasetDict): dataset charged
    """

    with open(
        f"../data/{data_train}.json", encoding="cp437", errors="ignore"
    ) as json_file:
        train = json.load(json_file)
    with open(
        f"../data/{data_test}.json", encoding="cp437", errors="ignore"
    ) as json_file:
        test = json.load(json_file)

    train = pd.DataFrame(train)
    train = train[(train["length_0"] <= 1024) & (train["length_1"] <= 1024)]
    list_of_dicts_train = train.to_dict(orient="records")

    test = pd.DataFrame(test)
    test = test[(test["length_0"] <= 1024) & (test["length_1"] <= 1024)]
    list_of_dicts_eval = test.to_dict(orient="records")

    prepared_dataset = DatasetDict(
        {
            "train": Dataset.from_list(list_of_dicts_train),
            "test": Dataset.from_list(list_of_dicts_eval),
        }
    )
    return prepared_dataset


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def charge_qlora(dataset, model_name):
    """
    Charge the model and tokenize the dataset

    Args:
        dataset (DatasetDict): dataset to tokenize
        model_name (str): name of the model to charge

    Returns:
        model (PreTrainedModel): model charged
        tokenizer (PreTrainedTokenizer): tokenizer charged
        dataset (DatasetDict): dataset tokenized
    """

    print(f"Loading model {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    print("Loading lora adapter...")
    model = get_peft_model(model, peft_config)
    print("Model loaded!")
    print_trainable_parameters(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    def formatting_func(examples):
        kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": 1024,
            "return_tensors": "pt",
        }

        # Prepend the prompt and a line break to the original_response and response-1 fields.
        prompt_plus_chosen_response = examples["final_chat_1"]
        prompt_plus_rejected_response = examples["final_chat_0"]

        # Then tokenize these modified fields.
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0],
            "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0],
            "attention_mask_rejected": tokens_rejected["attention_mask"][0],
        }

    dataset = dataset.map(formatting_func)

    return model, tokenizer, dataset


def charge_qlora_model(path_model):
    """
    Charge the model

    Args:
        path_model (str): path of the model to charge

    Returns:
        model (PreTrainedModel): model charged
    """

    base_model = AutoModelForSequenceClassification.from_pretrained(
        f"./{path_model}/base_model"
    )

    lora_model = PeftModel.from_pretrained(base_model, f"./{path_model}/lora_model")
    num_parameters = sum([p.numel() for p in lora_model.parameters()])
    print(f"Number of parameters: {num_parameters:,}")
