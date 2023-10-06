# This code has been taken from Alpaca repository: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
# All credit for it goes to the original authors of this code
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import json
from torch.utils.data import Dataset
from transformers import Trainer
from utils import print_trainable_parameters, smart_tokenizer_and_embedding_resize, make_supervised_data_module


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def train():
    print('Loading model...')
    # model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    print('Model loaded!')
    print_trainable_parameters(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained("google/flan-t5-small", model_max_length=512,padding_side="right",use_fast=False)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    train_data_path = "data_train.json"
    eval_data_path = "data_test.json"
    train_dataset, eval_dataset, data_collator = make_supervised_data_module(tokenizer=tokenizer, train_data_path=train_data_path, eval_data_path=eval_data_path)

    training_args = transformers.TrainingArguments(
                        output_dir="./results",
                        evaluation_strategy="epoch",
                        save_strategy="epoch",
                        report_to=None,
                        save_total_limit=1,
                        learning_rate=2e-5,
                        num_train_epochs=8
                    )
    
    trainer = Trainer(model=model, 
                      args=training_args, 
                      tokenizer=tokenizer, 
                      train_dataset=train_dataset, 
                      eval_dataset=eval_dataset, 
                      data_collator=data_collator)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir="models/fine-tuned")


if __name__ == "__main__":
    train()
