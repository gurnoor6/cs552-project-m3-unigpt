import random
import torch
import transformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer, Trainer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from utils import print_trainable_parameters, smart_tokenizer_and_embedding_resize, make_supervised_data_module


random.seed(12)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def charge_llama():
    model_name = 'decapoda-research/llama-7b-hf'

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("Loading LLama model...")
    model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config, 
        trust_remote_code=True
    )
    model.config.use_cache = False

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    print("Chargin Lora adapaters...")
    model = get_peft_model(model, peft_config)

    print('\nModel loaded!')

    print_trainable_parameters(model)
    
    return model, model_name

    
def train():

    

    model, model_name  = charge_llama()


    tokenizer = LlamaTokenizer.from_pretrained(model_name, model_max_length=512,padding_side="right",use_fast=False)


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

    # data_path = "/home/gurnoor/project-m3-unigpt/fine-tuning/data/test.json"
    train_data_path = "data_short_train.json"
    test_data_path = "data_short_test.json"
    train_dataset, eval_dataset, data_collator = make_supervised_data_module(tokenizer=tokenizer, train_data_path=train_data_path, eval_data_path=test_data_path)

    training_args = transformers.TrainingArguments(
                        output_dir="./results",
                        per_device_train_batch_size=4,
                        gradient_accumulation_steps=4,
                        save_strategy="epoch",
                        report_to=None,
                        logging_steps=20,
                        save_total_limit=1,
                        learning_rate=2e-5,
                        num_train_epochs=8,
                        evaluation_strategy="steps",
                        optim = "paged_adamw_8bit",
                        fp16=True
                    )
  # optim = "paged_adamw_8bit"
  # fp16=True
    trainer = Trainer(model=model, 
                      args=training_args, 
                      tokenizer=tokenizer, 
                      train_dataset = train_dataset, 
                      eval_dataset = eval_dataset, 
                      data_collator = data_collator)
    for name, module in trainer.model.named_modules():
      if "norm" in name:
          module = module.to(torch.float16)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir="models/fine-tuned")

if __name__ == '__main__':
    train()