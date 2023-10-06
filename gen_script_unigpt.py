import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

torch.manual_seed(42)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = 'generative-model/'
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512,padding_side="right",use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


def generate(sentence):
  input_ids = tokenizer(sentence, return_tensors="pt").input_ids
  input_ids = input_ids.to(device)
  outputs = model.generate(input_ids, num_beams=5, max_new_tokens=512, no_repeat_ngram_size=3)
  return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    data = json.load(open("prompts.json"))
    answers = []
    for line in tqdm(data):
        if 'choices' in line and line['choices'] != None:
            question = line["question"] + ' The choices are: ' + ' '.join([x + ';' for x in line['choices'][:-1]])
        else:
            question = line["question"]
        answer = generate(question)
        answers.append({
            "guid": line["guid"],
            "model_answer": answer
        })
    
    with open(f"answers_unigpt.json", "w") as f:
        json.dump(answers, f, indent=4)

if __name__ == '__main__':
    main()