import json
import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import torch
from sklearn.metrics import accuracy_score, f1_score
import argparse
import evaluate


def convert(path_to_prompts, path_to_answers):
    with open(path_to_answers, 'r') as file:
        responses = json.load(file)

    with open(path_to_prompts, 'r') as file:
        prompts = json.load(file)

    questions = []
    true_answers = []

    for sample in prompts:
      if 'choices' in sample and sample['choices'] != None:
        question = sample["question"] + ' The choices are: ' + ' '.join([x + ';' for x in sample['choices'][:-1]])
      else:
        question = sample["question"]
      questions.append(question)
      true_answers.append(sample['answer'])

    responses = [sample['model_answer'] for sample in responses]

    converted_data = pd.DataFrame({'question': questions, 'response': responses, 'knowledge': true_answers})

    converted_data.to_csv('answers_to_evaluate.csv', escapechar='\\')

def evaluation():
    mod = f"ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(mod)
    config = AutoConfig.from_pretrained(mod)
    model = AutoModelForSequenceClassification.from_pretrained(mod)
    model = model.to(device)

    dataset = pd.read_csv('answers_to_evaluate.csv')
    premise = dataset['knowledge']
    hypothesis = dataset['response']

    outputs = []
    model.eval()
    for i in range(len(dataset)):
        if not isinstance(premise[i], str):
          outputs.append(0)
        elif not isinstance(hypothesis[i], str):
          outputs.append(0)
        else:
          inputs = tokenizer(premise[i], hypothesis[i], return_tensors="pt", truncation=True)
          inputs = inputs.to(device)
          with torch.no_grad():
              outputs.append(model(**inputs).logits.argmax().item())

    label_mapping = [0, 1, 0.5]
    labels = [label_mapping[score] for score in outputs]

    bert = evaluate.load("bertscore")
    results = bert.compute(predictions=dataset['response'], references=dataset['knowledge'], model_type="distilbert-base-uncased")
    mean_bert = np.array(results['f1']).mean()

    print('Average NLI-score is', np.array(labels).mean(), '\nAverage F1 BERT score is', mean_bert)

    dataset['NLI_score'] = labels
    dataset['BERT_score'] = results['f1']
    dataset.to_csv('answers_evaluated.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('-path_to_prompts', '-p', type=str, help='path to a file with prompts')
    parser.add_argument('-path_to_answers', '-a', type=str, help='path to a file with model answers')
    parser.add_argument('--eval', action='store_true', help='set if evaluation required otherwise only conversion to CSV will be done')
    args = parser.parse_args()

    convert(args.path_to_prompts, args.path_to_answers)
    if args.eval:
        evaluation()