## Evaluation pipeline

There are two evaluation techniques in our project: NLI-based metric and BERT-score.

1. To evaluate model performance using natural language inference, we use pre-trained [RoBERTa-Large NLI model](https://huggingface.co/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli) using ground truth answers as *premise* and model responses as *hypothesis*. Predictions **entailment**, **neutral**, and **contradiction** were converted into scores 1, 0.5, and 0, respectively. Then, we computed an average score. To compute BERT-score, we used DistilBERT-base model. To get these two scores, you need to run a script `run_evaluation.py` providing paths to prompts.json and model's responses and setting `--eval` as in the example below:

```python
python run_evaluation.py -p prompts.json -a answers_unigpt.json --eval
```
This will output average NLI-score and F1 BERT-score and create two .csv files: `answers_to_evaluate.csv` containing responses and ground truth answers to compare and `answers_evalauted.csv` with scores for each sample.

