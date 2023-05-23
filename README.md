

## BERT Fine-tuning

Lora

```
python bert_sentiment_analysis.py --output-dir bert-sentiment-analysis-lora --fine-tuning=lora --lr=5e-4 --warmup=0.06 --r=4
```


Full

```
python bert_sentiment_analysis.py --output-dir bert-sentiment-analysis-lora --fine-tuning=full --lr=5e-5 --warmup=0.1
```