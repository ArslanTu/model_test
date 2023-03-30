# model_test

test ChatGLM-6B、T5-3B、Flan-T5-3B on MMLU dataset.

# repo list

- [hendrycks/test](https://github.com/hendrycks/test)：MMLU datasets
- [THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)：ChatGLM-6B model
- [lich99/ChatGLM-finetune-LoRA](https://github.com/lich99/ChatGLM-finetune-LoRA)：finetuning ChatGLM-6B

# How to run

## Evaluate ChatGLM-6B

1. run the api with `python3 api.py` under `ChatGLM-6B/`
2. run the evaluate script with `python3 evaluate_chatglm.py -e chatglm` under `test/`

### Process

Run a ChatGLM-6B API instance at local port (8000), send prompt through POST and get a response. We assume the first character is the choice which the model select. Then we can calculate an accuracy.

In fact, with observation of the run log (either api or evaluate script), this assumption holds true in most cases.

## Evaluate T5-3B or Flan-T5-3B

run the evaluate script with `python3 evaluate_t5.py -m t5-3b` or `python3 evaluate_t5.py -m flan-t5-3b`

All results will be stored under `test/results` as `.csv` file.

### Process

Make a model instance, feed the prompt, extract the logits and apply softmax to it and get a probability list for all choice which the model select. Take the choice with highest probability as model's answer. Then we can calculate the accuracy.