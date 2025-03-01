# MAQD
This is the source code of MAQD model.

## Requirements
* Python (tested on 3.7.2)
* CUDA (tested on 11.3)
* [PyTorch](http://pytorch.org/) (tested on 1.11.0)
* [Transformers](https://github.com/huggingface/transformers) (tested on 4.18.0)
* numpy (tested on 1.21.6)
* spacy (tested on 2.3.7)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* tqdm
* sklearn
* scipy (tested on 1.5.2)
* redis

## Data
### Raw data
For data preparation and processing steps, please follow the instructions provided in the [CodRED repository](https://github.com/thunlp/CodRED/tree/master).

### Augmented query sentences from LLMs
Put the generated relation answers, paths and summaries into the `./data/answer`, `./data/path`  and `./data/summary`. 

## Run
### Sentence reranking
Save the splited sentences and their corresponding reranking scores with the following command:

```bash
>> python main_save.py --train --raw_only
>> python main_save.py --dev
>> python main_save.py --test
>> python main_rerank.py --train --raw_only
>> python main_rerank.py --dev
>> python main_rerank.py --test
```

### Model training and evaluation
Train and evaluate the MAQD model on CodRED dataset with the following command:
```bash
>> python main.py  --train --dev --test --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1 --learning_rate 3e-5 --epochs 10
```
