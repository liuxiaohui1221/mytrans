import os.path
import numpy as np
from transformers import AutoTokenizer
from models.ModelPath import get_t5_path, get_data_path, get_models_path
from reader.data_reader import load_dataset

model_path=os.path.join(get_t5_path(),"pretrained")
tokenizer = AutoTokenizer.from_pretrained(model_path)
source_lang = "fr"
target_lang = "zh"
prefix = source_lang + " to " + target_lang+" "
books = load_dataset(source_lang+"_"+target_lang)
books = books["train"].train_test_split(test_size=0.001)


def preprocess_function(examples):
    inputs = [prefix + example for example in examples[source_lang]]
    targets = examples[target_lang]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_books = books.map(preprocess_function, batched=True)

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    fp16=True
)
from datasets import load_metric

# metric = load_metric("../metric/bleu.py")
metric = load_metric("sacrebleu")
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(np.argmax(preds,axis=-1), skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    print(result)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions1 = np.argmax(logits[0], axis=-1)
#     predictions2 = np.argmax(logits[1], axis=-1)
#     return metric.compute(predictions=predictions1, references=labels)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.evaluate()
trainer.train()
print(trainer.evaluate())