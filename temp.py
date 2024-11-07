# %%
import os
import json
import string
import numpy as np 
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
# from lightgbm import LGBMClassifier
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# %%
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from datasets import Dataset
from unsloth import is_bfloat16_supported

# Saving model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Warnings
import warnings
warnings.filterwarnings("ignore")


# %%
train = pl.read_csv("train.csv")
test = pl.read_csv("test.csv")

# %%
train = train.with_columns(pl.col("category").str.to_lowercase(), 
                    pl.col("sub_category").str.to_lowercase().fill_null("NULL"), 
                    pl.col("crimeaditionalinfo").str.to_lowercase(),
                    (pl.struct(["category", "sub_category"])
                    .map_elements(lambda e: json.dumps({"category": e["category"], "sub_category": e["sub_category"]})))
                    .alias("output"))
test = test.with_columns(pl.col("category").str.to_lowercase(), 
                    pl.col("sub_category").str.to_lowercase().fill_null("NULL"), 
                    pl.col("crimeaditionalinfo").str.to_lowercase(),
                    (pl.struct(["category", "sub_category"])
                    .map_elements(lambda e: json.dumps({"category": e["category"], "sub_category": e["sub_category"]})))
                    .alias("output"))

# %% [markdown]
# ### For Null text use category = "online financial fraud",  sub_category = "upi related frauds"

# %%
# train.filter(pl.col("len").is_null()).group_by(["category", "sub_category"]).len()

# %%
train = train.filter(pl.col("crimeaditionalinfo").is_not_null())
test = test.filter(pl.col("crimeaditionalinfo").is_not_null())

# %%
max_seq_length = 4000
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/SmolLM2-135M-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
    random_state = 32,
    loftq_config = None,
)
print(model.print_trainable_parameters())


# %%
import json
data_prompt = """Analyze the provided text from a Cybercrime Prevention Assisatant perspective. Identify the category and sub_category of the complaint reported.

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompt(df):
    inputs  = df["crimeaditionalinfo"]
    outputs = df["output"]
    texts = []
    for input_, output in zip(inputs, outputs):
        text = data_prompt.format(input_, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


# %%
training_data = Dataset.from_pandas(train.to_pandas())
training_data = training_data.map(formatting_prompt, batched=True)


# %%
trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=training_data,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        num_train_epochs=10,
        fp16=False,
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
    ),
)

trainer.train()


# %%
model = FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    data_prompt.format(
        #instructions
        text,
        #answer
        "",
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 5020, use_cache = True)
answer=tokenizer.batch_decode(outputs)
answer = answer[0].split("### Response:")[-1]
print("Answer of the question is:", answer)


# %%
model.save_pretrained("model/1B_finetuned_llama3.2")
tokenizer.save_pretrained("model/1B_finetuned_llama3.2")


# %%



