'''
    This file deals with the fine-tuning of the gpt-neo model for the purposes of abstract generation.
'''

import torch
from torch.utils.data import Dataset

from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments

import pandas as pd



# Loading the model

load_saved_model = False
if load_saved_model:
    tokenizer = GPT2Tokenizer.from_pretrained("tokenizer")
    model = GPTNeoForCausalLM.from_pretrained("model").cuda()
    print("Loaded saved model")
else:
    model_name = "EleutherAI/gpt-neo-125M"

    tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token='<SENTENCE>', eos_token='</SENTENCE>', pad_token='<PAD/>')
    model = GPTNeoForCausalLM.from_pretrained(model_name).cuda()

    model.resize_token_embeddings(len(tokenizer))
    print("Loaded new model")


# Loading the dataset

class ArxivDataset(Dataset):
    def __init__(self, categories_list, title_list, abstract_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []

        for categories, title, abstract in zip(categories_list, title_list, abstract_list):
            prep_txt = f'<SENTENCE>Categories: {categories}\nTitle: {title}\nAbstract: {abstract}</SENTENCE>'
            encodings_dict = tokenizer(prep_txt, truncation=True, max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            
        self.labels = self.input_ids.copy()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.attn_masks[index], self.labels[index]

def load_dataset(tokenizer, num_samples, start_samples=0):
    print("Reading dataset")
    df = pd.read_json("../dataset/prepared.json", lines=True)
    print("Read", df.shape[0], "samples")
    train_sample = df[(start_samples):(start_samples+num_samples)]
    print("First sample is '{}'".format(train_sample.iloc[0]["title"]))
    print("Loading dataset")
    train_dataset = ArxivDataset(train_sample["categories"], train_sample["title"], train_sample["abstract"], tokenizer, max_length=512)
    print("Loaded", num_samples, "samples")
    
    return train_dataset


train_dataset = load_dataset(tokenizer, 100000, start_samples=0)


# Do the training

_ = model.train()
training_args = TrainingArguments(output_dir='training_models', num_train_epochs=1, logging_steps=100, save_steps=2500, per_device_train_batch_size=8, per_device_eval_batch_size=8, warmup_steps=100, weight_decay=0.01, logging_dir='logs', learning_rate=5e-5, lr_scheduler_type="linear")
Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]), 'attention_mask': torch.stack([f[1] for f in data]), 'labels': torch.stack([f[0] for f in data])}).train()


# Save the model
tokenizer.save_pretrained("tokenizer")
model.save_pretrained("model")