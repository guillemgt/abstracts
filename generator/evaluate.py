'''
    This file deals with the abstract generation.
    It outputs a few results to terminal or many to a file (or more).
'''

import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

import os
import random
import re
import json


# Loading the model

load_saved_model = True
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

model.eval()


# Evaluate

SAVE_TO_FILE = False
RESULTS_PATH = "results/"

if SAVE_TO_FILE:

  if not os.path.exists(RESULTS_PATH):
      os.makedirs(RESULTS_PATH)

  MAX_CATEGORIES = 3
  CATEGORIES = ["math.AC", "math.AG", "math.AP", "math.AT", "math.CA", "math.CO", "math.CT", "math.CV", "math.DG", "math.DS", "math.FA", "math.GM", "math.GN", "math.GR", "math.HO", "math.IT", "math.KT", "math.LO", "math.MG", "math.MP", "math.NT", "math.OA", "math.OC", "math.PR", "math.QA", "math.RA", "math.RT", "math.SG", "math.SP", "math.ST"]
  NUM = 1000
  NUM_PER_FILE = 1000
  FILES = NUM // NUM_PER_FILE

  for file_num in range(FILES):

    result = []
    for i in range(0, NUM_PER_FILE):
      l = random.randint(1, MAX_CATEGORIES)
      categories = " ".join(random.sample(CATEGORIES, l))

      prompt = f'<SENTENCE>Categories: {categories}\nTitle:'
      generated = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
      sample_outputs = model.generate(generated, do_sample=True, top_p=0.96, max_length=512, temperature=0.65, num_beams=1, repetition_penalty=1.15, no_repeat_ngram_size=6, num_return_sequences=1)

      predicted_texts = tokenizer.batch_decode(sample_outputs, skip_special_tokens=False)
      for predicted_text in predicted_texts:
        clean_text = predicted_text.split("</SENTENCE>")[0].split("Categories: ")[1]
        splits = re.split('(?:\nTitle: )|(?:\nAbstract: )', clean_text)
        if len(splits) == 3:
            result.append({
              "categories": splits[0].split(),
                "title": splits[1].strip(),
                "abstract": splits[2].strip()
            })
        print("[{} {}]".format(file_num, i), clean_text + "\n")

    with open(os.path.join(RESULTS_PATH, "results_" + str(file_num) + ".json"), "w") as of:
      of.write(json.dumps(result))
      of.close()

else:

  torch.manual_seed(100)
  prompt = f'<SENTENCE>Categories: math.NT math.RT\nTitle:'
  generated = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
  sample_outputs = model.generate(generated, do_sample=True, top_p=0.96, max_length=512, temperature=0.65, num_beams=1, repetition_penalty=1.15, no_repeat_ngram_size=6, num_return_sequences=4)
  predicted_texts = tokenizer.batch_decode(sample_outputs, skip_special_tokens=False)
  for predicted_text in predicted_texts:
    clean_text = predicted_text.split("</SENTENCE>")[0].split("Categories: ")[1]
    print(clean_text + "\n")
