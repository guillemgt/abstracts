'''
    This file deals with the abstract classification.
    It tests 1000 abstracts and outputs the results to console.
'''

from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import pandas as pd


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


# Loading the dataset

def load_dataset_for_testing(tokenizer, num_samples, start_samples=0):
  print("Reading dataset")
  df = pd.read_json("../dataset/prepared.json", lines=True)
  print("Read", df.shape[0], "samples")
  #sample = df.sample(num_samples)
  sample = df[(start_samples):(start_samples+num_samples)]
  print("First sample is '{}'".format(sample.iloc[0]["title"]))
  print("Loading dataset")

  texts, labels = [], []
  for categories, title, abstract in zip(sample["categories"], sample["title"], sample["abstract"]):
    texts.append(f'<SENTENCE>Title: {title}\nAbstract: {abstract}\nCategories:')
    labels.append(categories)

  print("Loaded", num_samples, "samples")

  return texts, labels

original_text, original_label = load_dataset_for_testing(tokenizer, num_samples=500, start_samples=100000)


# Making predictions

model.eval()

predicted_label = []

for text, label in zip(original_text, original_label):
  generated = tokenizer(text, return_tensors="pt").input_ids.cuda()
  sample_outputs = model.generate(generated, do_sample=False, max_length=512)
  predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=False).split("</SENTENCE>")[0]
  split = predicted_text.split("Categories:")
  if len(split) < 2:
    predicted_categories = ""
  else:
    predicted_categories = split[1].strip()
  predicted_label.append(predicted_categories)

  print(text, predicted_categories, "(correct answer: {})".format(label))


# Computing scores

total = 0
all_correct = 0
at_least_one_correct = 0
for predicted, original in zip(predicted_label, original_label):
  pred_c = set(predicted.split())
  orig_c = set(original.split())

  intersection = pred_c.intersection(orig_c)

  score = len(intersection) / (len(pred_c) + len(orig_c) - len(intersection))

  total += score

  if score == 1:
    all_correct += 1

  if score != 0:
    at_least_one_correct += 1

print("Overall core:", total / len(predicted_label))
print("All correct:", all_correct / len(predicted_label))
print("At least one correct:", at_least_one_correct / len(predicted_label))