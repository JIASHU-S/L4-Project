import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import pandas as pd
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast
from tqdm import tqdm

device = torch.device("cuda")

model = PegasusForConditionalGeneration.from_pretrained(r"./pegasus_paraph")
model.to(device)
model.eval()

tokenizer = PegasusTokenizerFast.from_pretrained(r"./pegasus_paraph")


df=pd.read_csv('IMDB Dataset.csv')
df['length']=[len(t) for t in df['review']]
df=df.sort_values('length',ignore_index=True)
positive_df = df.loc[df['sentiment']=="positive", :].reset_index(drop=True).loc[:249, :]
negative_df = df.loc[df['sentiment']=="negative", :].reset_index(drop=True).loc[:249, :]
df = pd.concat([positive_df, negative_df], ignore_index=True)
reviews=[]
sentiments=[]
for i in range(len(df)):
    reviews.append(df['review'][i])
    sentiments.append(df['sentiment'][i])


def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=3, num_beams=5):
  # tokenize the text to be form of a list of token IDs
  inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
  inputs = inputs.to(device)
  # generate the paraphrased sentences
  with torch.no_grad():
    outputs = model.generate(
      **inputs,
      num_beams=num_beams,
      num_return_sequences=num_return_sequences,
    )
  # decode the generated sentences using the tokenizer to get them back to text
  return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def return_paraphrases(reviews):
    paraphrased_reviews=[]
    for i in tqdm(range(len(reviews))):
        review = reviews[i]
        sentence_list=get_paraphrased_sentences(model, tokenizer, review, num_beams=5, num_return_sequences=5)
        paraphrased_reviews.append(sentence_list[0])
        print(review, sentence_list)
    return paraphrased_reviews


# reviews=reviews[0:10]
paraphrased_reviews=return_paraphrases(reviews)
df["paraphrase"] = paraphrased_reviews
df.to_csv("paraphrase.csv",index=False)
# print(reviews)
# print("\n")
# print(paraphrased_reviews)