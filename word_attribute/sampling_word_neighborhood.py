import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import pandas as pd
import nltk
from nltk import pos_tag
from nltk.tree import Tree
from nltk.chunk import conlltags2tree
import nltk
from transformers import pipeline, BertTokenizer, BertForMaskedLM
from tqdm import tqdm
import re
import torch
device = torch.device("cuda")

tokenizer = BertTokenizer.from_pretrained(r'./bert-fill')
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
model = BertForMaskedLM.from_pretrained(r'./bert-fill').to(device)
model.eval()
pipeline_model = pipeline('fill-mask', model=model, tokenizer=tokenizer,device=device)

df=pd.read_csv('IMDB Dataset.csv')
df['length']=[len(t) for t in df['review']]
df=df.sort_values('length',ignore_index=True)
sample_num = 500
positive_df = df.loc[df['sentiment']=="positive", :].reset_index(drop=True).loc[:249, :]
negative_df = df.loc[df['sentiment']=="negative", :].reset_index(drop=True).loc[:249, :]
df = pd.concat([positive_df, negative_df], ignore_index=True)
reviews=[]
sentiments=[]
for i in range(len(df)):
    reviews.append(df['review'][i].replace('<br /><br />', ' '))
    sentiments.append(df['sentiment'][i])

# Sampling
def count_string_occurrences(string, target_string):
    count = 0
    start_index = 0

    while True:
        index = string.find(target_string, start_index)
        if index == -1:
            break
        count += 1
        start_index = index + len(target_string)

    return count

def generate_masked_and_verb_perturbations(texts):
    replacements_list=[]

    for i in tqdm(range(len(texts)),desc="verb replacement"):
        replacements = {}
        pos_tagged = nltk.pos_tag(texts[i].split())
        verbs = []
        for item in pos_tagged:
            if('VB' in item[1]):
                verbs.append(item[0])
        verbs = list(set(verbs))
        for v in verbs:
            found_results = re.findall(r"\b[\w']+\b", v)
            if found_results:
                v = found_results[0]
            else:
                continue
            pattern = r'\b' + re.escape(v) + r'\b'
            masked_sentence = re.sub(pattern, tokenizer.mask_token, texts[i])
            pred = pipeline_model(masked_sentence,tokenizer_kwargs=tokenizer_kwargs)
            mask_num = count_string_occurrences(masked_sentence,tokenizer.mask_token)
            if mask_num >= 1 and type(pred[0])==type([]):   # many [MASK] tokens
                for j in range(len(pred[0])):
                    possible_ans = pred[0][j]['token_str']
                    if possible_ans!=v:
                        replaced_sentence = re.sub(pattern, possible_ans, texts[i])
                        replacements[v] = replaced_sentence
                        break
            elif mask_num >= 1:
                for j in range(len(pred)):
                    possible_ans = pred[j]['sequence']
                    if possible_ans!=v:
                        replacements[v] = possible_ans
        replacements_list.append(replacements)
    return replacements_list

def generate_masked_and_nouns_perturbations(texts):
    replacements_list=[]

    for i in tqdm(range(len(texts)),desc="noun replacement"):
        replacements={}
        pos_tagged = nltk.pos_tag(texts[i].split())
        nouns = []
        for item in pos_tagged:
            if('NN' in item[1]):
                nouns.append(item[0])
        nouns = list(set(nouns))

        for v in nouns:
            found_results = re.findall(r"\b[\w']+\b", v)
            if found_results:
                v = found_results[0]
            else:
                continue
            pattern = r'\b' + re.escape(v) + r'\b'
            masked_sentence = re.sub(pattern, tokenizer.mask_token, texts[i])
            pred = pipeline_model(masked_sentence,tokenizer_kwargs=tokenizer_kwargs)
            mask_num = count_string_occurrences(masked_sentence,tokenizer.mask_token)
            if mask_num >= 1 and type(pred[0])==type([]):   # many [MASK] tokens
                for j in range(len(pred[0])):
                    possible_ans = pred[0][j]['token_str']
                    if possible_ans!=v:
                        replaced_sentence = re.sub(pattern, possible_ans, texts[i])
                        replacements[v] = replaced_sentence
                        break
            elif mask_num >= 1:
                for j in range(len(pred)):
                    possible_ans = pred[j]['sequence']
                    if possible_ans!=v:
                        replacements[v] = possible_ans
        replacements_list.append(replacements)
    return replacements_list


def generate_masked_and_adjective_perturbations(texts):
    replacements_list=[]

    for i in tqdm(range(len(texts)),desc="adjective replacement"):
        replacements={}
        pos_tagged = nltk.pos_tag(texts[i].split())
        nouns = []
        
        for item in pos_tagged:
            if('JJ' in item[1]):
                nouns.append(item[0])
        nouns = list(set(nouns))

        for v in nouns:
            found_results = re.findall(r"\b[\w']+\b", v)
            if found_results:
                v = found_results[0]
            else:
                continue
            pattern = r'\b' + re.escape(v) + r'\b'
            masked_sentence = re.sub(pattern, tokenizer.mask_token, texts[i])
            pred = pipeline_model(masked_sentence,tokenizer_kwargs=tokenizer_kwargs)
            mask_num = count_string_occurrences(masked_sentence,tokenizer.mask_token)
            if mask_num >= 1 and type(pred[0])==type([]):   # many [MASK] tokens
                for j in range(len(pred[0])):
                    possible_ans = pred[0][j]['token_str']
                    if possible_ans!=v:
                        replaced_sentence = re.sub(pattern, possible_ans, texts[i])
                        replacements[v] = replaced_sentence
                        break
            elif mask_num >= 1:
                for j in range(len(pred)):
                    possible_ans = pred[j]['sequence']
                    if possible_ans!=v:
                        replacements[v] = possible_ans
        replacements_list.append(replacements)
    return replacements_list

v_replacements=generate_masked_and_verb_perturbations(reviews)
n_replacements=generate_masked_and_nouns_perturbations(reviews)
adj_replacements=generate_masked_and_adjective_perturbations(reviews)

# write them into a csv file
df['v_replacement'] = v_replacements
df['n_replacement'] = n_replacements
df['adj_replacement'] = adj_replacements

df_new = df[['review', 'sentiment', 'v_replacement', 'n_replacement','adj_replacement']]
df.to_csv("word_replacement.csv",index=False)