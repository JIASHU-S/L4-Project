import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers
transformers.logging.set_verbosity_error()

import pandas as pd
from tqdm import tqdm
import ast

data = pd.read_csv("word_replacement.csv")

def SentimentDataset():
    prompt_template = "Example 1: The movie had a captivating storyline and kept me engaged throughout. It made me feel excited and entertained. ## Positive. Example 2: The customer service at the restaurant was exceptional. The staff was friendly and attentive, which made me feel valued and satisfied. ## Positive. Example 3: I found the user interface of the app to be confusing and cluttered, resulting in a sentiment of frustration. ## Negative. Example 4: The concert venue was poorly organized, with long queues and inadequate seating arrangements. ## Negative. Example 5: The book had beautiful prose and evocative descriptions. The language used was poetic, and it made me feel enchanted and moved. ## Positive. \nAnother text: {} Is it positive or negative? ##"

    texts = data['review']
    v_replacements = data["v_replacement"]
    n_replacements = data["n_replacement"]
    adj_replacements = data["adj_replacement"]
    
    # v_replacements = [ast.literal_eval(t) for t in v_replacements]
    # n_replacements = [ast.literal_eval(t) for t in n_replacements]
    # adj_replacements = [ast.literal_eval(t) for t in adj_replacements]
    

    prompt_texts = [prompt_template.format(text) for text in texts]
    
    prompt_v = []
    prompt_n = []
    prompt_adj = []

    for v_rep in v_replacements:
        v_rep = ast.literal_eval(v_rep)
        prompt_v_rep = {}
        for _, (key, value) in enumerate(v_rep.items()):
            prompt_v_rep[key] = prompt_template.format(value)
        prompt_v.append(prompt_v_rep)
    
    for n_rep in n_replacements:
        n_rep = ast.literal_eval(n_rep)
        prompt_n_rep = {}
        for _, (key, value) in enumerate(n_rep.items()):
            prompt_n_rep[key] = prompt_template.format(value)
        prompt_n.append(prompt_n_rep)

    for adj_rep in adj_replacements:
        adj_rep = ast.literal_eval(adj_rep)
        prompt_adj_rep = {}
        for _, (key, value) in enumerate(adj_rep.items()):
            prompt_adj_rep[key] = prompt_template.format(value)
        prompt_adj.append(prompt_adj_rep)
        
    return prompt_texts, prompt_v, prompt_n, prompt_adj

def test():
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(r"./gpt2")
    model = AutoModelForCausalLM.from_pretrained(r"./gpt2")
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    # Load dataset
    texts, prompt_vs, prompt_ns, prompt_adjs = SentimentDataset()
    assert len(texts)==len(prompt_ns)==len(prompt_vs)==len(prompt_adjs)
    
    # Metrics
    
    verb_losses = []
    noun_losses = []
    adj_losses = []
    
    for i in tqdm(range(len(texts))):
        # the default input
        text = texts[i]
        tokenized_input = tokenizer.encode_plus(
                        text,
                        max_length=1000,
                        truncation=True,
                        return_tensors='pt'
                    )
        tokenized_input = tokenized_input.to(device)
        with torch.no_grad():
            outputs = model.generate(
                            **tokenized_input,
                            max_new_tokens=1,
                            num_beams=5,
                            early_stopping=True,
                            output_scores=True,
                            return_dict_in_generate=True
                        )
        logits = outputs.scores
        last_token_logits = logits[0][-1,:]
        prob_default = torch.softmax(last_token_logits, dim=-1)


        # verb replacements
        verb_loss = {}
        for _,(key,value) in enumerate(prompt_vs[i].items()):
            tokenized_input = tokenizer.encode_plus(
                            value,
                            max_length=1000,
                            truncation=True,
                            return_tensors='pt'
                        )
            tokenized_input = tokenized_input.to(device)
            with torch.no_grad():
                outputs = model.generate(
                                **tokenized_input,
                                max_new_tokens=1,
                                num_beams=5,
                                early_stopping=True,
                                output_scores=True,
                                return_dict_in_generate=True
                            )
            logits = outputs.scores
            last_token_logits = logits[0][-1,:]
            prob_verb = torch.softmax(last_token_logits, dim=-1)
            entropy = F.binary_cross_entropy(prob_verb, prob_default)
            verb_loss[key] = entropy.item()
            
        verb_losses.append(verb_loss)
        
        # noun replacements
        noun_loss = {}
        for _,(key,value) in enumerate(prompt_ns[i].items()):
            tokenized_input = tokenizer.encode_plus(
                            value,
                            max_length=1000,
                            truncation=True,
                            return_tensors='pt'
                        )
            tokenized_input = tokenized_input.to(device)
            with torch.no_grad():
                outputs = model.generate(
                                **tokenized_input,
                                max_new_tokens=1,
                                num_beams=5,
                                early_stopping=True,
                                output_scores=True,
                                return_dict_in_generate=True
                            )
            logits = outputs.scores
            last_token_logits = logits[0][-1,:]
            prob_noun = torch.softmax(last_token_logits, dim=-1)
            entropy = F.binary_cross_entropy(prob_noun, prob_default)
            noun_loss[key] = entropy.item()
        noun_losses.append(noun_loss)
        
        # adj replacements
        adj_loss = {}
        for _,(key,value) in enumerate(prompt_adjs[i].items()):
            tokenized_input = tokenizer.encode_plus(
                            value,
                            max_length=1000,
                            truncation=True,
                            return_tensors='pt'
                        )
            tokenized_input = tokenized_input.to(device)
            with torch.no_grad():
                outputs = model.generate(
                                **tokenized_input,
                                max_new_tokens=1,
                                num_beams=5,
                                early_stopping=True,
                                output_scores=True,
                                return_dict_in_generate=True
                            )
            logits = outputs.scores
            last_token_logits = logits[0][-1,:]
            prob_adj = torch.softmax(last_token_logits, dim=-1)
            entropy = F.binary_cross_entropy(prob_adj, prob_default)
            adj_loss[key] = entropy.item()
        adj_losses.append(adj_loss)
        
    # Save the result to a new file
    data["v_entropy"] = verb_losses
    data["n_entropy"] = noun_losses
    data["adj_entropy"] = adj_losses
    data.to_csv("word_attribute.csv",index=False)
    
    
    
if __name__=="__main__":
    test()