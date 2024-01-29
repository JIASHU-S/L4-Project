import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(device)

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from lime.lime_text import LimeTextExplainer
import numpy
import random
import transformers
transformers.logging.set_verbosity_error()

import pandas as pd
from tqdm import tqdm

format_ = '''
input: This is possibly the most perfect film I have ever seen - in acting, adaptation and direction. It can stand by itself as a perfect work of art!
## sentiment: positive

input: I gave this a 3 out of a possible 10 stars. Unless you like wasting your time watching an anorexic actress. There are various intrigues in the film.
## sentiment: negative

input: this film has no plot, no good acting, to be honest it has nothing, the same songs play over and over awful acting and if you can actually sit there and watch the whole thing and enjoy it there is something wrong with you. I wish i could give this 0 out of 10. If you have this film there are two things you can do sell it to someone who doesn't know about it or burn it!
## sentiment: negative

input: {}
## sentiment:

'''

data = pd.read_csv("imdb_dataset.csv")

def SentimentDataset():
    # prompt_template = "Example 1: The movie had a captivating storyline and kept me engaged throughout. It made me feel excited and entertained. ## Positive. Example 2: The customer service at the restaurant was exceptional. The staff was friendly and attentive, which made me feel valued and satisfied. ## Positive. Example 3: I found the user interface of the app to be confusing and cluttered, resulting in a sentiment of frustration. ## Negative. Example 4: The concert venue was poorly organized, with long queues and inadequate seating arrangements. ## Negative. Example 5: The book had beautiful prose and evocative descriptions. The language used was poetic, and it made me feel enchanted and moved. ## Positive. \nAnother text: {} Is it positive or negative? ##"

    labels = list(data['sentiment'])
    texts = list(data['review'])

    prompt_texts = [format_.format(text) for text in texts]

    return texts, prompt_texts, labels


tokenizer = GPT2Tokenizer.from_pretrained(r"./gpt2-m/")
model = GPT2LMHeadModel.from_pretrained(r"./gpt2-m/", device_map = "auto")


class_names = ['POSITIVE', 'NEGATIVE']
explainer = LimeTextExplainer(class_names=class_names)

# pipeline with prompt format as parameter
def my_pipe(input_, format_= format_):
    content = format_.format(input_)
    tokenized_input = tokenizer(content, return_tensors='pt').input_ids.to('cuda')

    outputs = model.generate(tokenized_input,
                             min_length = 1,
                             max_new_tokens = 1,
                             length_penalty = 2,
                             num_beams = 2,
                             no_repeat_ngram_size = 3,
                             temperature = 0.8,
                             top_k  = 50,
                             top_p = 0.92,
                             repetition_penalty = 2.1,
                             return_dict_in_generate=True,
                             output_scores=True,
                             pad_token_id=tokenizer.eos_token_id)

    out = torch.transpose(outputs['scores'][0], 0, 1)
    prob = torch.nn.functional.softmax(out, dim =0)
    out_prob = torch.max(prob,0)
    score = torch.IntTensor.item(out_prob.values[1])

    lis = numpy.empty((1,2))
    pred = tokenizer.decode(outputs[0][0], skip_special_tokens = True)
    if pred == 'negative':
        lis = numpy.array([1-score, score])
    else:
        lis = numpy.array([score, 1-score])

    return lis

def classifier_fxn(string_list):
    d = len(string_list)
    arr = numpy.empty((d,2))
    for idx, x in enumerate(string_list):
        arr[idx] = my_pipe(x)
    return arr

# pipeline for already complete prompt
def pipe_2(input):
    tokenized_input = tokenizer(input, return_tensors='pt').input_ids.to('cuda')

    outputs = model.generate(tokenized_input,
                             min_length = 1,
                             max_new_tokens = 1,
                             length_penalty = 2,
                             num_beams = 2,
                             no_repeat_ngram_size = 3,
                             temperature = 0.8,
                             top_k  = 50,
                             top_p = 0.92,
                             repetition_penalty = 2.1,
                             return_dict_in_generate=True,
                             output_scores=True,
                             pad_token_id=tokenizer.eos_token_id)

    out = torch.transpose(outputs['scores'][0], 0, 1)
    prob = torch.nn.functional.softmax(out, dim =0)
    out_prob = torch.max(prob,0)
    score = torch.IntTensor.item(out_prob.values[1])

    lis = numpy.empty((1,2))
    pred = tokenizer.decode(outputs[0][0], skip_special_tokens = True)
    if pred == 'negative':
        lis = numpy.array([1-score, score])
    else:
        lis = numpy.array([score, 1-score])

    return lis

def classifier_fxn2(string_list):
    d = len(string_list)
    arr = numpy.empty((d,2))
    for idx, x in enumerate(string_list):
        arr[idx] = pipe_2(x)
    return arr

def get_similar_word(x):
    content = f'What is a word with similar meaning to {x}, but not {x}'
    tokenized_input = tokenizer(content, return_tensors='pt').input_ids.to('cuda')

    outputs = model.generate(tokenized_input,
                             min_length = 1,
                             max_new_tokens = 1,
                             length_penalty = 2,
                             num_beams = 2,
                             no_repeat_ngram_size = 3,
                             temperature = 0.8,
                             top_k  = 50,
                             top_p = 0.92,
                             repetition_penalty = 2.1,
                             return_dict_in_generate=True,
                             output_scores=True,
                             pad_token_id=tokenizer.eos_token_id)

    pred = tokenizer.decode(outputs[0][0], skip_special_tokens = True)
    if pred != x:
        return pred
    else:
        return pred[:-1]
    
def exp_supporting_fid(org_input):
    org_pred = numpy.max(my_pipe(org_input))
    exp = explainer.explain_instance(org_input, classifier_fxn, num_features=6, num_samples=40)
    features = [x[0] for x in exp.as_list()]

    acc = 0
    for x in features:
        y = get_similar_word(x)
        pert_input = org_input.replace(f'{x} ', y)
        pert_pred = numpy.max(my_pipe(pert_input))
        acc += abs((1-(pert_pred/org_pred)))

    fid =  1- acc/len(features)
    return fid

def exp_contrary_fid(org_input):
    org_pred = numpy.max(my_pipe(org_input))
    exp = explainer.explain_instance(org_input, classifier_fxn, num_features=6, num_samples=40)
    features = [x[0] for x in exp.as_list()]

    acc = 0
    txt = org_input.split()
    sam = [x for x in txt if x not in set(features)]
    for x in features:
        y = random.choice(sam)
        pert_input = org_input.replace(f'{x} ', f'{y} ')
        pert_pred = numpy.max(my_pipe(pert_input))
        acc += abs((1-(pert_pred/org_pred)))

    fid =  acc/len(features)
    return fid

def main():
    contrary_fid_list = []
    supporting_fid_list = []
    
    ori_texts, res, labels = SentimentDataset()
    # print(res)
    for i in tqdm(range(len(ori_texts))):
        # Visualisation
        exp = explainer.explain_instance(ori_texts[i], classifier_fxn, num_features=8, num_samples=50)
        # print('document text: ', ori_texts[i])
        # print('probabilities: (positive negative) =', classifier_fxn([ori_texts[i]]))
        # print(exp.as_list())
        exp.save_to_file('visual_result/'+str(i)+'_lime.html',text=True)
        
        # Fidelity Calculation
        ori_input = ori_texts[i]
        contrary_fid = exp_contrary_fid(ori_input)
        supporting_fid = exp_supporting_fid(ori_input)
        contrary_fid_list.append(contrary_fid)
        supporting_fid_list.append(supporting_fid)
        
    data["supporting_fid"] = supporting_fid_list
    data["contrary_fid"] = contrary_fid_list
    data.to_csv("fid_results.csv",index=False)
    
    print("Average Contrary Score: ", sum(contrary_fid_list)/len(contrary_fid_list))
    print("Average Supporting Score: ", sum(supporting_fid_list)/len(supporting_fid_list))
    
if __name__=="__main__":
    main()