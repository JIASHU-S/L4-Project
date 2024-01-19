import os


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers
transformers.logging.set_verbosity_error()

import pandas as pd
from tqdm import tqdm

data = pd.read_csv("F:\L4-Project\word_attribute")

def SentimentDataset():
    prompt_template = "Example 1: The movie had a captivating storyline and kept me engaged throughout. It made me feel excited and entertained. ## Positive. Example 2: The customer service at the restaurant was exceptional. The staff was friendly and attentive, which made me feel valued and satisfied. ## Positive. Example 3: I found the user interface of the app to be confusing and cluttered, resulting in a sentiment of frustration. ## Negative. Example 4: The concert venue was poorly organized, with long queues and inadequate seating arrangements. ## Negative. Example 5: The book had beautiful prose and evocative descriptions. The language used was poetic, and it made me feel enchanted and moved. ## Positive. \nAnother text: {} Is it positive or negative? ##"

    labels = data['sentiment']
    texts = data['review']

    prompt_texts = [prompt_template.format(text) for text in texts]

    return prompt_texts, labels

def test():
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    # Load dataset
    texts, labels = SentimentDataset()
    assert len(texts)==len(labels)
    
    # Metrics
    class_nums = [25000,25000]
    counts = {"positive": 0 , "negative": 0}
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    prediction_list = []
    
    for i in tqdm(range(len(labels))):
        text = texts[i]
        label = labels[i]
        default_strlen = len(text)

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
                            early_stopping=True
                        )
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = output[default_strlen:].lower()
        prediction_list.append(prediction)

        # if i % 10 == 0:
        #     print(prediction)
        
        if label in prediction:
            if label=="positive":
                tp += 1
            else:
                tn += 1
            counts[label] += 1
        else:
            if label=="positive":
                fp += 1
            else:
                fn += 1

    
    # Save the prediction result to a new file
    data["prediction"] = prediction_list
    data.to_csv("predicted_result.csv",index=False)
    
    # Calculate the metrics
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    # Print the metrics
    print("Overall accuracy: %.4f" % accuracy)

    print("Accuracy per class: ")
    print("\t Positive: %.4f, Negative: %.4f" % (counts["positive"]/class_nums[0], counts["negative"]/class_nums[1]))

    print("Precision: %.4f \nRecall: %.4f \nF1: %.4f" % (precision,recall,f1_score))
    
    
if __name__=="__main__":
    test()