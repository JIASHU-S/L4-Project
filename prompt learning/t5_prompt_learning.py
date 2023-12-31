import pandas as pd
import os
import logging
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm, T5TokenizerWrapper
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptDataLoader, PromptForClassification
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(df):
    df = df.replace(-1, 0)
    df['label'] = df['toxic'] + df['severe_toxic'] + df['obscene'] + df['threat'] + df['insult'] + df['identity_hate']
    df['label'] = df['label'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1)
    df_records = df.to_dict(orient='records')

    # Adding unique IDs to each record if not present
    for i, record in enumerate(df_records):
        if 'id' not in record:
            record['id'] = f'data_{i}'

    return df_records

# Function to add in-context examples
def add_contextual_examples(data, num_examples=6):
    # Adjusted list of contextual examples with ids
    contextual_examples = [
        {"id": "context_1", "comment_text": "Example of toxicity: 'You are an idiot!' - This comment is aggressive and insulting.", "label": 1},
        {"id": "context_2", "comment_text": "Non-toxic example: 'I disagree with your point, but I see where you're coming from.' - This comment respectfully disagrees.", "label": 0},
        {"id": "context_3", "comment_text": "Toxic behavior includes personal attacks, like: 'You're always spouting nonsense!' - This is derogatory.", "label": 1},
        {"id": "context_4", "comment_text": "Healthy discussion example: 'Your evidence is compelling. I'll reconsider my stance.' - Open and respectful.", "label": 0},
        {"id": "context_5", "comment_text": "Hate speech is toxic: 'People like you should not be allowed to speak.' - Targeting groups is harmful.", "label": 1},
        {"id": "context_6", "comment_text": "Constructive feedback is non-toxic: 'Your idea has merit, but here's how it might be improved.' - This is helpful and polite.", "label": 0}
    ]
    
    # Prepend the contextual examples to the actual data
    return contextual_examples + data



def main():
    data_path = 'F:\L4-Project\prompt learning'

    train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
    train_data = preprocess_data(train_data)
    # Adding contextual examples to training data
    train_data = add_contextual_examples(train_data)

    test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
    test_label_data = pd.read_csv(os.path.join(data_path, 'test_labels.csv'))
    merged_test_data = pd.merge(test_data, test_label_data, on='id')
    merged_test_data = preprocess_data(merged_test_data)
    # Adding contextual examples to test data
    merged_test_data = add_contextual_examples(merged_test_data)

    raw_dataset = {
        'train': train_data,
        'test': merged_test_data
    }

    classes = [0, 1]

    logging.info(raw_dataset['train'][0])

    dataset = {}
    for split in ['train', 'test']:
        dataset[split] = []
        for data in raw_dataset[split]:
            input_example = InputExample(text_a=data['comment_text'], label=int(data['label']), guid=data['id'])
            dataset[split].append(input_example)
    logging.info(dataset['train'][0])

    plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-small")
    mytemplate = ManualTemplate(tokenizer=tokenizer, text='Comment: {"placeholder":"text_a"} Question: Is the comment toxic? {"mask"}.')
    wrapped_t5tokenizer = T5TokenizerWrapper(max_seq_length=256, decoder_max_length=3, tokenizer=tokenizer, truncate_method="head")

    model_inputs = {}
    for split in ['train', 'test']:
        model_inputs[split] = []
        for sample in tqdm(dataset[split]):
            tokenized_example = wrapped_t5tokenizer.tokenize_one_example(mytemplate.wrap_one_example(sample), teacher_forcing=False)
            model_inputs[split].append(tokenized_example)

    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                       batch_size=64, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                       truncate_method="head")

    myverbalizer = ManualVerbalizer(
        tokenizer,
        num_classes=2,
        classes=classes,
        label_words={
            1: ["yes"],
            0: ["no"],
        }
    )
    logging.info(myverbalizer.label_words_ids)
    logits = torch.randn(2, len(tokenizer))  # creating a pseudo output from the plm
    logging.info(myverbalizer.process_logits(logits))

    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    use_cuda = True
    if use_cuda:
        prompt_model = prompt_model.cuda()

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

    for epoch in range(1):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 1:
                logging.info("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)))

    validation_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                             tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                             batch_size=4, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                             truncate_method="head")

    all_preds = []
    all_labels = []
    for step, inputs in enumerate(validation_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    acc = sum([int(i == j) for i, j in zip(all_preds, all_labels)]) / len(all_preds)
    logging.info(f"Accuracy: {acc:.4f}")

    correct_count = sum([int(i == j) for i, j in zip(all_preds, all_labels)])
    total_count = len(all_preds)
    logging.info(f"Correctly predicted cases: {correct_count} out of {total_count}")
    logging.info(f"Incorrectly predicted cases: {total_count - correct_count} out of {total_count}")

     # Calculate precision, recall, and F1 score
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # Log the results
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")



if __name__ == "__main__":
    main()
