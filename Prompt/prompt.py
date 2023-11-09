import os
import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn
from transformers import AutoTokenizer,AutoModel,BertModel,BertTokenizer
from tqdm import tqdm
import pandas as pd


def get_data(path, max_len=None, mode='train'):

    # Get data
# Args:
#     path (_type_): File path
#     max_len (_type_, optional): Number of data to be used for training/testing. Defaults to None.
#     mode (str, optional): Load training set data or test set data. Defaults to 'train'.
# Returns:
#     _type_: Returns text and corresponding labels

    df = pd.read_csv(os.path.join(path, f'{mode}.csv'))

    all_text = df.comment_text.values.tolist()
    all_label = df.toxic.values.tolist()
    all_label = [index_2_label[int(i)] for i in all_label]

    if max_len is not None:
        return all_text[:max_len], all_label[:max_len]
    return all_text, all_label


class TCdataset(Dataset):
    def __init__(self,all_text,all_label,tokenizers, max_length=500):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizers = tokenizers
        self.prompt_text1 = "This comment text \""
        self.prompt_text2 = "\" is "  #   toxic  or  profitable
    def __getitem__(self, x):
        text = self.all_text[x][:450]
        label = self.all_label[x]

        text_prompt = self.prompt_text1 + text + self.prompt_text2
        return text_prompt, label, len(text_prompt)+1, label # Give it one [MASK], so add 1
    def process_data(self,data):
        batch_text,batch_label,batch_len,cla = zip(*data)
        batch_max = max(batch_len)+1

        batch_text_idx = []
        batch_label_idx = []
        for text, label in zip(batch_text,batch_label):
            text = text+"[MASK]"
            text_idx = self.tokenizers.encode(text, add_special_tokens=True)  # be+xxx+[MASK][MASK]+ed label[len_]

            label_idx = [-100]*(len(text_idx)-2) + self.tokenizers.encode(label, add_special_tokens=False)

            text_idx += [0]*(batch_max-len(text_idx))
            label_idx += [-100]*(batch_max-len(label_idx))

            assert(len(text_idx)==len(label_idx))

            batch_text_idx.append(text_idx)
            batch_label_idx.append(label_idx)

        return torch.tensor(batch_text_idx),torch.tensor(batch_label_idx),cla

    def __len__(self):
        return len(self.all_text)

class Bert_Model(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = BertModel.from_pretrained(model_name)
        self.generater = nn.Linear(768, 30522)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,x,label=None):
        x,_ = self.backbone(x, return_dict=False, attention_mask = ( (x!=103) & (x!=0) ))

        x = self.generater(x)

        if label is not None:
            loss = self.loss_fn(x.reshape(-1,x.shape[-1]),label.reshape(-1))
            return loss
        else:
            return x

    def get_emb(self, x):
        x,_ = self.backbone(x, return_dict=False, attention_mask = ( (x!=103) & (x!=0) ))
        return x


if __name__ == "__main__":
    with open(os.path.join('./data/raw','index_2_label.txt'), 'r', encoding='utf-8') as f:
        index_2_label = f.read().split('\n')
    dict_ver = [15282, 11704]

    train_text, train_label = get_data(os.path.join('./data/raw'), max_len=10000)
    dev_text, dev_label = get_data(os.path.join('./data/raw'), max_len=300, mode='dev')
    test_text, test_label = get_data(os.path.join('./data/raw'), max_len=300, mode='test')

    batch_size = 8
    epoch = 10
    lr = 1e-5
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    model_name = "bert-base-uncased"

    tokenizers = BertTokenizer.from_pretrained(model_name)

    train_dataset = TCdataset(train_text,train_label,tokenizers)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=train_dataset.process_data)
    dev_dataset = TCdataset(dev_text, dev_label, tokenizers)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=dev_dataset.process_data)
    test_dataset = TCdataset(test_text, test_label, tokenizers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.process_data)

    model = Bert_Model(model_name).to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    print(f'train on {device}.....')
    for e in range(epoch):
        loss_sum = 0
        ba = 0
        for x,y,cla in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            loss = model(x,y)
            loss.backward()
            opt.step()

            loss_sum+=loss
            ba += 1
        print(f'e = {e} loss = {loss_sum / ba:.6f}')
        right = 0
        model.eval()
        for x, y,cla in tqdm(dev_dataloader):
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                pre = model(x)

            for p, label, cl in zip(pre, y, cla):
                p = p[label != -100]
                w = torch.argmax(p[0][dict_ver])
                right += int(index_2_label[w]==cl)
        print(f'acc={right/len(dev_dataset):.5f}')
        model.train()

    results = []
    model.eval()
    for x, y, cla in tqdm(test_dataloader):
        x = x.to(device)
        y = y.to(device)
        id = torch.argmax((x == 103).float(), axis=-1)

        with torch.no_grad():
            pre = model(x)

        for p, i, _ in zip(pre, id, cla):
            p = p[i]
            w = torch.argmax(p[dict_ver])
            results.append(w.cpu().detach().numpy())
    
    results = pd.DataFrame(results,columns=['toxic'])
    results.to_csv('./test_predict.csv', index=False)