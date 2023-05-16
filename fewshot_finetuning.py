import pandas as pd
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from transformers import AutoFeatureExtractor, ASTForAudioClassification, Trainer, AutoModelForAudioClassification
import pdb
import librosa
import numpy as np

from collections import defaultdict
from tqdm import tqdm

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        
        encoding = torch.tensor(self.encodings[idx])
        label = self.labels[idx]
        label = nn.functional.one_hot(torch.tensor(label, dtype=torch.long), num_classes=14)
        label = label.type(torch.FloatTensor)
        item = {'encoding': encoding, 'category': label}
        return item

    def __len__(self):
        return len(self.encodings)

class few_shot_finetuning_layer(nn.Module):
    def __init__(self, w_init):
        super(few_shot_finetuning_layer, self).__init__()
        self.W = torch.nn.Parameter(w_init)
        self.W.requires_grad = True
        self.b = torch.nn.Parameter(torch.Tensor([0]))
        self.b.requires_grad = True
    def forward(self, x):
        x = torch.matmul(self.W,x)+self.b
        return x

if __name__ == "__main__": 

    data = pd.read_csv("few-shot.csv")
    cat = ['Ambulance (siren)','Crying, sobbing','Explosion','Gunshot, gunfire','Laughter','Screaming', 'Smash, crash']
    data = data.sample(frac=1).reset_index(drop=True)

    shot = 5
    group = data.groupby('category')
    train = group.head(shot).groupby('category')
    test = group.tail(30).groupby('category')

    input_dir = '/home/ubuntu/data/wav'
    feature_extractor = AutoFeatureExtractor.from_pretrained("bookbot/distil-ast-audioset")
    model = AutoModelForAudioClassification.from_pretrained("bookbot/distil-ast-audioset")#.to('cuda')


    model.eval()
    mu = {}
    q = defaultdict(lambda: [])

    ##support set
    x_j = []
    y_j = []

    with torch.no_grad():
        for index, category in enumerate(tqdm(train)):
            outputs = []
            for filename in category[1]['filename']:
                if category[0] in cat:#source == 'audioset':
                    track, _ = librosa.load(f'{input_dir}/AudioSet/train_wav/{filename}.wav', sr=16000, dtype=np.float32)
                else:
                    track, _ = librosa.load(f'{input_dir}/{category[0]}/_chunked/{filename}', sr=16000, dtype=np.float32)
                input_tensor = torch.from_numpy(track)
                f = feature_extractor(input_tensor, sampling_rate=16000, return_tensors="pt")
                output = model(**f).logits
                x_j.append(output)
                y_j.append(index)
                outputs.append(output)
                torch.cuda.empty_cache()
            outputs = torch.stack(outputs)
            outputs = torch.reshape(outputs, (outputs.shape[0], -1))
            outputs = torch.mean(outputs, 0)
            outputs = nn.functional.normalize(outputs, dim=0)
            mu[category[0]] = outputs

        trainset = Dataset(torch.stack(x_j), y_j)
        for category in tqdm(test):
            for filename in category[1]['filename']:
                if category[0] in cat:
                    track, _ = librosa.load(f'{input_dir}/AudioSet/train_wav/{filename}.wav', sr=16000, dtype=np.float32)
                else:
                    track, _ = librosa.load(f'{input_dir}/{category[0]}/_chunked/{filename}', sr=16000, dtype=np.float32)
                input_tensor = torch.from_numpy(track)
                f = feature_extractor(input_tensor, sampling_rate=16000, return_tensors="pt")
                output = model(**f).logits
                output = torch.flatten(output)
                output = nn.functional.normalize(output, dim=0)
                torch.cuda.empty_cache()
                q[category[0]].append(output)
    index = {}

    for idx, cat in enumerate(mu):
        index[idx] = cat

    from torch import nn
    import torch
    from loss import MultiBinaryCrossentropy

    w_init = torch.stack([mu[key] for key in mu.keys()])
    softmax = nn.Softmax()
    few_shot = few_shot_finetuning_layer(w_init)

    optimizer = torch.optim.SGD(few_shot.parameters(), lr=0.00005)
    #loss_function = nn.CrossEntropyLoss()
    loss_function = MultiBinaryCrossentropy(14)
    num_epoch = 100

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
    ####Test#####
    gt = []
    predicted = []

    for cat_q in q:
        for item in q[cat_q]:
            output = []  
            for idx, cat_m in enumerate(mu):
                output.append(torch.dot(mu[cat_m],item))

            output = torch.stack(output)
            result = torch.argmax(softmax(output)).item()

            gt.append(cat_q)
            predicted.append(index[result])

    df = pd.DataFrame()
    df["GT"] = gt
    df["predicted"] = predicted

    acc = len(df[df["GT"] == df["predicted"]])/len(df)

    print(f"------------------test before training: {acc}")

    for epoch in range(0, num_epoch):
        running_loss = 0.0
        running_corrects = 0
        for data in trainloader:
            inputs = data['encoding']
            targets = data['category']

            inputs = torch.Tensor(inputs)
            targets = torch.Tensor(targets)

            optimizer.zero_grad()

            outputs = []
            for input_tensor in inputs:
                input_tensor = input_tensor.reshape(input_tensor.shape[1], )
                output = few_shot(input_tensor)

                outputs.append(softmax(torch.Tensor(output)))

            loss = loss_function(torch.stack(outputs), targets.to('cuda'))
            loss.requires_grad= True

            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()*inputs.size(0)

        epoch_loss = running_loss / len(trainloader.dataset)
        print('Train Loss: {:.4f}'.format(epoch_loss))


        ####Test#####
        gt = []
        predicted = []

        for cat_q in q:
            for item in q[cat_q]:
                output = []  
                for idx, cat_m in enumerate(mu):
                    output.append(few_shot(item))

                output = torch.stack(output)
                result = torch.argmax(softmax(output)).item()

                gt.append(cat_q)
                predicted.append(index[result])

        df = pd.DataFrame()
        df["GT"] = gt
        df["predicted"] = predicted

        acc = len(df[df["GT"] == df["predicted"]])/len(df)   
        print(f"------------------test after epoch {epoch}: {acc}")