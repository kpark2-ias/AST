import pandas as pd
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from pytorchtools import EarlyStopping
from transformers import AutoFeatureExtractor, ASTForAudioClassification, Trainer, AutoModelForAudioClassification
import pdb
import librosa
import numpy as np
import sklearn.metrics

from collections import defaultdict
from tqdm import tqdm

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings_1, encodings_2, labels=None):
        self.encodings_1 = encodings_1
        self.encodings_2 = encodings_2
        self.labels = labels

    def __getitem__(self, idx):
        
        encoding_1= torch.tensor(self.encodings_1[idx])
        encoding_2= torch.tensor(self.encodings_2[idx])
        label = self.labels[idx]
        item = {'input1': encoding_1, 'input2': encoding_2, 'label': label}
        return item

    def __len__(self):
        return len(self.encodings_1)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(527, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input1, input2):
        x = torch.abs(input1-input2)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x

def preprocess_input(data, model_name):
    
#     train = pd.read_csv("few-shot_train.csv")
#     test = pd.read_csv("few-shot_test.csv")
#     print("test_len : ", len(test))
#     group = train.groupby('category')
#     train = pd.DataFrame(group.head(shot)).groupby('category')
#     test = test.groupby('category')
    
        ##support set
        
    if model_name == 'BEATs':
        model_filename = 'fine-tuned-cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
        checkpoint = torch.load(f'models/{model_filename}')
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
    elif model_name == 'AST':
        feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    elif model_name == 'Distil-AST':
        feature_extractor = AutoFeatureExtractor.from_pretrained("bookbot/distil-ast-audioset")
        model = AutoModelForAudioClassification.from_pretrained("bookbot/distil-ast-audioset")

    model.eval()
    
    encodings1 = []
    encodings2 = []
    y = []

    with torch.no_grad():
        for index, row in tqdm(data.iterrows()):
            input1 = row['f1']
            input2 = row['f2']
        
            track1, _ = librosa.load(f'{input_dir}/AudioSet/train_wav/{input1}.wav', sr=16000, dtype=np.float32)
            track2, _ = librosa.load(f'{input_dir}/AudioSet/train_wav/{input2}.wav', sr=16000, dtype=np.float32)
            
            input_tensor1 = torch.from_numpy(track1)
            input_tensor2 = torch.from_numpy(track2)

            if model_name == 'BEATs':
                input_tensor1 = input_tensor1.reshape(1, input_tensor1.shape[0])
                padding_mask1 = torch.zeros(len(input_tensor1), input_tensor1.shape[1]).bool().to('cuda')
                output1 = model.extract_features(input_tensor1, padding_mask1)[0]

                input_tensor2 = input_tensor2.reshape(1, input_tensor2.shape[0])
                padding_mask2 = torch.zeros(len(input_tensor2), input_tensor2.shape[1]).bool().to('cuda')
                output2 = model.extract_features(input_tensor2, padding_mask2)[0]

            elif 'AST' in model_name:
                f1 = feature_extractor(input_tensor1, sampling_rate=16000, return_tensors="pt")
                output1 = model(**f1).logits

                f2 = feature_extractor(input_tensor2, sampling_rate=16000, return_tensors="pt")
                output2 = model(**f2).logits

            encodings1.append(output1)
            encodings2.append(output2)
            y.append(row['label'])
            torch.cuda.empty_cache()
                
        trainset = Dataset(torch.stack(encodings1), torch.stack(encodings2), y)
    
    return trainset

def compute_metrics(df):
    
    import sklearn.metrics
    df_concated = pd.DataFrame({'category': category})
    precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(df["GT"], df["predicted"])
    acc = len(df[df["GT"] == df["predicted"]])/len(df)
    
    df2 = pd.DataFrame({'model':model_name, 'shot':shot, 'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'support':support})
    
    df_concated = pd.concat([df_concated, df2], axis=1)
    
    df_concated.to_csv(f"results/finetuning_{finetuning}/{model_name}_{shot}shot.csv", index=False)

def inference(dataset, batch_size):

    # Set fixed random number seed

    #### Validate the model###
    
    dataloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size)
    
    gt = []
    predicted_prob = []
    predicted = []
    model = SiameseNetwork()
    model.load_state_dict(torch.load('model-fold-0.pth'))
    model.eval()
    
    with torch.no_grad():

        # Iterate over the test data and generate predictions
        for i, data in enumerate(dataloader, 0):

            # Get inputs
            inputs1 = data['input1']
            inputs2 = data['input2']
            targets = data['label']

            inputs1 = torch.tensor(inputs1)
            inputs2 = torch.tensor(inputs2)

            targets = torch.tensor(targets).type(torch.FloatTensor).to('cuda')
            
            model.eval()
            outputs = model(inputs1, inputs2)
            outputs = torch.flatten(outputs)
            preds = torch.where(outputs > 0.5, 1, 0)
            
            gt.extend(targets.tolist())
            predicted_prob.extend(outputs.tolist())
            predicted.extend(preds.tolist())

        
    df = pd.DataFrame()
    df["GT"] = gt
    df["predicted"] = predicted
    df["prob"] = predicted_prob

    precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(df["GT"], df["predicted"])
    acc = len(df[df["GT"] == df["predicted"]])/len(df)
    
    print(acc, precision, recall, f1, support)
    df.to_csv("results/results_siamese_infrence_audioset.csv", index=False)

def main(model_name, labels, input_dir, batch_size):
    
    data = pd.read_csv(labels)[:30000]
    data = data.sample(frac=1).reset_index(drop=True)
  
    dataset = preprocess_input(data, model_name)
    
    inference(dataset, batch_size)

model_name = 'AST'
labels = '/home/ubuntu/data/_ipynb/siamese_test_200k.csv'
input_dir = '/home/ubuntu/data/wav'
batch_size = 32

main(model_name, labels, input_dir, batch_size)

# if __name__ == "__main__": 
    
#     from argparse import ArgumentParser
#     import torch
#     from torch import nn
#     parser = ArgumentParser()
#     parser.add_argument('-i','--input_dir', default='/home/ubuntu/data/wav')
#     parser.add_argument('-l','--labels', default='/home/ubuntu/data/wav/siamese_train.csv')
#     #parser.add_argument('-n','--shot', default=5)
#     parser.add_argument('-b', '--batch_size', default = 32)
#     #parser.add_argument('-f','--finetuning', default=None)
#     parser.add_argument('-e', '--num_epoch', type=int, default=30)
#     parser.add_argument('-m', '--model_name', default = 'Distil-AST')
#     args = parser.parse_args()
    
#     main(**vars(args))
#     for finetuning in [None, True]:
#         for i in range(1, 6):
#             for model_name in ["BEATs","AST", "Distil-AST"]:
#                 args.shot = i
#                 args.model_name = model_name
#                 args.finetuning = finetuning
#                 main(**vars(args))
      
