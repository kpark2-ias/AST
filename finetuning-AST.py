import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from torch.utils.data import DataLoader#, Dataset
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary

import glob
import os
import librosa
import pdb

from BEATs import BEATs, BEATsConfig
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def labels_to_num_esc50():
    esc50_labels = ['dog', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm',
       'door_wood_knock', 'can_opening', 'crow', 'clapping', 'fireworks',
       'chainsaw', 'airplane', 'mouse_click', 'pouring_water', 'train',
       'sheep', 'water_drops', 'church_bells', 'clock_alarm',
       'keyboard_typing', 'wind', 'footsteps', 'frog', 'cow',
       'brushing_teeth', 'car_horn', 'crackling_fire', 'helicopter',
       'drinking_sipping', 'rain', 'insects', 'laughing', 'hen', 'engine',
       'breathing', 'crying_baby', 'hand_saw', 'coughing',
       'glass_breaking', 'snoring', 'toilet_flush', 'pig',
       'washing_machine', 'clock_tick', 'sneezing', 'rooster',
       'sea_waves', 'siren', 'cat', 'door_wood_creaks', 'crickets']

    category = {}

    for i in range(50):
        category[esc50_labels[i]] = i

    return category

def preprocess_audio(input_dir,  files, SAMPLE_RATE = 16000, verbose=False, batch_size = 16):
                     
        #paths_to_audio = glob.glob(f'{input_dir}/*.wav')

        tracks = list()
                     
        for idx, file in enumerate(tqdm(files)):
            
            track, _ = librosa.load(f'{input_dir}/{file}', sr=SAMPLE_RATE, dtype=np.float32)
            track = torch.from_numpy(track)

            tracks.append(track)
            
            if not idx % batch_size:

                if verbose:
                    print([track.shape for track in tracks])
                    
                #maxtrack =  max([ta.shape[-1] for ta in tracks])
                maxtrack = 160125

                padded = [torch.nn.functional.pad(torch.from_numpy(np.array(ta)),(0,maxtrack-ta.shape[-1])) for ta in tracks]
                if verbose:
                    print( [track.shape for track in padded])


                audio = torch.stack(padded)

                if verbose:
                    print(audio.shape)
                
                yield audio 
                    
                tracks = list()

# Read data

data = pd.read_csv("/home/ubuntu/data/ESC-50-master/meta/esc50.csv")
audio_dir = '/home/ubuntu/data/ESC-50-master/audio/'

# ----- 1. Preprocess data -----#
# Preprocess data
X = list(data["filename"])
y = list(data["category"])

features = []

for audio in preprocess_audio(audio_dir, X):
    features.extend(torch.Tensor(audio))
#     padding_mask = torch.zeros(len(audio), audio.shape[1]).bool().to('cuda')
#     feature = beats.extract_features(audio, padding_mask=padding_mask)[0]

    torch.cuda.empty_cache()
    #pdb.set_trace()
#features = torch.stack(features)


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        
        encoding = torch.tensor(self.encodings[idx])

#         label = torch.tensor(self.labels_to_num(self.labels[idx]))
#         label = nn.functional.one_hot(label, num_classes = 50)
        
        label = self.labels_to_num(self.labels[idx])
        #print(label)
        
        item = {'encoding': encoding, 'category': label}
        return item

    def __len__(self):
        return len(self.encodings)
    
    def labels_to_num(self, cat):
        esc50_labels = ['dog', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm',
           'door_wood_knock', 'can_opening', 'crow', 'clapping', 'fireworks',
           'chainsaw', 'airplane', 'mouse_click', 'pouring_water', 'train',
           'sheep', 'water_drops', 'church_bells', 'clock_alarm',
           'keyboard_typing', 'wind', 'footsteps', 'frog', 'cow',
           'brushing_teeth', 'car_horn', 'crackling_fire', 'helicopter',
           'drinking_sipping', 'rain', 'insects', 'laughing', 'hen', 'engine',
           'breathing', 'crying_baby', 'hand_saw', 'coughing',
           'glass_breaking', 'snoring', 'toilet_flush', 'pig',
           'washing_machine', 'clock_tick', 'sneezing', 'rooster',
           'sea_waves', 'siren', 'cat', 'door_wood_creaks', 'crickets']
        #esc50_labels = ['clean', 'protest', 'smoke']

        labels_to_num = {}

        for i in range(len(esc50_labels)):
            labels_to_num[esc50_labels[i]] = i

        return labels_to_num[cat]

dataset = Dataset(features, y)

import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from transformers import AutoFeatureExtractor, ASTForAudioClassification, Trainer, AutoModelForAudioClassification
import pdb

#if __name__ == '__main__':
  
# Configuration options
k_folds = 5
num_epochs = 30
loss_function = nn.CrossEntropyLoss()

# For fold results
results = {}

# Set fixed random number seed
torch.manual_seed(42)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

# Start print
print('--------------------------------')

# K-fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=8, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=8, sampler=test_subsampler)

    # Init the neural network
    feature_extractor = AutoFeatureExtractor.from_pretrained("xpariz10/ast-finetuned-audioset-10-10-0.4593-finetuning-ESC-50")
    model = ASTForAudioClassification.from_pretrained("xpariz10/ast-finetuned-audioset-10-10-0.4593-finetuning-ESC-50")#("MIT/ast-finetuned-audioset-10-10-0.4593")
#     feature_extractor = AutoFeatureExtractor.from_pretrained("bookbot/distil-ast-audioset")
#     model = AutoModelForAudioClassification.from_pretrained("bookbot/distil-ast-audioset")

    for name, param in model.named_parameters():
        if 'classifier' not in name:   
            param.requires_grad = False
        else:
            param.requires_grad=True

    model.classifier = nn.Sequential(nn.LayerNorm((768, ), eps=1e-12), 
                                     nn.Linear(768, 3))

    model.train()

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):

        running_loss = 0.0
        running_corrects = 0

        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for data in trainloader:

            # Get inputs
            #inputs, targets = data
            inputs = data['encoding']
            targets = data['category']

            inputs = torch.tensor(inputs)
            targets = torch.tensor(targets)


            optimizer.zero_grad()

            outputs = []
            for input_tensor in inputs:
                f = feature_extractor(input_tensor, sampling_rate=16000, return_tensors="pt")
                output = model(**f).logits
                outputs.append(output)

            outputs = torch.stack(outputs)
            outputs = torch.reshape(outputs, (outputs.shape[0], -1))

            # Compute loss
            loss = loss_function(outputs, targets)

            _, preds = torch.max(outputs, 1)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            running_loss += loss.item()*inputs.size(0)

            #pdb.set_trace()
            running_corrects += torch.sum(preds==targets)

        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = running_corrects.double() / len(trainloader.dataset)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')

    # Saving the model
    save_path = f'./finetuned_models/distil-ast/model-fold-{fold}.pth'
    torch.save(model.state_dict(), save_path)

    # Evaluationfor this fold
    correct, total = 0, 0
    with torch.no_grad():

        # Iterate over the test data and generate predictions
        for i, data in enumerate(testloader, 0):

            # Get inputs
            inputs = data['encoding']
            targets = data['category']
            model.eval()

            # Generate outputs

            outputs = []
            for input_tensor in inputs:
                f = feature_extractor(input_tensor, sampling_rate=16000, return_tensors="pt")
                output = model(**f).logits
                outputs.append(output)

            outputs = torch.stack(outputs)

            outputs = torch.reshape(outputs, (outputs.shape[0], -1))

            # Set total and correct
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)

            correct += (predicted == targets).sum().item()

        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)

# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
    print(f'Average: {sum/len(results.items())} %')