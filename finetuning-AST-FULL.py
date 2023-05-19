import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from torch.utils.data import DataLoader
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

data_type = 'realworld'
print(data_type)
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
# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        
        encoding = torch.tensor(self.encodings[idx])

        label = self.labels_to_num(self.labels[idx])

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
           'sea_waves', 'siren', 'cat', 'door_wood_creaks', 'crickets', 'protest', 'smoke']
        #esc50_labels = ['clean', 'protest', 'smoke']

        labels_to_num = {}

        for i in range(len(esc50_labels)):
            labels_to_num[esc50_labels[i]] = i

        return labels_to_num[cat]
    
# Read data
data = pd.read_csv("/home/ubuntu/data/finetuning/labels/finetuning_{}_train_esc50_categories.csv".format(data_type))
data = data.sample(frac=1).reset_index(drop=True)

audio_dir = '/home/ubuntu/data/finetuning/audio'

holdout = "/home/ubuntu/data/finetuning/labels/finetuning_holdout_test_esc50_categories.csv"
holdout = pd.read_csv(holdout)
holdout = holdout.sample(frac=1).reset_index(drop=True)

# ----- 1. Preprocess data -----#
# Preprocess data
X = list(data["filename"])
y = list(data["category"])

features = []

for audio in preprocess_audio(audio_dir, X):
    features.extend(torch.Tensor(audio))
    torch.cuda.empty_cache()

dataset = Dataset(features, y)

####Test data#####

# Preprocess data
X = list(holdout["filename"])
y = list(holdout["category"])

features = []

for audio in preprocess_audio(audio_dir, X):
    features.extend(torch.Tensor(audio))
    torch.cuda.empty_cache()

testset = Dataset(features, y)

import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from transformers import AutoFeatureExtractor, ASTForAudioClassification, Trainer
import pdb

#if __name__ == '__main__':
  
# Configuration options
k_folds = 5
num_epochs = 100
loss_function = nn.CrossEntropyLoss()

batch_size = 16
# For fold results
results = {}

# Set fixed random number seed
torch.manual_seed(42)

# Start print
print('--------------------------------')

# K-fold Cross Validation model evaluation

trainloader = torch.utils.data.DataLoader(
                  dataset, 
                  batch_size=batch_size)

# Init the neural network
feature_extractor = AutoFeatureExtractor.from_pretrained("xpariz10/ast-finetuned-audioset-10-10-0.4593-finetuning-ESC-50")
model = ASTForAudioClassification.from_pretrained("xpariz10/ast-finetuned-audioset-10-10-0.4593-finetuning-ESC-50")


for name, param in model.named_parameters():
    if 'classifier' not in name:   
        param.requires_grad = False
    else:
        param.requires_grad=True

model.classifier = nn.Sequential(nn.LayerNorm((768, ), eps=1e-12), 
                                 nn.Linear(768, 52))

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

    if (epoch+1) % 10 == 0:
        # Saving the model

                        # Process is complete.
        print('Saving trained model after epoch{}'.format(epoch+1))

        save_path = '/home/ubuntu/BEATs/beats/results/ast/esc50-{}/finetuned-FULL_{}_epoch{}.pth'.format(data_type,data_type, epoch+1)
        torch.save(model.state_dict(), save_path)


        # Print about testing
        print('Starting testing')


        # Define data loaders for training and testing data in this fold
        testloader = torch.utils.data.DataLoader(
                          testset, 
                          batch_size=batch_size)

        # Evaluationfor this fold
        correct, total = 0, 0

        gt = []
        predicted_categories = []
        confidence = []

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
                values, predicted = torch.max(outputs.data, 1)

                # save the predicted outputs & confidences
                gt.extend(targets.cpu().numpy())
                predicted_categories.extend(predicted.cpu().numpy())
                confidence.extend(values.cpu().numpy())
                
                total += targets.size(0)

                correct += (predicted == targets).sum().item()

            df = pd.DataFrame({'GT': gt, 'predicted': predicted_categories, 'confidence': confidence})
            df.to_csv("/home/ubuntu/BEATs/beats/results/ast/esc50-{}/test_results_{}_epoch{}.csv".format(data_type, data_type, epoch+1), index=False)

            # Print accuracy
            print('Test Accuracy: %d %%' % (100.0 * correct / total))