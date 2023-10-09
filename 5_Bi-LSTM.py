import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F 
#from torch.utils.data import DataLoader
#import torchvision.datasets as datasets
#import torchvision.transforms as transforms
from sklearn.metrics import classification_report
import numpy as np
#import torchmetrics
import pandas as pd
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# from scipy.stats import zscore


class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, \
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        # hidden state
        # h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        # cell state
        # c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        # x, _ = self.lstm(x, (h0, c0))
        x, _ = self.lstm(x)
        x = self.fc(x[:,-1,:])
        out = self.softmax(x)
        return out


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyper perameters
input_size = 14
hidden_size = 64
num_classes = 2
learning_rate = 0.0001
# batch_size = 512
num_epochs = 10
num_layers = 1

# Init Networks
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and Optimazier
loss_fc = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

with open('./project_status.json', 'r') as f:
    project_status = json.load(f)
    project_status = {s: set(project_status[s]) for s in project_status}

df = pd.read_csv('./network_data.csv')
data_dic = {}

for index, row in df.iterrows():
    #if row['proj_name'] in project_status['graduated']:
    proj = row['proj_name']
    if proj not in data_dic:
        data_dic[proj] = []
    data_dic[proj].append(row.tolist()[2:])

X = []
y = []
for proj in data_dic:
    if proj in project_status['graduated']:
        X.append(torch.tensor(data_dic[proj]))
        y.append(torch.tensor([1]))
    if proj in project_status['retired']:
        X.append(torch.tensor(data_dic[proj]))
        y.append(torch.tensor([0]))


X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                    test_size=0.33, random_state=48)

# training
for epoch in range(num_epochs):
    for data, target in tqdm(list(zip(X_train, y_train))):
        data = data.to(device)# .squeeze(1)
        data = data.reshape(1, data.shape[0], -1)
        target = target.to(device)
        # forward 
        pred = model(data)        
        # backward
        loss = loss_fc(pred, target)      
        optimizer.zero_grad()
        loss.backward()
        #gradiant descent or adam
        optimizer.step()

preds_list = np.empty(0)
targets_list = np.empty(0)

for data, target in zip(X_test, y_test):
    data = data.to(device)# .squeeze(1)
    data = data.reshape(1, data.shape[0], -1)
    targets = target.to(device)
    preds = torch.argmax(model(data), dim=1).to(device)
    preds_list = np.concatenate((preds_list, preds.cpu().detach().numpy()))
    targets_list = np.concatenate((targets_list, target.cpu().detach().numpy()))

preds_list.flatten()
targets_list.flatten()
print(classification_report(targets_list, preds_list, labels=list(range(2))))


'''

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, \
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):\
        # init hidden state
        # hidden state
        #h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        # cell state
        #c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        # out, _ = self.lstm(x, (h0, c0))
        
        out, _ = self.lstm(x)
        out = self.fc(out[:,-1,:])
        out = self.softmax(out)
        return out


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyper perameters
input_size = 28
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 512
num_epochs = 3
num_layers = 2

# Load data
train_dataset = datasets.MNIST(root='dataset/', train=True, \
                               transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train = False, \
                              transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)




# Init Networks
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and Optimazier
loss_fc = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# initialize metric
acc_f = torchmetrics.F1().to(device)
rec_f = torchmetrics.Recall().to(device)


# training
for epoch in range(num_epochs):
    for batch_index, (data, targets) in enumerate(train_loader):
        print(targets)
        print(data[0])
        raise KeyError
        data = data.to(device).squeeze(1)
        targets = targets.to(device)
        # get to correct shape
        # data = data.reshape(data.shape[0], -1)
        # for-ward 
        preds = model(data)
        acc = acc_f(preds, targets)        
        print(f"Accuracy on epoch {epoch} batch {batch_index}: {acc}")
        #rec = rec_f(preds, targets)
        #print(f"Recall on epoch {epoch} batch {batch_index}: {rec}")
        # back-ward
        loss = loss_fc(preds, targets)      
        optimizer.zero_grad()
        loss.backward()
        #gradiant descent or adam
        optimizer.step()

preds_list = np.empty(0)
targets_list = np.empty(0)

for batch_index, (data, targets) in enumerate(test_loader):
    data = data.to(device).squeeze(1)
    targets = targets.to(device)
    preds = torch.argmax(model(data), dim=1).to(device)
    preds_list = np.concatenate((preds_list, preds.cpu().detach().numpy()))
    targets_list = np.concatenate((targets_list, targets.cpu().detach().numpy()))

preds_list.flatten()
targets_list.flatten()

print(classification_report(targets_list, preds_list, labels=list(range(10))))

'''




'''

# Trains a Bidirectional LSTM on the IMDB sentiment classification task.
# Output after 4 epochs on CPU: ~0.8146
# Time per epoch on CPU (Core i7): ~150s.

from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb


max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=4,
          validation_data=[x_test, y_test])
'''


