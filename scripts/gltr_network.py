import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from gltr import LM, BERTLM

class GLTR_classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(2, 8, 7)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, 5)
        self.bn2 = nn.BatchNorm1d(16)
        self.mp1 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(16, 32, 5)
        self.mp2 = nn.MaxPool1d(2)
        self.fc = nn.Linear(8032,1)

    def forward(self, input_batch):
        x = self.conv1(input_batch)
        x = F.dropout(F.relu(self.bn1(x)),p=0.2)
        x = self.conv2(x)
        x = F.dropout(F.relu(self.bn2(x)),p=0.2)
        x = self.mp1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.mp2(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

training_set = pd.read_json('drive/MyDrive/data582/train_set.json')[["summary", "label"]]
class trainset_GLTR(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.lm = LM()
        #get source and target texts
        self.summaries = self.df["summary"]
        self.labels = self.df["label"]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        summary = self.summaries[index]
        label = self.labels[index]
        lm_output = self.lm.check_probabilities(summary,topk=0)
        nn_input = lm_output["real_topk"]
        nn_input_padded = nn_input + [(-1,-1) for i in range(1024-len(nn_input))]
        return torch.transpose(torch.tensor(nn_input_padded),0,1), torch.tensor(label,dtype=torch.float)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tset = trainset_GLTR(training_set)
train_loader = torch.utils.data.DataLoader(tset, batch_size=32, shuffle=True)
loss_fn = nn.BCEWithLogitsLoss()
model = GLTR_classifier().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
for input, label in train_loader:
    output = model(input.to(device))
    loss = loss_fn(output, label.unsqueeze(1).to(device))
    print("loss on training set="+str(loss.detach().cpu().item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()