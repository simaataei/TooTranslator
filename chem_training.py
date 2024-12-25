
from torch import optim
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
from data_pre import X_train, X_val, y_train, y_val
from model import ProteinTester,  ProteinClassifier, ProteinDescriptionClassifier, Label_encoder, Chem_BERT_encoder
import json



file_path = 'Dataset/Label_name_list_transporter_uni_ident100_t3'  # Replace with the path to your file
label_chebi = {}

with open(file_path, 'r') as file:
    for line in file:
        # Split the line by comma
        parts = line.strip().split(',')
        if len(parts) == 3:  # Ensure the line has exactly 3 parts
            key = int(parts[2])
            value = parts[0]
            label_chebi[key] = value
#print(label_chebi)
model_label = Chem_BERT_encoder().cuda()
with open('./Dataset/chebi_smiles_up100_t3.txt') as f:
      data = f.read()
      term_smiles = json.loads(data)
      print(type(list(term_smiles.keys())[0]))
      class_smiles = list(term_smiles.values())

print(term_smiles)
print(term_description)
y_train_smiles = []

for i in y_train:
#    print(label_chebi[i])
 #   print(term_smiles[label_chebi[i]])
    y_train_smiles.append(term_smiles[label_chebi[i]])

label_encoding= {}
for i in term_smiles.keys():
    label_encoding[i] = []
model_label.eval()
with torch.no_grad():
  for k,v in term_smiles.items():
      for item in v:
          label_encoding[k].append(model_label(item))
del model_label
for item in label_encoding.keys():
    if len(label_encoding[item])==0:
       print(item)
#print(label_encoding)
mean_labels = {}
for item in label_encoding.keys():
    #print(item)
    #print(label_encoding[item])
    mean_labels[item] = torch.mean(torch.stack(label_encoding[item]), dim =0)


     
'''
model = ProteinTester().cuda()
#model = ProteinClassifier(768).cuda()

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.000005)
num_epochs = 100


pred_scores = []
gold_scores = []


all_loss_val = []
all_mcc = []
all_f1 = []
all_acc = []
all_rec = []
all_pre = []
loss_min = 100
for epoch in range(1,num_epochs+1):
    all_loss=list()
    start = time.time()
    for i in tqdm(range(len(X_train))):
        optimizer.zero_grad()
        #gold = label_encoding[int(y_train[i])] 
        sample = X_train[i]
        description = y_train_smiles[i]
        print(sample)
        pred = model(sample)
        print(len(pred))
       #print(len(gold))
        print(label_encoding[int(y_train[i])])
        gold = label_encoding[int(y_train[i])]
        loss = loss_function(pred, gold)
        loss.backward()
        all_loss.append(loss.cpu().detach().numpy())
        optimizer.step()
    torch.save(model.state_dict(), "./Prot_bert_bfd_up100_e10_spec_up100")
''''''
    with torch.no_grad():
        model.eval()

        all_gold=list()
        all_pred=list()
        optimizer.zero_grad()
        for j in range(len(X_val)):
            pred = model(X_val[j])
            all_gold.append(y_val[j])
            gold = torch.tensor([y_val[j]],dtype=torch.long).cuda()
            loss = loss_function(pred,gold)
            all_loss_val.append(loss.cpu().detach().numpy())
            prediction = np.argmax(pred.cpu().detach().numpy())
            all_pred.append(prediction)
            if epoch == num_epochs:
                pred_scores.append(pred.cpu().detach().numpy())
                gold_scores.append(gold.cpu().detach().numpy())


    all_mcc.append(matthews_corrcoef(all_gold,all_pred))
    all_rec.append(sklearn.metrics.recall_score(all_gold,all_pred, average = 'macro'))
    all_pre.append(sklearn.metrics.precision_score(all_gold,all_pred, average = 'macro'))
    all_f1.append(sklearn.metrics.f1_score(all_gold,all_pred, average = 'macro'))
    all_acc.append(sklearn.metrics.accuracy_score(all_gold,all_pred))'''
    #torch.save(model.state_dict(), "./Prot_bert_bfd_inorganic_up100")


