from sklearn.metrics import classification_report
from torch import optim
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
from data_pre import X_train, X_val, y_train, y_val
from model import ProteinTester,  ProteinClassifier, ProteinDescriptionClassifier, Label_encoder, Chem_BERTa_encoder, ProteinTesterAttention, ProteinTesterResidual, ProteinTesteriAttentionResidual
import json
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def weighted_mse_loss(y_pred, y_true, weights):
    return torch.mean(weights * (y_pred - y_true) ** 2)

def mse_distance(a, b):
    return np.mean((a - b) ** 2)

# Function to find the closest vector using MSE
def find_closest_vector(target_vector, vector_dict):
    min_distance = float('inf')
    closest_vector = None
    print('target '+ str(type(target_vector[0])))
    for key, vec in vector_dict.items():
        print('value ' +str(type(np.array(vec))))
        print('key '+str(key))
        distance = mse_distance(target_vector[0], np.array(vec))
        #distance = cdist(target_vector[0].reshape(1, -1), np.array(vec).reshape(1, -1), metric='cityblock')
        print('distance'+str(distance))
        if distance < min_distance:
            min_distance = distance
            closest_vector = key
        print(closest_vector)
    return closest_vector, min_distance


def cosine_distance(vec1, vec2):
    # Cosine similarity is between 0 and 1, but we need a distance metric
    # So, we return 1 - cosine similarity
    cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return 1 - cosine_similarity

model_label = Label_encoder().cuda()
with open('./Dataset/ICAT/chebi_des_ICAT.txt') as f:
      data = f.read()
      term_description = json.loads(data)
    #  print(type(list(term_description.keys())[0]))
      class_description = list(term_description.values())


#print(term_description)
y_train_desc = []

for i in y_train:
    y_train_desc.append(class_description[i])

label_encoding_des= []
model_label.eval()
with torch.no_grad():
   for item in class_description:
  #     print(item)
       label_encoding_des.append(model_label(item))
del model_label


model_label = Chem_BERTa_encoder().cuda()
with open('./Dataset/ICAT/chebi_smiles_ICAT.txt') as f:
      data = f.read()
      term_smiles = json.loads(data)
   #   print(type(list(term_smiles.keys())[0]))
      class_smiles = list(term_smiles.values())


#print(term_description)
y_train_smiles = []

for i in y_train:
    y_train_smiles.append(class_smiles[i])

label_encoding_smiles= []
model_label.eval()
with torch.no_grad():
   for item in class_smiles:
 #      print(item)
       if len(item) > 0:
            encs = []
            for i in item:
                encs.append(model_label(i.strip('"')))
            stacked_embeddings = torch.stack(encs)
            label_encoding_smiles.append(torch.mean(stacked_embeddings, dim=0))
       else:
            # Append a dummy SMILES code for padding
            dummy_smiles = "XXX"
            label_encoding_smiles.append(model_label(dummy_smiles))
del model_label


#label_encoding = label_encoding_smiles
#label_encoding = [torch.cat((t1,t2)) for t1, t2 in zip(label_encoding_des, label_encoding_smiles) ]
#label_encoding = [t1+t2 for t1, t2 in zip(label_encoding_des, label_encoding_smiles) ]
label_encoding = label_encoding_des
label_encoding_dict = {i: label_encoding[i].detach().cpu() for i in range(len(label_encoding))}


class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

#print(len(label_encoding))
#print(len(label_encoding[0]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = ProteinTester().to(device)

#model = ProteinTesterAttentionResidual().to(device)
model = ProteinTesterResidual().to(device)
#model_path = 'Prot_bert_bfd_e90_lr5e-06_spec_up100_sum_des_smiles_res'
#model_name = model_path.strip('./')
#model.load_state_dict(torch.load(model_path))

#loss_function = nn.MSELoss()
#loss_function = nn.SmoothL1Loss()
#loss_function = nn.L1Loss()

#loss_function = weighted_mse_loss()
l='wMSE'
name = 'des'
optimizer = optim.AdamW(model.parameters(), lr=0.000005)
num_epochs =101
lr = str(0.000005)


pred_scores = []
gold_scores = []


all_loss_val = []
all_mcc = []
all_f1 = []
all_acc = []
all_rec = []
all_pre = []
#batch_size = 100

for epoch in range(0,num_epochs+1):
    model.train()
    all_loss=list()
    start = time.time()
    optimizer.zero_grad()
    for i in tqdm(range(len(X_train))):
        optimizer.zero_grad()
        #gold = label_encoding[int(y_train[i])] 
        sample = X_train[i]
     #   description = y_train_desc[i]
     #   print(sample)
        pred = model(sample)
      #  print(len(pred))
       #print(len(gold))
 #       print(label_encoding[int(y_train[i])])
        gold = label_encoding[int(y_train[i])]
        loss = weighted_mse_loss(pred, gold, class_weights[y_train[i]])
       # loss = loss_function(pred,gold)
        print(loss)
        loss.backward()
        all_loss.append(loss.cpu().detach().numpy())
        optimizer.step()
#        if (i + 1) % batch_size == 0 or (i + 1) == len(X_train):
 #           optimizer.step()  # Update the model parameters
  #          optimizer.zero_grad()  # Reset gradients for the next batch
    if epoch % 10==0:
       torch.save(model.state_dict(), f"./Prot_bert_bfd_e{epoch}_lr{lr}_loss_{l}_ICAT_{name}_f1")



with torch.no_grad():
     model.eval()
     all_pred=list()
     optimizer.zero_grad()
     for i in tqdm(range(len(X_val))):
         pred = model(X_val[i])
         all_pred.append(pred.detach().cpu().numpy())

neighbor =[]
for item in all_pred:
    closest_vector, min_distance= find_closest_vector(item, label_encoding_dict)
    neighbor.append(closest_vector)
    #closest_vector, min_distance= find_closest_vector(item, label_encoding)
    #index = label_encoding.index(closest_vactor)
    #neigbor .append(y_train[index])
   # for key, val in number_encoding_dict.items():
 #       print('closest_vector')
  #      print(closest_vector)
   #     print('val')
    #    print(val)
   #     if (closest_vector == val):
    #       neighbor.append(int(key))
     #      continue
       # else:
        #   neigbor.append(1000)
print('Neighbor')
print(neighbor)
print('y val')
print(y_val)
print(classification_report(y_val,neighbor))

'''

    all_mcc.append(matthews_corrcoef(all_gold,all_pred))
    all_rec.append(sklearn.metrics.recall_score(all_gold,all_pred, average = 'macro'))
    all_pre.append(sklearn.metrics.precision_score(all_gold,all_pred, average = 'macro'))
    all_f1.append(sklearn.metrics.f1_score(all_gold,all_pred, average = 'macro'))
    all_acc.append(sklearn.metrics.accuracy_score(all_gold,all_pred))'''
    #torch.save(model.state_dict(), "./Prot_bert_bfd_inorganic_up100")


