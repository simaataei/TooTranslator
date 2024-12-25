from model import ProteinTester,  ProteinClassifier, ProteinDescriptionClassifier, Label_encoder
from transformers import BertTokenizer, BertModel
import gc
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json
from torch.nn import functional as F
import time
from peft import LoraConfig, TaskType
from peft import get_peft_model
from transformers import BertTokenizer,BertForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification, AdamW, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import datasets
from datasets import load_dataset
import re
from torch.utils.data import Dataset
import csv
from sklearn.metrics import classification_report
from Bio import SeqIO
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize



def cosine_distance(a, b):
    # Cosine similarity is between 0 and 1, but we need a distance metric
    # So, we return 1 - cosine similarity
    return 1 - cosine_similarity(a, b)

class ProteinSeqDataset(Dataset):
    def __init__(self, filepath, max_length=1024):
        self.dataframe = pd.read_csv(filepath)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        #idx = int(idx)
        assert isinstance(idx, int), f"Index must be an integer, got {type(idx)}"
        text = self.dataframe.iloc[idx, 0]  # Assuming text is the first column
        label = self.dataframe.iloc[idx, 1]  # Assuming label is the second column
        return {
            'seq':text,  # Remove batch dimension
            'labels': label
        }
device = torch.device("cuda")
# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
best_mcc =-1


train_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
#val_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
test_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_test_f2.csv')
file_name  ='spec'


model_label = Label_encoder().cuda()
with open('./Dataset/chebi_des_up100_t3.txt') as f:
      data = f.read()
      term_description = json.loads(data)
      print(type(list(term_description.keys())[0]))
      class_description = list(term_description.values())




#load model

model = ProteinTester().cuda()
model_path = './Prot_bert_bfd_e100_spec_up100_sum_des_smiles'
model.load_state_dict(torch.load(model_path))

#testing with KNN
# get the train embedingsi
train_emb = []
val_emb = []
y_train =[]
y_val =[]
for item in train_dataset:
#for i in range(10):
 #   item = train_dataset[i]  
    print(item)
    with torch.no_grad():
         outputs = model(item['seq'])
         train_emb.append(outputs.detach().cpu().numpy()) 
         y_train.append(item['labels'])



for item in test_dataset:
#for i in range(10):
#	item = val_dataset[i]
	with torch.no_grad():
             outputs = model(item['seq'])
             val_emb.append(outputs.detach().cpu().numpy())
             y_val.append(item['labels'])


	# Manually invoke garbage collection in critical places
	if gc.isenabled():
	    gc.collect()
	torch.cuda.empty_cache()

del model


y_train_desc = []

for i in y_train:
    y_train_desc.append(class_description[i])

label_encoding_train= []
model_label.eval()
with torch.no_grad():
  for item in y_train_desc:
    label_encoding_train.append(model_label(item).cpu().numpy())





y_test_desc = []

for i in y_val:
    y_test_desc.append(class_description[i])

label_encoding_test= []
model_label.eval()
with torch.no_grad():
  for item in y_test_desc:
    label_encoding_test.append(model_label(item).cpu().numpy())


del model_label






# test val set with knn
knn = KNeighborsClassifier(n_neighbors=1)


scaler = StandardScaler()
train_emb = scaler.fit_transform(train_emb)
val_emb = scaler.transform(val_emb)


# Train the classifier
knn.fit(train_emb, label_encoding_train)
predictions = knn.predict(val_emb)
#mcc = matthews_corrcoef(y_val, predictions)
#print('mcc'+str(mcc))
#report = classification_report(y_val, predictions, output_dict=True)
#df = pd.DataFrame(report)
#latex_code = df.to_latex(index=False, caption='Classification Report', label='tab:prediction_table', longtable=True)
#with open('spec_classification_report_offline_e97_latex.tex', 'w') as file:
#   file.write(latex_code)




print('predictions')
print(predictions)
print(classification_report(label_encoding_test, predictions))
label_name = {}
label_chebi ={}
with open('./Dataset/Label_name_list_transporter_uni_ident100_t3') as f:
    data = f.readlines()
    for d in data:
        d = d.split(',')
        label_name[int(d[2].strip('\n'))] = d[1]
        label_chebi[int(d[2].strip('\n'))] = d[0]

seq_name = [i for i in range(0,len(val_emb))]
with open(f'Results/spec_resutls_offline_{model_name}_{file_name}.txt', 'w') as file:
    file.write("Sample, Predicted Label , Predicted Label ChEBI ID, Predicted Label Name\n")
    # Loop through all predictions and corresponding sequence names
    for name, label in zip(seq_name, predictions):
        # Fetch the label name and ChEBI ID from the dictionaries
        label_name_list = label_name.get(label, "Unknown Label Name")
        chebi_id_list = label_chebi.get(label, "Unknown ChEBI ID")

        # Write to file
        file.write(f"{name}, {label}, {chebi_id_list}, {label_name_list}\n")

df = pd.DataFrame({
    'Seq Name': seq_name,
    'Predicted Label': predictions,
    'Predicted Label ChEBI ID': chebi_id_list,
    'Predicted Label Name': label_name_list
})
# Convert the DataFrame to LaTeX code
latex_code = df.to_latex(index=False, caption='Predicted labels and their corresponding ChEBI IDs and names', label='tab:prediction_table', longtable=True)

with open(f'{file_name}_resutls_offline_{model_name}_latex.tex', 'w') as file:
    file.write(latex_code)



'''

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_emb, y_train)
label_dict = {}
with open('./Dataset/Label_name_list_transporter_uni_ident100_t3') as f:
    data = f.readlines()
    for d in data:
        d = d.split(',')
        label_dict[d[2].strip('\n')] = d[1]
label_dict = {int(k):v for k,v in label_dict.items()}



# Retrieve the 10 closest neighbors and their distances for the validation set
distances, indices = knn.kneighbors(val_emb)
print(indices)
nei_name= [
    [label_dict[y_train[number]] for number in row]
    for row in indices
]
#print("MCC:", mcc)
print("Distances:\n", distances)

print("Names of Neighbors:\n",nei_name )
name_distance = [
    [(nei_name[i][j], distances[i][j]) for j in range(len(nei_name[i]))]
    for i in range(len(distances))
]
print("Name-Distances:\n", name_distance)
df = pd.DataFrame({'neighbours':name_distance})
with open(f'./Results/{file_name}_offline_sampling_up_100_{model_name}_3nn.txt', 'w') as f:
        with pd.option_context("max_colwidth", 1000):
            f.write(df.to_latex(index=True))

'''
