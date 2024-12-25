from model import ProteinTester,  ProteinClassifier, ProteinDescriptionClassifier, Label_encoder, Chem_BERTa_encoder,SimpleNN, ProteinTesterAttention, ProteinTesterResidual,ProteinTesteriAttentionResidual
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
from model import ProteinTester
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE





def create_number_to_encoding_dict(file_path, result_dict):
    # Initialize the new dictionary
    indexed_dict = {}

    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line by comma
            parts = line.strip().split(',')
            chebi_term = parts[0]
            index = int(parts[2])

            # Assign the value from result_dict to the new dictionary using the index as the key
            if chebi_term in result_dict:
                indexed_dict[index] = result_dict[chebi_term]

    return indexed_dict

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





model = ProteinTesterResidual().cuda()
model_path = 'Prot_bert_bfd_FR'
#model.load_state_dict(torch.load(model_path))
model.to(device)
model_name = model_path.strip('./')
train_emb = []
val_emb = []
y_train =[]
y_val =[]
for item in test_dataset:
#for i in range(10):
#        item = test_dataset[i]
        with torch.no_grad():
             outputs = model(item['seq'])
             val_emb.append(outputs.detach().cpu().numpy())
             print(outputs)
             y_val.append(int(item['labels']))
             print(int(item['labels']))

        # Manually invoke garbage collection in critical places
        if gc.isenabled():
            gc.collect()
        torch.cuda.empty_cache()

del model
print(len(val_emb))


model_label = Label_encoder().cuda()
with open('./Dataset/chebi_des_up100_t3.txt') as f:
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
with open('./Dataset/chebi_smiles_up100_t3.txt') as f:
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

#label_encoding = [torch.cat((t1,t2)) for t1, t2 in zip(label_encoding_des, label_encoding_smiles) ]
label_encoding = [t1+t2 for t1, t2 in zip(label_encoding_des, label_encoding_smiles) ]



label_encoding_dict = {i: label_encoding[i].detach().cpu() for i in range(len(label_encoding))}

#val_emb_filtered = val_emb[y_val == 2]

val_emb = np.array(val_emb)
label_encoding = np.array([le.detach().cpu().numpy() for le in label_encoding])

# Concatenate val_emb and label_encoding for t-SNE

combined_emb = np.concatenate((val_emb, label_encoding), axis=0)
print(len(combined_emb))
print(len(combined_emb[0]))
# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(combined_emb)

# Plotting
plt.figure(figsize=(8, 8))
plt.scatter(tsne_results[:len(val_emb), 0], tsne_results[:len(val_emb), 1], color='lightgray', alpha=0.7, label='Protein')

# Plot label_encoding in red
plt.scatter(tsne_results[len(val_emb):, 0], tsne_results[len(val_emb):, 1], color='red', alpha=0.7, label='Substrate')

#plt.title('t-SNE plot of val_emb and label_encoding')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

# Add a legend
plt.legend()

# Save the plot as an SVG file
plt.savefig(f'Results/tsne_plot_prot_label_enc_{model_name}.svg', format='svg')
