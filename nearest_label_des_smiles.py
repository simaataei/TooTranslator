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
def mae_distance(vec1, vec2):
    """
    Calculate the MAE-like distance between two vectors.

    Parameters:
    vec1 (np.ndarray): The first vector.
    vec2 (np.ndarray): The second vector.

    Returns:
    float: The MAE-like distance between the two vectors.
    """
    # Compute the absolute difference between the vectors
    abs_diff = np.abs(vec1 - vec2)
    
    # Sum the absolute differences to get the total distance
    return np.sum(abs_diff)

def huber_distance(vec1, vec2, delta=1.0):
    """
    Calculate the Huber-like distance between two vectors.

    Parameters:
    vec1 (np.ndarray): The first vector.
    vec2 (np.ndarray): The second vector.
    delta (float): The threshold at which to transition from quadratic to linear.

    Returns:
    float: The Huber-like distance between the two vectors.
    """
    # Compute the difference between the vectors
    diff = vec1 - vec2
    
    # Apply the Huber Loss element-wise
    huber_loss = np.where(np.abs(diff) <= delta,
                          0.5 * diff**2,
                          delta * (np.abs(diff) - 0.5 * delta))
    
    # Sum the losses to get the total distance
    return np.sum(huber_loss)


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


train_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f3.csv')
#val_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
test_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_test_f3.csv')

minority_dataset = ProteinSeqDataset('./Dataset/data_label_tansporter_D_minority_tokenized.csv')
file_name  ='minority'





#load model
#model = BertModel.from_pretrained('Rostlab/prot_bert_bfd', output_hidden_states=True)
#lora_config = LoraConfig(
#    task_type=TaskType.FEATURE_EXTRACTION, r=1, lora_alpha=1, lora_dropout=0.1,  target_modules= ["embedding", "query","key","value"])
#tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd")

#model = SimpleNNWithBERT()
#model = get_peft_model(model, lora_config)
#model.print_trainable_parameters()
#model = ProteinTesterAttention()

model = ProteinTesterResidual().cuda()
model_path = 'Prot_bert_bfd_e100_lr5e-06_loss__Huberloss_spec_up100_sum_des_smiles_res'
model.load_state_dict(torch.load(model_path))
#model_path= 'Prot_bert_bfd_FR'
#model.load_state_dict(torch.load(model_path), strict=False)
model.to(device)
model_name = model_path.strip('./')
#model_name = 'prot_bert_FR'
#testing with KNN
# get the train embedingsi
train_emb = []
val_emb = []
y_train =[]
y_val =[]
#for item in minority_dataset:
for item in test_dataset:
#for i in range(10):
	#item = test_dataset[i]
	with torch.no_grad():
#             tokenized_seq = tokenizer(item['seq'],max_length=1024, padding=True, truncation=True, return_tensors='pt')
 #            input_ids = tokenized_seq['input_ids'].to(device)
  #           attention_mask = tokenized_seq['attention_mask'].to(device)
             # print(tokenized_seq)
     #        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
             print(item['seq'])
             outputs = model(item['seq'])
        #     embeddings = outputs.last_hidden_state[:, 0, :]
             val_emb.append(outputs.detach().cpu().numpy())
             print(outputs)
        #     outputs = model(item['seq'])
         #    val_emb.append(outputs.cpu().numpy())
             y_val.append(int(item['labels']))
             print(int(item['labels']))

	# Manually invoke garbage collection in critical places
	if gc.isenabled():
	    gc.collect()
	torch.cuda.empty_cache()

del model
print(len(val_emb))

model_label = Label_encoder().cuda()
with open('./Dataset/chebi_des_all_t3.txt') as f:
      data = f.read()
      term_description = json.loads(data)
    #  print(type(list(term_description.keys())[0]))
      class_description = list(term_description.values())


#print(term_description)
y_train_desc = []

#for i in y_train:
for i in y_val:
    print(i)
    y_train_desc.append(class_description[i])

label_encoding_des= []
model_label.eval()
with torch.no_grad():
   for item in class_description:
  #     print(item)
       label_encoding_des.append(model_label(item))
del model_label

model_label = Chem_BERTa_encoder().cuda()
with open('./Dataset/chebi_smiles_all_t3.txt') as f:
      data = f.read()
      term_smiles = json.loads(data)
   #   print(type(list(term_smiles.keys())[0]))
      class_smiles = list(term_smiles.values())


#print(term_description)
y_train_smiles = []

#for i in y_train:
for i in y_val:
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
print(label_encoding_dict)    
#number_encoding_dict = create_number_to_encoding_dict('Dataset/Label_name_list_transporter_uni_ident100_t3', label_encoding_dict)
#print('labels')
#print(len(list(number_encoding_dict.values())[0]))

#label_encoding_dict = {}

#model_label.eval()
#with torch.no_grad():
#     for label in set(y_train):
#         label_encoding_dict[label] = model_label(class_description[label]).detach().cpu().numpy()



#for i in range(95, len(class_description)-95):
 #   label_encoding_dict[i] =model_label(class_description[i]).detach().cpu().numpy()


#print('dict label encoding')
#print(label_encoding_dict)
#label_encoding= []
#model_label.eval()
#with torch.no_grad():
#  for item in class_description:
#    label_encoding.append(model_label(item).detach().cpu().numpy())


#del model_label

def find_closest_index(new_array, dictionary):
    min_distance = float('inf')  # Start with a large number
    closest_index = None
    distance_list =[] 
    for index, array in dictionary.items():
        distance = mse_distance(new_array, array.detach().cpu().numpy())
        distance_list.append(distance)
        #print(array)
        print(index)
        print(distance)
        if distance < min_distance:
            min_distance = distance
            closest_index = index
    print('distance_list') 
    print(distance_list)            
    print(distance_list.index(min(distance_list)))
    return closest_index

# Call the function for each new array and print the result
#for i, array in enumerate(new_arrays):
#    closest_index = find_closest_index(array,  label_encoding_dict)
#    print(f'The index with the minimum distance for array {i} is: {closest_index}')

neighbor =[]
#del label_encoding_dict[10]
print(val_emb[0])
print(type(val_emb[0]))
print(len(val_emb[0]))
print(val_emb[3])
print(type(val_emb[3]))
print(len(val_emb[3]))
print(str(val_emb[0]== val_emb[1]))
for item in val_emb:
    #closest_vector, min_distance= find_closest_vector(item, label_encoding_dict)
   # print(item)
    closest_vector= find_closest_index(item, label_encoding_dict)
    neighbor.append(closest_vector) 
    #closest_vector, min_distance= find_closest_vector(item, label_encoding)
    #index = label_encoding.index(closest_vactor)
    #neigbor.append(y_train[index])
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
with open(f'Results/cv/all_pred_{model_name}.txt', 'w') as file:
    # Iterate through the list and write each number to a new line
    for item in neighbor:
        file.write(f"{item}\n")



with open(f'Results/cv/all_gold_{model_name}.txt', 'w') as file:
    # Iterate through the list and write each number to a new line
    for item in y_val:
        file.write(f"{item}\n")



