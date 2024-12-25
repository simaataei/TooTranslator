from model import ProteinTester,  ProteinClassifier, ProteinDescriptionClassifier, Label_encoder, Chem_BERTa_encoder
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
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist





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
def manhattan_distance(x1, x2):
    return cdist(x1, x2, metric='cityblock')
#    return torch.cdist(x1, x2, p=1)


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
def find_closest_vector(target_vector, vector_set):
    min_distance = float('inf')
    closest_vector = None
    target_vector = target_vector.numpy()
    for vec in vector_set:
        vec= vec.numpy()
        distance = mse_distance(target_vector, vec)
        if distance < min_distance:
            min_distance = distance
            closest_vector = vec
    print(closest_vector)    
    return closest_vector, min_distance

def find_top_k_closest_vectors(target_vector, vector_set, k=5):
    distances = []
    target_vector = target_vector.numpy()
    
    for vec in vector_set:
        vec = vec.numpy()
        distance = mse_distance(target_vector, vec)
        distances.append((vec, distance))
    
    # Sort the list of tuples by distance
    distances.sort(key=lambda x: x[1])
    
    # Retrieve the top k closest vectors
    top_k_closest_vectors = distances[:k]
    
    # Extract vectors and distances separately
    closest_vectors = [item[0] for item in top_k_closest_vectors]
    closest_distances = [item[1] for item in top_k_closest_vectors]
    
    return closest_vectors, closest_distances



def cosine_distance(a, b):
    # Cosine similarity is between 0 and 1, but we need a distance metric
    # So, we return 1 - cosine similarity
   dot_product = np.dot(a, b)

   # Calculate the magnitudes (norms) of the vectors
   norm_a = np.linalg.norm(a)
   norm_b = np.linalg.norm(b)

   # Calculate the cosine similarity
   cosine_similarity = dot_product / (norm_a * norm_b) 
   return 1 - cosine_similarity

def euqlidean_distance(a, b):
    differences = a - b

    # Square the differences
    squared_differences = differences ** 2

    # Sum the squared differences

    # Take the square root of the sum
    euclidean_distance = np.sqrt(sum_of_squares)
    return euclidean_distance    
 
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
            'seq':' '.join(text),  # Remove batch dimension
            'labels': label
        }








device = torch.device("cuda")
# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
best_mcc =-1



test_file = './Dataset/spot/spot_full_seq_numbers.csv'
test_dataset = ProteinSeqDataset(test_file)


#load model

model = ProteinTester().cuda()

model_path = 'Prot_bert_bfd_e100_lr5e-06_loss__Huberloss_spec_up100_sum_des_smiles_res'
#model_name = 'Prot_bert_bfd_FR'
model_name = model_path.strip('./')
model.load_state_dict(torch.load(model_path))

test_emb =[]
y_test = []
for item in test_dataset:
#for i in range(10):
#       item = test_dataset[i]
        with torch.no_grad():
             #print(item['seq'])
             pred = model(item['seq']).detach().cpu()
             #print(pred)
             #print(type(pred))
             test_emb.append(pred)
             y_test.append(int(item['labels']))

del model
print('test samples')
print(len(test_emb))
print(len(test_emb[0]))
model_label = Label_encoder().cuda()
term_name_dict = {}
number_name_dict = {}
with open('./Dataset/spot/chebi_name_spot.txt') as f:
    data = f.readlines()
    for d in data:
        d = d.split(',')
        term_name_dict[d[0].strip('\n')] = d[1]



with open('./Dataset/spot/number_name_spot.txt') as f:
      data = f.read()
      number_name_dict  = json.loads(data)
number_name_int_keys = {int(k):v for k, v in number_name_dict.items()}
number_name_list = list(number_name_dict.values())
print('number_name')
print(number_name_int_keys)
print(number_name_int_keys.get(123))
print(number_name_int_keys[123])
with open('./Dataset/spot/number_des_spot.txt') as f:
      data = f.read()
      number_description  = json.loads(data)
#number_description_train = {index: value for index, value in enumerate(term_description_train.values())}      
number_description_int_keys = {int(k): v for k, v in number_description.items()}

y_test_desc = []
for i in y_test:
    y_test_desc.append(number_description_int_keys[i])
 
label_encoding_des= []
model_label.eval()
with torch.no_grad():
   for item in number_description_int_keys.values():
  #     print(item)
       label_encoding_des.append(model_label(item))
del model_label

#print('number of descriptions')
#print(len(label_encoding_des))
#print(len(label_encoding_des[1]))



model_smiles = Chem_BERTa_encoder().cuda()
with open('./Dataset/spot/number_smiles_spot.txt') as f:
      data = f.read()
      number_smiles = json.loads(data)

number_smiles_int_keys = {int(k): v for k, v in number_smiles.items()}


y_test_smiles = []
for i in y_test:
    y_test_smiles.append(number_smiles_int_keys[i])


label_encoding_smiles= []
model_smiles.eval()
with torch.no_grad():
   for item in number_smiles_int_keys.values():
       if len(item) > 0:
          encs = []
          for i in item:
              encs.append(model_smiles(i.strip('"')))
          stacked_embeddings = torch.stack(encs)
          label_encoding_smiles.append(torch.mean(stacked_embeddings, dim=0))
       else:
            # Append a dummy SMILES code for padding
          dummy_smiles = "XXX"
          label_encoding_smiles.append(model_smiles(dummy_smiles))
del model_smiles

#print('number of smiles')
#print(len(label_encoding_smiles))
#print(len(label_encoding_smiles[1]))



number_encoding_dict ={}
for item in range(len(label_encoding_des)):
    number_encoding_dict[item] = label_encoding_des[item].detach().cpu()+label_encoding_smiles[item].detach().cpu()

#print('final enc')
#print(number_encoding_dict.keys())
#print(len(list(number_encoding_dict.values())[1]))


true_label_distance = {}
for test_number, item in enumerate(y_test):
    true_label_distance[test_number] = number_encoding_dict[item]



#print(list(number_encoding_dict.values()))
label_encoding= []
#print('dict label encoding')
#print(len(number_encoding_dict))
#print('unique')
#print(np.unique(list(number_encoding_dict.values())))
preds = []
golds = []
neighbor_info = []
for idx, item in enumerate(test_emb):
    closest_vector, min_distance = find_closest_vector(item, list(number_encoding_dict.values()))

    for key, val in number_encoding_dict.items():
        if (closest_vector == val.numpy()).all():
            term = key  # Get the term (key in number_encoding_dict)
            preds.append(key)
            print(term)
            label_name = number_name_int_keys.get(term)  # Get the label name from label_dict or "Unknown" if not found
    #        print('label name'+str(label_name))
            # Calculate the distance to the true label
            true_label_vector = number_encoding_dict[y_test[idx]]
 #           print(true_label_vector)
  #          print(type(item))
   #         print(type(true_label_vector))
            distance_to_true_label = mse_distance(item.numpy(), true_label_vector.numpy())
             
            # Get the true label name
            true_label_name = number_name_int_keys.get(y_test[idx])
            golds.append(y_test[idx])
            is_same_label = label_name == true_label_name

            neighbor_info.append((idx, label_name, min_distance, true_label_name, distance_to_true_label, is_same_label))
            break


number_encoding_dict_serializable = {k: v.tolist() for k, v in number_encoding_dict.items()}

# Save the serialized dictionary to a JSON file
with open('Dataset/spot/number_encoding_dict.json', 'w') as file:
    json.dump(number_encoding_dict_serializable, file)
print(classification_report(golds,preds))
pd.set_option('display.max_rows', None)  # Set no limit on rows
pd.set_option('display.max_colwidth', None)
# Save the neighbor info to a file
with open(f'Results/spot/spot_predictions_{model_name}.txt', 'w') as file:
    # Write the header
    file.write("SampleNumber,LabelName,Distance,TrueLabelName,DistanceToTrueLabel,IsSameLabel\n")
    # Iterate through the list and write each tuple to the file with formatted distances
    for info in neighbor_info:
        file.write(f"{info[0]},{info[1]},{info[2]:.2f},{info[3]},{info[4]:.2f},{info[5]}\n")

columns = ["SampleNumber", "LabelName", "Distance", "TrueLabelName", "DistanceToTrueLabel", "IsSameLabel"]
df = pd.DataFrame(neighbor_info, columns=columns)

# Save the DataFrame as a LaTeX table
latex_table = df.to_latex(index=False, float_format="%.2f", caption="Results of the model predictions", label="tab:results", longtable=True)

# Save the LaTeX table to a .tex file
with open(f'Results/spot/spot_{model_name}_latex.tex', 'w') as file:
     with pd.option_context('display.max_rows', None, "max_colwidth", 2000):
          file.write(df.to_latex(index=False, float_format="%.2f", caption="Results of the model predictions", label="tab:results", longtable=True))
# Print the neighbor label names and generate the classification report
file_path = 'Dataset/Label_name_list_transporter_uni_ident100_t3'  # Update this with your file's path
names_list = []

# Read the file and extract the second column (the name) into a list
with open(file_path, 'r') as f:
    for line in f:
        columns = line.split(',')  # Split the line by commas
        names_list.append(columns[1].strip())  # Keep the second column (name) and strip extra spaces

# Create a mapping for the order of names in the list
name_order = {name: i for i, name in enumerate(names_list)}
seen_class_samples = df[df['TrueLabelName'].isin(names_list)]['SampleNumber'].tolist()
unseen_class_samples = df[~df['TrueLabelName'].isin(names_list)]['SampleNumber'].tolist()
# separate seen and unseen predictions for evaluation

gold_seen = []
gold_unseen = []
preds_seen = []
preds_unseen = []
for i, (gold_value, preds_value) in enumerate(zip(golds, preds)):
    if i in seen_class_samples:
        gold_seen.append(gold_value)
        preds_seen.append(preds_value)
    elif i in unseen_class_samples:
        gold_unseen.append(gold_value)
        preds_unseen.append(preds_value)
with open('Results/spot/gold_seen.txt', 'w') as f:
    f.write(','.join(map(str, gold_seen)))

with open('Results/spot/gold_unseen.txt', 'w') as f:
    f.write(','.join(map(str, gold_unseen)))

with open('Results/spot/preds_seen.txt', 'w') as f:
    f.write(','.join(map(str, preds_seen)))

with open('Results/spot/preds_unseen.txt', 'w') as f:
    f.write(','.join(map(str, preds_unseen)))

# Sort the DataFrame so that rows with LabelName in names_list come first and maintain the same order as the list
df['name_order'] = df['TrueLabelName'].map(name_order)

# Sort first by name order, and then by rows not in the list (NaNs in 'name_order' will be pushed to the end)
df_sorted = df.sort_values(by=['name_order'], na_position='last').drop(columns='name_order')

# Reset the index if desired
df_sorted.reset_index(drop=True, inplace=True)


with open(f'Results/spot/spot_{model_name}__sorted_latex.tex', 'w') as file:
     with pd.option_context('display.max_rows', None, "max_colwidth", 2000):
          file.write(df_sorted.to_latex(index=False, float_format="%.2f", caption="Results of the model predictions", label="tab:results", longtable=True))

neighbor_info = []
for idx, item in enumerate(test_emb):
    closest_vectors, distances = find_top_k_closest_vectors(item, list(number_encoding_dict.values()), k=10)
    
    for i in range(10):  # Iterate over the top 5 closest vectors
        closest_vector = closest_vectors[i]
        min_distance = distances[i]
        
        # Find the corresponding label for the closest vector
        for key, val in number_encoding_dict.items():
            if (closest_vector == val.numpy()).all():
                term = key  # Get the term (key in number_encoding_dict)
                print(term)
                label_name = number_name_int_keys.get(term, "Unknown")  # Get the label name from label_dict or "Unknown" if not found
                print(f'Label name: {label_name}')
                
                # Calculate the distance to the true label
                true_label_vector = number_encoding_dict[y_test[idx]]
                distance_to_true_label = mse_distance(item.numpy(), true_label_vector.numpy())

                # Get the true label name
                true_label_name = number_name_int_keys.get(y_test[idx], "Unknown")
                is_same_label = label_name == true_label_name

                neighbor_info.append((idx, label_name, min_distance, true_label_name, distance_to_true_label, is_same_label))
                break

columns = ["SampleNumber", "LabelName", "Distance", "TrueLabelName", "DistanceToTrueLabel", "IsSameLabel"]
df = pd.DataFrame(neighbor_info, columns=columns)
df_rounded = df.round(2)
# Save the DataFrame as a LaTeX table
latex_table = df.to_latex(index=False, float_format="%.2f", caption="Results of the model predictions", label="tab:results", longtable=True)

# Save the LaTeX table to a .tex file
with open(f'Results/spot/spot_top10_{model_name}_latex.tex', 'w') as file:
     with pd.option_context('display.max_rows', None, "max_colwidth", 2000):
          file.write(df.to_latex(index=False, float_format="%.2f", caption="Results of the model predictions", label="tab:results", longtable=True))
with open(f'Results/spot/spot_top10_{model_name}__latex.tex', 'w') as file:
    # Write the beginning of the LaTeX longtable
    file.write(r'''\begin{longtable}{''' + 'l' * len(df.columns) + r'''}
\caption{Results of the model predictions} \\
\hline
''')

    # Write the header row
    file.write(' & '.join(df_rounded.columns) + r' \\' + '\n')
    file.write(r'\hline' + '\n')

    # Write each row of the DataFrame as a line in the LaTeX table
    for _, row in df_rounded.iterrows():
        row_data = ' & '.join([str(x) for x in row])
        file.write(row_data + r' \\' + '\n')

    # Write the end of the longtable
    file.write(r'\end{longtable}')

neighbor_labels = [info[1] for info in neighbor_info]
#print(neighbor_labels)
#print(y_test)
