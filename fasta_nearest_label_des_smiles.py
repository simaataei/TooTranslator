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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE




def tsne(val_emb, label_encoding):
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
    plt.savefig(f'Results/tsne_plot_prot_label_enc_de_nove_minortiy_lables_{model_name}.svg', format='svg')



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
        distance = huber_distance(target_vector, vec)
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
        distance = huber_distance(target_vector, vec)
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



test_file = './Dataset/data_label_tansporter_D_minority.txt'
#test_file = './Dataset/transporter_uniprot_ident100_t3_test_f1.csv'
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
#       item = val_dataset[i]
        with torch.no_grad():
             print(item['seq'])
             pred = model(item['seq']).detach().cpu()
             print(pred)
             print(type(pred))
             test_emb.append(pred)
             y_test.append(int(item['labels']))

del model
print('test samples')
print(len(test_emb))
print(len(test_emb[0]))
model_label = Label_encoder().cuda()
term_name_dict = {}
number_name_dict = {}
with open('./Dataset/Label_name_list_transporter_uni_ident100_t3') as f:
    data = f.readlines()
    for d in data:
        d = d.split(',')
        term_name_dict[d[0].strip('\n')] = d[1]
        number_name_dict[int(d[2].strip())]=d[1] 

with open('./Dataset/Label_name_list_transporter_uni_ident100_D_minority') as f:
    data = f.readlines()
    for d in data:
        d = d.split(',')
        term_name_dict[d[0].strip('\n')] = d[1]
        number_name_dict[int(d[2].strip())]=d[1]
#print('term name')
#print(term_name_dict.keys())
#print('number name')
#print(number_name_dict.keys())


number_name_dict = {int(k): v for k, v in number_name_dict.items()}
number_name_list = list(number_name_dict.values())


with open('./Dataset/chebi_des_up100_t3.txt') as f:
      data = f.read()
      term_description_train = json.loads(data)
number_description_train = {index: value for index, value in enumerate(term_description_train.values())}      
#print(len(number_description_train))

with open('./Dataset/chebi_des_all_D_minority.txt') as f:
      data = f.read()
     # print(data)
      term_description_D_minority = json.loads(data)
     # print(term_description_D_minority)
number_description_D_minority = {index+96: value for index, value in enumerate(term_description_D_minority.values())}
#print(len(number_description_D_minority))


#term_description = term_description_train
term_description = {**term_description_train, **term_description_D_minority}
number_description = {**number_description_train, **number_description_D_minority}
#number_description = number_description_D_minority
#number_description = number_description_train

print(number_description.values())
y_test_desc = []

for i in y_test:
 #   print(i)
  #  print(number_description[i])
   # y_test_desc.append(number_description[i])
    y_test_desc.append(number_description[i])
label_encoding_des= []
model_label.eval()
with torch.no_grad():
   for item in number_description.values():
  #     print(item)
       label_encoding_des.append(model_label(item))
del model_label

print('number of descriptions')
print(len(label_encoding_des))
print(len(label_encoding_des[1]))

#number_description = number_description_D_minority



 
#print(len(term_description))
#print(len(number_description))




model_smiles = Chem_BERTa_encoder().cuda()
with open('./Dataset/chebi_smiles_all_D_minority.txt') as f:
      data = f.read()
      term_smiles_D_minority = json.loads(data)
   #   print(type(list(term_smiles.keys())[0]))
      class_smiles_D_minority = list(term_smiles_D_minority.values())

with open('./Dataset/chebi_smiles_all_t3.txt') as f:
      data = f.read()
      term_smiles_train = json.loads(data)
   #   print(type(list(term_smiles.keys())[0]))
      class_smiles_train = list(term_smiles_train.values())


class_smiles = class_smiles_train+class_smiles_D_minority
#class_smiles = class_smiles_train
print('class smiles')
print(class_smiles)
print(len(class_smiles))
number_smiles_D_minority = {index+96: value for index, value in enumerate(class_smiles_D_minority)}
number_smiles = {index: value for index, value in enumerate(class_smiles)}
y_test_smiles = []
#number_smiles = number_smiles_D_minority
for i in y_test:
    print(i)
#    y_test_smiles.append(class_smiles[i])
    y_test_smiles.append(number_smiles[i])


label_encoding_smiles= []
model_smiles.eval()
with torch.no_grad():
   for item in class_smiles:
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

print('number of smiles')
print(len(label_encoding_smiles))
print(len(label_encoding_smiles[1]))



number_encoding_dict ={}
for item in range(len(label_encoding_des)):
    number_encoding_dict[item+96] = label_encoding_des[item].detach().cpu()+label_encoding_smiles[item].detach().cpu()

#for item in range(len(label_encoding_des)):
 #   number_encoding_dict[item] = label_encoding_des[item].detach().cpu()+label_encoding_smiles[item].detach().cpu()

print('final enc')
print(number_encoding_dict.keys())
print(len(list(number_encoding_dict.values())[1]))

tsne(test_emb, list(number_encoding_dict.values()))
true_label_distance = {}
for test_number, item in enumerate(y_test):
    true_label_distance[test_number] = number_encoding_dict[item]



#print(list(number_encoding_dict.values()))
label_encoding= []
#print('dict label encoding')
#print(len(number_encoding_dict))
#print('unique')
#print(np.unique(list(number_encoding_dict.values())))

neighbor_info = []
for idx, item in enumerate(test_emb):
    closest_vector, min_distance = find_closest_vector(item, list(number_encoding_dict.values()))

    for key, val in number_encoding_dict.items():
        if (closest_vector == val.numpy()).all():
            term = key  # Get the term (key in number_encoding_dict)
            label_name = number_name_dict.get(term, "Unknown")  # Get the label name from label_dict or "Unknown" if not found
            print('label name'+str(label_name))
            # Calculate the distance to the true label
            true_label_vector = number_encoding_dict[y_test[idx]]
            print(true_label_vector)
            print(type(item))
            print(type(true_label_vector))
            distance_to_true_label = huber_distance(item.numpy(), true_label_vector.numpy())
           
            # Get the true label name
            true_label_name = number_name_dict.get(y_test[idx], "Unknown")
            is_same_label = label_name == true_label_name

            neighbor_info.append((idx, label_name, min_distance, true_label_name, distance_to_true_label, is_same_label))
            break

# Save the neighbor info to a file
with open(f'Results/predictions_testset_{model_name}.txt', 'w') as file:
    # Write the header
    file.write("SampleNumber,LabelName,Distance,TrueLabelName,DistanceToTrueLabel,IsSameLabel\n")
    # Iterate through the list and write each tuple to the file with formatted distances
    for info in neighbor_info:
        file.write(f"{info[0]},{info[1]},{info[2]:.2f},{info[3]},{info[4]:.2f},{info[5]}\n")

columns = ["SampleNumber", "LabelName", "Distance", "TrueLabelName", "DistanceToTrueLabel", "IsSameLabel"]
df = pd.DataFrame(neighbor_info, columns=columns)

# Save the DataFrame as a LaTeX table
latex_table = df.to_latex(index=False, float_format="%.2f", caption="Results of the model predictions", label="tab:results")

# Save the LaTeX table to a .tex file
with open(f'Results/predictions_D_minority_{model_name}_latex.tex', 'w') as file:
    file.write(latex_table)
# Print the neighbor label names and generate the classification report


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
                label_name = number_name_dict.get(term, "Unknown")  # Get the label name from label_dict or "Unknown" if not found
                print(f'Label name: {label_name}')
                
                # Calculate the distance to the true label
                true_label_vector = number_encoding_dict[y_test[idx]]
                distance_to_true_label = huber_distance(item.numpy(), true_label_vector.numpy())

                # Get the true label name
                true_label_name = number_name_dict.get(y_test[idx], "Unknown")
                is_same_label = label_name == true_label_name

                neighbor_info.append((idx, label_name, min_distance, true_label_name, distance_to_true_label, is_same_label))
                break

columns = ["SampleNumber", "LabelName", "Distance", "TrueLabelName", "DistanceToTrueLabel", "IsSameLabel"]
df = pd.DataFrame(neighbor_info, columns=columns)

# Save the DataFrame as a LaTeX table
latex_table = df.to_latex(index=False, float_format="%.2f", caption="Results of the model predictions", label="tab:results")

# Save the LaTeX table to a .tex file
with open(f'Results/predictions_top10_D_minority_{model_name}_latex.tex', 'w') as file:
    file.write(latex_table)


neighbor_labels = [info[1] for info in neighbor_info]
print(neighbor_labels)
print(y_test)

# Assuming y_test is in terms of original classes, you might need to map y_test back to labels for the report
#inverse_label_dict = {v: k for k, v in number_name_dict.items()}
#predicted_classes = [inverse_label_dict.get(label, -1) for label in neighbor_labels]
#print(predicted_classes)
#print(classification_report(y_test, predicted_classes))

'''
neighbor =[]
for item in test_emb:
    closest_vector, min_distance= find_closest_vector(item, list(number_encoding_dict.values()))
    
    for key, val in number_encoding_dict.items():
#        print('closest_vector')
 #       print(len(closest_vector))
  #      print('val')
   #     print(len(val))
        if (closest_vector == val.numpy()).all():
           neighbor.append(int(key))
           continue
       # else:
        #   neigbor.append(1000)
print(neighbor)
print(y_test)
print(classification_report(y_test,neighbor))

with open(f'Results/all_pred_{model_name}.txt', 'w') as file:
    # Iterate through the list and write each number to a new line
    for item in neighbor:
        file.write(f"{item}\n")



with open(f'Results/all_gold_{model_name}.txt', 'w') as file:
    # Iterate through the list and write each number to a new line
    for item in y_test:
        file.write(f"{item}\n")

'''
