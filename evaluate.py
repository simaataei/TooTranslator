
#from knn import  y_val, predictions, model_name
#from Data_preparation_other import  X_train_other, X_val_other, X_test_other, y_train_other, y_test_other, y_val_other
#from Data_preparation_rej import  X_train, X_val, X_test, y_train, y_test, y_val
#from Data_preparation import  X_train, X_val, X_test, y_train, y_test, y_val
from pycm import *
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Dataset
#from Data_preparation_sc import read_data
import torch
import json
'''
all_gold = np.random.randint(0, 100, 500)  # Example data
all_pred = np.random.randint(0, 100, 500)  # Example data
label_list = [f'Class {i}' for i in range(100)]  # Example class labels

# Compute the confusion matrix
cm_skl_mis = confusion_matrix(all_gold, all_pred)
cm_skl_mis_normal = cm_skl_mis.astype('float') / cm_skl_mis.sum(axis=1)[:, np.newaxis]

# Create a large figure and axis
fig, ax = plt.subplots(figsize=(20, 20))  # Increase figure size

fig.set_facecolor('white')
cmap_colors = [(1, 1, 1)]  # White color for value 0
cmap_colors.extend(plt.cm.Greens(np.linspace(0, 1, 256)))  # Shades of green
cmap = LinearSegmentedColormap.from_list('CustomGreen', cmap_colors)

# Plot the confusion matrix
im = ax.imshow(cm_skl_mis_normal, cmap=cmap)

# Customize the plot
ax.set_xticks(np.arange(cm_skl_mis.shape[1]))
ax.set_yticks(np.arange(cm_skl_mis.shape[0]))
ax.set_xticklabels(label_list, rotation=90, fontsize=5)  # Smaller font size for space
ax.set_yticklabels(label_list, fontsize=5)
ax.set_xlabel('Predicted label', fontsize=12)
ax.set_ylabel('True label', fontsize=12)
ax.set_title('Confusion Matrix', fontsize=16)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax, fraction =0.046, pad = 0.04)
cbar.set_label('Normalized Values', fontsize=12)

# Add text annotations for confusion matrix values
for i in range(cm_skl_mis.shape[0]):
    for j in range(cm_skl_mis.shape[1]):
        text = ax.text(j, i, f'{cm_skl_mis[i, j]:.2f}', ha='center', va='center', color='black', fontsize=4)  # Adjust text font size

plt.savefig('./Results/temp.svg', bbox_inches='tight')


'''
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

#def evaluate(model_name, all_pred, all_gold):
def evaluate():
    train_dataset = ProteinSeqDataset('./Dataset/ICAT/inorganic_uniprot_ident100_t10_train.csv')
    #val_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
    test_dataset = ProteinSeqDataset('./Dataset/ICAT/inorganic_uniprot_ident100_t10_test.csv')

    X_train = [train_dataset[i]['seq'] for i in range(len(train_dataset))]
    y_train =  [train_dataset[i]['labels'] for i in range(len(train_dataset))]

    X_val = [test_dataset[i]['seq'] for i in range(len(test_dataset))]
    y_val =  [test_dataset[i]['labels'] for i in range(len(test_dataset))]
    model_name = 'Prot_bert_bfd_e10_lr5e-06_loss_wMSE_ICAT_des_sum_smile_f1'

    #model_name = 'triplet_euqlidean_online'
    file_name = 'ICAT'
    dis_metric = 'wMSE'
    #dis_metric = model_name.split('_')[2]
    print(dis_metric)
    #all_gold = y_val
    #all_pred = predictions
    #all_gold, all_pred = knn(model_name)
    label_dict = {}
    with open('./Dataset/ICAT/Label_name_list_ICAT_uni_ident100_t10') as f:
         data = f.readlines()
         for d in data:
             d = d.split(',')
             label_dict[d[2].strip('\n')] = d[1]
#    with open('./Dataset/spot/number_name_spot.txt') as f:
#          label_dict = json.load(f)
     
    label_dict = {int(k.strip().strip('"')):v.strip().strip('"') for k,v in label_dict.items()}
    label_list = list(label_dict.values())

    overall = []
    #all_gold = [int(i) for i in y_val]
    # Define the file path
    file_path_pred = f'Results/ICAT/all_pred_{model_name}.txt'
    file_path_gold = f'Results/ICAT/all_gold_{model_name}.txt'
    #file_path = 'Results/spot/preds_seen.txt'
    # Initialize an empty list to store the array elements
    all_pred = []

    # Open the file and read each line
    with open(file_path_pred, 'r') as file:
         lines = file.read().strip()
    all_pred =[int(x) for x in lines.split('\n')]
    
    all_gold = []

    # Open the file and read each line
    with open(file_path_gold, 'r') as file:
         lines = file.read().strip()
    all_gold =[int(x) for x in lines.split('\n')]

        # Iterate over each line in the file
         #for line in file:
          #   print(line)
            # Strip any whitespace and convert to integer
           #  all_pred.append(int(line.strip()))
    
    '''file_path = 'Results/spot/gold_seen.txt'
    with open(file_path, 'r') as file:
         lines = file.read().strip()
    all_gold =[int(x) for x in lines.split(',')] '''
    #all_pred = [int(i) for i in all_pred]
    print(len(all_pred))
    print(len(all_gold))
    cm = ConfusionMatrix(all_gold, all_pred ,digit=5)
    print(cm)
    overall.append('Overall')
    overall.append(len(X_train))
    overall.append(len(X_val))
    overall.append(sum(list(cm.TP.values())))
    overall.append(sum(list(cm.FP.values())))
    overall.append(sum(list(cm.FN.values())))
    overall.append(sum(list(cm.TN.values())))

    overall.append(round(cm.overall_stat['ACC Macro'],3))
    overall.append(round(cm.overall_stat['PPV Micro'],3))
    overall.append(round(cm.overall_stat['TPR Macro'],3) if cm.overall_stat['TPR Macro']!= 'None' else '-')
    overall.append(round(cm.overall_stat['F1 Macro'],3))
    overall.append(round(cm.overall_stat['Overall MCC'],3) if cm.overall_stat['Overall MCC']!= 'None' else '-')


    acc = list(cm.class_stat['ACC'].values())
    mcc = list(cm.class_stat['MCC'].values())
    f1 = list(cm.class_stat['F1'].values())
    pre = list(cm.class_stat['PPV'].values())
    rec = list(cm.class_stat['TPR'].values())

    acc = ["{:1.3f}".format(float(i)) for i in acc]
    f1 = ["{:1.3f}".format(float(i)) for i in f1]

    pre = [ "{:1.3f}".format(float(i)) if i != 'None' else '-' for i in pre ]
    rec = ["{:1.3f}".format(float(i)) if i != 'None' else '-'  for i in rec]
    mcc = ["{:1.3f}".format(float(i)) if i != 'None' else '-' for i in mcc ]

    df_train = pd.DataFrame({'Seq':X_train,'Class':y_train})
    df_train = df_train.groupby(by='Class').count()
    df_train.columns = ['Trainset']
    df_train

    df_val = pd.DataFrame({'Seq':X_val,'Class':y_val})
    df_val= df_val.groupby(by='Class').count()
    df_val.columns = ['Validation']
    df_val

    print(label_list)
    df_ind_ss = pd.DataFrame(list(zip(label_list,list(df_train.Trainset),list(df_val.Validation),acc,pre,rec,f1,mcc)))
    df_ind_ss.columns = ['Substrate','Trainset','Testset','Accuracy', 'Precision','Recall','F1-Score','MCC']

    d_TP = {int(k): int(v) for k,v in cm.TP.items()}
    df_TP = pd.DataFrame.from_dict(d_TP,orient='index')
    df_TP.sort_index(inplace=True)
    df_TP.columns = ['TP']
    d_FP = {int(k): int(v) for k,v in cm.FP.items()}
    df_FP = pd.DataFrame.from_dict(d_FP,orient='index')
    df_FP.sort_index(inplace=True)
    df_FP.columns = ['FP']
    d_FN = {int(k): int(v) for k,v in cm.FN.items()}
    df_FN = pd.DataFrame.from_dict(d_FN,orient='index')
    df_FN.sort_index(inplace=True)
    df_FN.columns = ['FN']
    d_TN = {int(k): int(v) for k,v in cm.TN.items()}
    df_TN = pd.DataFrame.from_dict(d_TN,orient='index')
    df_TN.sort_index(inplace=True)
    df_TN.columns = ['TN']

    print(df_ind_ss.Substrate)
    df_ind_ss_d = df_ind_ss.join(df_TP.join(df_FP).join(df_FN).join(df_TN))
    cols = df_ind_ss_d.columns
    df_ind_ss_d = df_ind_ss_d[['Substrate','Trainset','Testset','TP','FP','FN','TN','Accuracy', 'Precision','Recall','F1-Score','MCC']]
    df_ind_ss_d = df_ind_ss_d.sort_values('Trainset', ascending=False)

    print(df_ind_ss_d.Substrate)
    overall_reshaped = np.reshape(overall, (1, 12))
    overall_df = pd.DataFrame(overall_reshaped, columns=df_ind_ss_d.columns)
    df_ind_ss_d  = pd.concat([df_ind_ss_d,overall_df], ignore_index=True)


    #with open(f'./Results/{dis_metric}/{file_name}_offline_sampling_up_100_{model_name}_latex.txt', 'w') as f:
    with open(f'./Results/ICAT/evaluation_{model_name}.txt', 'w') as f:
         with pd.option_context("max_colwidth", 1000):
              f.write(df_ind_ss_d.to_latex(index=False))


evaluate()

def evaluate_manual_avg():
    # Dataset loading and preparation (same as before)
    train_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f3.csv')
    test_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_test_f3.csv')

    X_train = [train_dataset[i]['seq'] for i in range(len(train_dataset))]
    y_train = [train_dataset[i]['labels'] for i in range(len(train_dataset))]

    X_val = [test_dataset[i]['seq'] for i in range(len(test_dataset))]
    y_val = [test_dataset[i]['labels'] for i in range(len(test_dataset))]

    model_name = 'Prot_bert_bfd_e0_lr5e-06_loss_wMSE_spec_up100_sum_des_smiles_f1'
    file_name = 'spec'
    dis_metric = 'wMSE'
    print(dis_metric)
    
    label_dict = {}
    with open('./Dataset/Label_name_list_transporter_uni_ident100_t3') as f:
        data = f.readlines()
        for d in data:
            d = d.split(',')
            label_dict[d[2].strip('\n')] = d[1]
    label_dict = {int(k): v for k, v in label_dict.items()}
    label_list = list(label_dict.values())

    all_gold = [int(i) for i in y_val]
    file_path = f'Results/cv/all_pred_{model_name}.txt'
    all_pred = []

    with open(file_path, 'r') as file:
        for line in file:
            all_pred.append(int(line.strip()))

    cm = ConfusionMatrix(all_gold, all_pred, digit=5)
    print(cm)

    # Extract class stats
    acc = list(cm.class_stat['ACC'].values())
    mcc = list(cm.class_stat['MCC'].values())
    f1 = list(cm.class_stat['F1'].values())
    pre = list(cm.class_stat['PPV'].values())
    rec = list(cm.class_stat['TPR'].values())

    # Extract TP, TN, FP, FN
    tp = list(cm.class_stat['TP'].values())
    tn = list(cm.class_stat['TN'].values())
    fp = list(cm.class_stat['FP'].values())
    fn = list(cm.class_stat['FN'].values())

    # Create DataFrame
    df = pd.DataFrame({
        'Substrate': label_list,
        'Trainset': [df_train.Trainset.get(i, 0) for i in range(len(label_list))],
        'Testset': [df_val.Validation.get(i, 0) for i in range(len(label_list))],
        'Accuracy': acc,
        'Precision': pre,
        'Recall': rec,
        'F1-Score': f1,
        'MCC': mcc,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    })

    # Filter out rows where MCC is NaN or negative
    valid_indices = df['MCC'].notna() & (df['MCC'] >= 0)
    df_filtered = df[valid_indices]

    # Weighted averages with 'Trainset' as weights
    total_weight = df_filtered['Trainset'].sum()

    weighted_acc = (df_filtered['Accuracy'] * df_filtered['Trainset']).sum() / total_weight
    weighted_pre = (df_filtered['Precision'] * df_filtered['Trainset']).sum() / total_weight
    weighted_rec = (df_filtered['Recall'] * df_filtered['Trainset']).sum() / total_weight
    weighted_f1 = (df_filtered['F1-Score'] * df_filtered['Trainset']).sum() / total_weight
    weighted_mcc = (df_filtered['MCC'] * df_filtered['Trainset']).sum() / total_weight

    # Overall metrics output
    overall = [
        round(weighted_acc, 3),
        round(weighted_pre, 3),
        round(weighted_rec, 3),
        round(weighted_f1, 3),
        round(weighted_mcc, 3)
    ]

    # Print and save the filtered DataFrame
    df_ind_ss_d = df[['Substrate', 'Trainset', 'Testset', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'TP', 'TN', 'FP', 'FN']]
    df_ind_ss_d = df_ind_ss_d.sort_values('Trainset', ascending=False)

    overall_reshaped = np.reshape(overall, (1, 5))
    overall_df = pd.DataFrame(overall_reshaped, columns=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC'])
    
    # Concatenate overall metrics with the DataFrame (ignoring indices)
    df_ind_ss_d = pd.concat([df_ind_ss_d, overall_df], ignore_index=True)

    with open(f'./Results/cv/evaluation_manual_avg_{model_name}.txt', 'w') as f:
        with pd.option_context("max_colwidth", 1000):
            f.write(df_ind_ss_d.to_latex(index=False))

#evaluate_manual_avg()




def conf_matrix(model_name, all_pred, all_gold):
    all_gold = [int(i) for i in all_gold]
    all_pred = [int(i) for i in all_pred]
    print(all_pred)
    print(all_gold)
    dis_metric = model_name.split('_')[2]
    print(dis_metric)
    file_name = 'spec'
    label_dict = {}
    with open('./Dataset/Label_name_list_transporter_uni_ident100_t3') as f:
         data = f.readlines()
         for d in data:
             d = d.split(',')
             label_dict[d[2].strip('\n')] = d[1]
    label_dict = {int(k):v for k,v in label_dict.items()}
    label_list = list(label_dict.values())
    cm_skl_mis = confusion_matrix(all_gold, all_pred)
    print(cm_skl_mis)
    cm_skl_mis_normal = cm_skl_mis.astype('float') / cm_skl_mis.sum(axis=1)[:, np.newaxis]
    # Create a figure and axis

    fig, ax = plt.subplots(figsize=(20, 20))  # Increase figure size

    fig.set_facecolor('white')
    cmap_colors = [(1, 1, 1)]  # White color for value 0
    cmap_colors.extend(plt.cm.Greens(np.linspace(0, 1, 256)))  # Shades of green
    cmap = LinearSegmentedColormap.from_list('CustomGreen', cmap_colors)

    # Plot the confusion matrix
    im = ax.imshow(cm_skl_mis_normal, cmap=cmap)

    # Customize the plot
    ax.set_xticks(np.arange(cm_skl_mis.shape[1]))
    ax.set_yticks(np.arange(cm_skl_mis.shape[0]))
    ax.set_xticklabels(label_list, rotation=90, fontsize=5)  # Smaller font size for space
    ax.set_yticklabels(label_list, fontsize=5)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=16)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction =0.046, pad = 0.04)
    cbar.set_label('Normalized Values', fontsize=12)
    add_numbers = True  # Set to False to save image without numbers
    if add_numbers:
       for i in range(cm_skl_mis.shape[0]):
           for j in range(cm_skl_mis.shape[1]):
               text = ax.text(j, i, f'{cm_skl_mis[i, j]:.2f}', ha='center', va='center', color='black', fontsize=6)

    # Save the plot with numbers
    #plt.savefig(f'./Results/{dis_metric}/offline_{model_name}_{file_name}_with_numbers.svg', bbox_inches='tight')
    plt.savefig('./Results/cv/online_eq_cm.svg', bbox_inches='tight')
    # Remove text annotations and save another version without numbers
    if add_numbers:
       for text in ax.texts:
           text.set_visible(False)
       plt.savefig(f'./Results/{dis_metric}/offline_{model_name}.svg', bbox_inches='tight')

