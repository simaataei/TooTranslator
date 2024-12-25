import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
import json
from torch import optim
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
from data_pre import X_train, X_val, y_train, y_val
from model import ProteinTester,  ProteinClassifier, ProteinDescriptionClassifier, Label_encoder, Chem_BERTa_encoder
import json
import torch.nn.functional as F

# Assuming X_train and y_train are your input features and labels
# Initialize model, optimizer, etc.

# Custom Mahalanobis distance loss function
def combined_mahalanobis_mse_loss(pred, target, target_encoded, class_means, class_covariances, alpha=0.5):
    """
    Compute the combined loss: Mahalanobis distance and MSE loss.

    Parameters:
    - pred: Tensor, model predictions (outputs).
    - target: Tensor, true class indices.
    - target_encoded: Tensor, true encoded class descriptions.
    - class_means: Dict, mean vectors for each class.
    - class_covariances: Dict, covariance matrices for each class.
    - alpha: float, weight for the Mahalanobis distance in the combined loss.

    Returns:
    - loss: Tensor, the combined loss.
    """
    # Mahalanobis distance loss
    mahalanobis_loss = 0.0
    for i in range(pred.size(0)):  # Loop over each sample
        label = int(target[i].item())  # Assuming target contains class indices
        mean = class_means[label]
        cov = class_covariances[label]
        
        # Ensure numerical stability with a small value added to the diagonal
        inv_cov = torch.inverse(cov + 1e-5 * torch.eye(cov.size(0)).cuda())
        
        # Compute difference between predicted and class mean
        diff = pred[i] - mean
        
        # Calculate Mahalanobis distance
        dist = torch.dot(diff, torch.mv(inv_cov, diff))
        mahalanobis_loss += dist

    mahalanobis_loss /= pred.size(0)  # Average the Mahalanobis loss

    # MSE loss
    mse_loss = nn.MSELoss()(pred, target_encoded)

    # Combine losses
    combined_loss = alpha * mahalanobis_loss + (1 - alpha) * mse_loss
    return combined_loss
class MahalanobisDistanceLoss(nn.Module):
    def __init__(self, covariance_matrix, device =None):
        super(MahalanobisDistanceLoss, self).__init__()
        # Convert covariance matrix to a tensor and compute its inverse 
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        self.covariance_matrix = torch.tensor(covariance_matrix, dtype=torch.float32).to(self.device)
        self.cov_inv = torch.linalg.inv(self.covariance_matrix)

    def forward(self, embeddings1, embeddings2, target):
        """
        embeddings1: torch.Tensor of shape (batch_size, embedding_dim)
        embeddings2: torch.Tensor of shape (batch_size, embedding_dim)
        target: torch.Tensor of shape (batch_size) with 1 for same class, 0 for different class
        """
        embeddings1 = embeddings1.to(self.device)
        embeddings2 = embeddings2.to(self.device)
        target = target.to(self.device)
        
        diff = embeddings1 - embeddings2
        if len(diff.shape) == 1:
           diff = diff.unsqueeze(0)
        # Compute Mahalanobis distance for each pair
        distances = torch.sqrt(torch.diag(torch.mm(torch.mm(diff, self.cov_inv), diff.t())))

        # Apply different losses based on the target
        loss_same = target * distances  # Target is 1 for same class pairs
        loss_diff = (1 - target) * F.relu(1.0 - distances)  # Target is 0 for different class pairs
        
        # Combine the losses
        loss = torch.mean(loss_same + loss_diff)
        
        return loss




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''model_label = Label_encoder().cuda()
with open('./Dataset/chebi_des_up100_t3.txt') as f:
      data = f.read()
      term_description = json.loads(data)
      print(type(list(term_description.keys())[0]))
      class_description = list(term_description.values())


#print(term_description)
y_train_desc = []

for i in y_train:
    y_train_desc.append(class_description[i])

label_encoding= []
model_label.eval()
with torch.no_grad():
  for item in class_description:
    label_encoding.append(model_label(item))
del model_label
'''
model_label = Label_encoder().cuda()
with open('./Dataset/chebi_des_up100_t3.txt') as f:
      data = f.read()
      term_description = json.loads(data)
      print(type(list(term_description.keys())[0]))
      class_description = list(term_description.values())


#print(term_description)
y_train_desc = []

for i in y_train:
    y_train_desc.append(class_description[i])

label_encoding_des= []
model_label.eval()
with torch.no_grad():
   for item in class_description:
       print(item)
       label_encoding_des.append(model_label(item))
del model_label

model_label = Chem_BERTa_encoder().cuda()
with open('./Dataset/chebi_smiles_up100_t3.txt') as f:
      data = f.read()
      term_smiles = json.loads(data)
      print(type(list(term_smiles.keys())[0]))
      class_smiles = list(term_smiles.values())


#print(term_description)
y_train_smiles = []

for i in y_train:
    y_train_smiles.append(class_smiles[i])

label_encoding_smiles= []
model_label.eval()
with torch.no_grad():
   for item in class_smiles:
       print(item)
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

# Training parameters
model = ProteinTester().cuda()
model.eval()
with torch.no_grad():
     X_encodings = []
     for item in X_train:
         X_encodings.append(model(item).detach().cpu().numpy())

covariance_matrix = np.cov(X_encodings, rowvar=False)
loss_function = MahalanobisDistanceLoss(covariance_matrix)
optimizer = optim.Adam(model.parameters(), lr=0.000005)
num_epochs = 100

# Training loop
for epoch in range(1, num_epochs + 1):
    all_loss = []
    start = time.time()
    model.train()
    
    for i in tqdm(range(len(X_train))):
        optimizer.zero_grad()
        
        # Get the sample and its corresponding description
#        sample = torch.tensor(X_train[i], dtype=torch.float32).cuda()
 #       description = torch.tensor(y_train_desc[i], dtype=torch.float32).cuda()
        
        # Forward pass
        pred = model(X_train[i])  # Ensure the sample has batch dimension
        gold = label_encoding[int(y_train[i])]
        target = torch.ones(1, dtype=torch.float32, device=device) 
        # Compute loss
        #loss = mahalanobis_distance_loss(pred, gold, class_means, class_covariances)
        loss = loss_function(pred, gold.to(device), target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        all_loss.append(loss.cpu().detach().numpy())
    
    # Save model checkpoint
    torch.save(model.state_dict(), f"./Prot_bert_bfd_e100_spec_up100_mahal")
    
    print(f"Epoch {epoch}, Loss: {np.mean(all_loss)}")

# Note:
# - Ensure your `X_train` is in a compatible format for the model input.
# - The `Label_encoder` model is used to encode descriptions for initial statistics computation.
# - `y_train_desc` should be tensor-encoded descriptions if needed.

