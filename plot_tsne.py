from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from embedding import  test_emb, y_test, train_emb, y_train



label_dict = {}
with open('./Dataset/Label_name_list_transporter_uni_ident100_t3') as f:
     data = f.readlines()
     for d in data:
         d = d.split(',')
         label_dict[d[2].strip('\n')] = d[1]
label_dict = {int(k):v for k,v in label_dict.items()}

test_emb = train_emb 
y_test = y_train
name = 'c95-96'

train_emb = [emb for emb, label in zip(train_emb, y_train)]
y_train = [label for label in y_train]
print(len(y_train))
print(len(train_emb))

embeddings_stack = np.vstack(train_emb)
print(len(embeddings_stack))
# Fit t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(embeddings_stack)
'''
unique_classes, counts = np.unique(y_test, return_counts=True)
classes_with_three_samples = unique_classes[counts == 3]
print(classes_with_three_samples)
# Filter data for classes with exactly 3 samples
#mask = np.isin(y_test, classes_with_three_samples)

#mask = [indx for indx in range(len(y_test)) if y_test[indx]>72]
#print(mask)
print('tsns')
print(tsne_results)
print(type(tsne_results))
print(type(tsne_results[0]))
y_test = np.array(y_test)
filtered_y_test = np.array([i for i in y_test if i>85 or i<10])
filtered_tsne_results = tsne_results[y_test>72 or y_test<10]
#filtered_tsne_results = [tsne_results[i] for i in mask] 
#filtered_y_test = [y_test[i] for i in mask]
# Plotssuming test_emb and y_test are lists and y_test contains numerical values.
#filtered_test_emb = [emb for emb, label in zip(test_emb, y_test) if label <= 72]
#filtered_y_test = [label for label in y_test if label <= 72]
print('filtered')
# Now filtered_test_emb and filtered_y_test contain only the elements where y_test <= 72
print(filtered_tsne_results)
print(filtered_y_test)
plt.figure(figsize=(14, 10))
scatter = plt.scatter(filtered_tsne_results[:, 0], filtered_tsne_results[:, 1], c=filtered_y_test, cmap='Spectral', s=5)
plt.colorbar(scatter, boundaries=np.arange((96-72) + 1) - 0.5).set_ticks(np.arange((96-72)))
plt.title('t-SNE projection of classes with 3 samples', fontsize=20)
#filtered_y_test = y_test[mask]
#filtered_y_test = y_test[mask]
plt.show()
plt.savefig('tsne_rep_triplet_off_e97_mask_after_tsne_top_bottom.svg', format='svg')


'''
# Plot
plt.figure(figsize=(20, 10))
#plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=['green' if label == 4 else 'gray' for label in y_train], s=5)
#plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_train, cmap='Spectral', s=5)
#plt.colorbar(boundaries=np.arange(97)-0.5).set_ticks(np.arange(96))

colors = ['lightgray'] * len(y_train)
for idx, label in enumerate(y_train):
    if 95 <= label <= 96:
        colors[idx] = plt.cm.Spectral((label - 95) / 2)  # Normalize label to range [0, 1] for the colormap

# Plot the t-SNE results
sc = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, s=5)

# Create a legend
legend_labels = ["Other classes"] + [label_dict[i] for i in range(95, 96)]
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=5)] \
          + [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.Spectral((i - 95) / 2), markersize=5) for i in range(95, 96)]

plt.legend(handles, legend_labels, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')



plt.title('t-SNE projection of the minority classes in TooT-SPEC dataset', fontsize=20)
#plt.colorbar().set_ticks([4])
plt.show()
plt.savefig(f'tsne_rep_triplet_off_e97_{name}.svg', format='svg')

