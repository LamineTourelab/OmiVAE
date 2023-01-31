#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:45:57 2023

@author: ltoure
"""
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
#####################################
#           Load Data               #
#####################################

rnaseq = pd.read_csv("/home/ldap/ltoure/multiomics/Multiomics/Input/Data/Rnaseq", sep=",", index_col=0)
traitData = pd.read_csv("//home/ldap/ltoure/multiomics/Multiomics/Input/Data/Response", index_col=0)
DataExome= pd.read_csv("/home/ldap/ltoure/multiomics/Multiomics/Input/Data/Exome", sep=",", index_col=0)
TestIndex100 = pd.read_csv("/home/ldap/ltoure/multiomics/Multiomics/Input/TestIndex100Split", sep=" ")

#############################################
#          Data transformation              #
#############################################

X=rnaseq
y=DataExome
labels=traitData



labels = labels.replace("R",1)
labels = labels.replace("NR",0)


feature_name=X.columns
feature_name_y=y.columns

rnaseq.to_csv("/home/ldap/ltoure/OmiVAE/Data/rnaseq.csv", header=False, index=True)
DataExome.to_csv("/home/ldap/ltoure/OmiVAE/Data/DataExome.csv", header=False, index=True)

labels.to_csv("/home/ldap/ltoure/OmiVAE/Data/labels.csv", header=True, index=True)

feature_name=pd.DataFrame(feature_name)
feature_name.to_csv("/home/ldap/ltoure/OmiVAE/Data/expr_name.csv", header=False, index=False)
feature_name_y=pd.DataFrame(feature_name_y) 
feature_name_y.to_csv("/home/ldap/ltoure/OmiVAE/Data/mut_name.csv", header=False, index=False)

sample_id=pd.read_csv("/home/ldap/ltoure/OmiVAE/Data/labels.csv", sep=",", index_col=0, dtype='str')

#################################################

all_cols_f32 = {col: np.float32 for col in sample_id}
expr_df = pd.read_csv('/home/ldap/ltoure/OmiVAE/Data/rnaseq.csv', sep=',', header=0, index_col=0, dtype=all_cols_f32)

sample_id=labels.index
 #"".join(str(x) for x in sample_id)
    # Loading label
label = pd.read_csv('/home/ldap/ltoure/OmiVAE/Data/labels.csv', sep=',',index_col=0)
class_num = len(label.response.unique())
label_array = label['response'].values
if separate_testing:
        # Get testing set index and training set index
        # Separate according to different tumour types
   testset_ratio = 0.2
   valset_ratio = 0.5

   train_index, test_index, train_label, test_label = train_test_split(sample_id, label_array,
                                                                            test_size=testset_ratio,
                                                                            random_state=42,
                                                                            stratify=label_array)
   val_index, test_index, val_label, test_label = train_test_split(test_index, test_label, test_size=valset_ratio,
                                                                        random_state=42, stratify=test_label)
   expr_df_test = expr_df[test_index.index]
   expr_df_val = expr_df[val_index.index]
   expr_df_train = expr_df[train_index.index]
   
   methy_chr_df_test = methy_df_list[test_index.index]
   methy_chr_df_val = methy_df_list[val_index.index]
   methy_chr_df_train = methy_df_list[train_index.index]
   
   # Get multi-omics dataset information
   sample_num = len(sample_id)
   expr_feature_num = expr_df.shape[0]
   methy_feature_num = methy_df_list.shape[0]
   print('\nNumber of samples: {}'.format(sample_num))
   print('Number of gene expression features: {}'.format(expr_feature_num))
   print('Number of methylation features: {}'.format(methy_feature_num))
   if classifier:
       print('Number of classes: {}'.format(class_num))
    
   class MultiOmiDataset(Dataset):
       """
       Load multi-omics data
       """

       def __init__(self, expr_df, methy_df_list, labels):
           self.expr_df = rnaseq
           self.methy_df_list = DataExome
           self.labels = labels

       def __len__(self):
           return self.rnaseq.shape[1]

       def __getitem__(self, index):
           omics_data = []
           # Gene expression tensor index 0
           expr_line = self.rnaseq.iloc[:, index].values
           expr_line_tensor = torch.Tensor(expr_line)
           omics_data.append(expr_line_tensor)
           
           methy_chr_line = self.DataExome.iloc[:, index].values
           methy_chr_line_tensor = torch.Tensor(methy_chr_line)
           omics_data.append(methy_chr_line_tensor)
           label = self.labels[index]
           return [omics_data, label]
   
    # DataSets and DataLoaders
    if separate_testing:
        train_dataset = MultiOmiDataset(expr_df=expr_df_train, methy_df_list=methy_chr_df_train, labels=train_label)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        val_dataset = MultiOmiDataset(expr_df=expr_df_val, methy_df_list=methy_chr_df_val, labels=val_label)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        test_dataset = MultiOmiDataset(expr_df=expr_df_test, methy_df_list=methy_chr_df_test, labels=test_label)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    else:
        train_dataset = MultiOmiDataset(expr_df=expr_df, methy_df_list=methy_chr_df_list, labels=label_array)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    full_dataset = MultiOmiDataset(expr_df=expr_df, methy_df_list=methy_chr_df_list, labels=label_array)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, num_workers=6)




# Visualize the original and reconstructed images
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader



for batch_index, sample in enumerate(full_loader):
    data = sample[0]
    y = sample[1]
    data = data.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    z, recon_data, mean, log_var, pred_y = vae_model(data)
    
    # since shuffle=True, this is a random sample of test data
    for batch_index, sample in enumerate(test_loader):
        data = sample[0]
        data = data.to(device)
        
data1=torch.Tensor.numpy(data, force=True)
reconstruct=torch.Tensor.numpy(recon_data, force=True)
rec_err = np.linalg.norm(torch.Tensor.numpy(data, force=True) - torch.Tensor.numpy(recon_data, force=True), axis = 1)
idx = list(rec_err).index(max(rec_err))
df = pd.DataFrame(data = reconstruct[idx], index = rnaseq.columns, columns = ['reconstruction_loss'])
df.T

def sort_by_absolute(df, index):
    df_abs = df.apply(lambda x: abs(x))
    df_abs = df_abs.sort_values('reconstruction_loss', ascending = False)
    df = df.loc[df_abs.index,:]
    return df
sort_by_absolute(df, idx).T

top_5_features = sort_by_absolute(df, idx).iloc[:5,:]
top_5_features.T

data_summary = shap.kmedata1ns(data1, 100)

 # model weights are updated
    model_feature = vae_model.copy()
	model_feature.get_layer('hid_layer1').update_weights(weights_feature)
 ## determine the SHAP values
 explainer_autoencoder = shap.DeepExplainer(vae_model, data)
    shap_values = explainer_autoencoder.shap_values(X_standard.loc[idx,:].values)
e = shap.DeepExplainer(vae_model.to(device), Variable(data))

test_images = sample[100:103]
e = shap.DeepExplainer(model, Variable( torch.from_numpy( train_features_df.to_numpy(dtype=np.float32) ) ) )
import shap 
e = shap.DeepExplainer(
        vae_model.to(device), torch.tensor(background, dtype=torch.float, device=device))

vae_model.to(device)
e = shap.DeepExplainer(vae_model, data)
shap_values = e.shap_values(test_images)

d = {0: 'red', 1: "green"}

colors = []       
for e in label['response']:
    colors.append(d[e])

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)
ax.set_xlabel('Latent feature 1')
ax.set_ylabel('Latent feature 2')

ax.scatter(mean[:,0].detach().numpy(), mean[:,1].detach().numpy(), 
           c=list(colors))
plt.savefig('/home/ldap/ltoure/OmiVAE/results/OmiVAe_AE.png')
