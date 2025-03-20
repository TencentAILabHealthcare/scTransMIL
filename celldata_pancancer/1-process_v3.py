# 把bag进一步拆分
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import torch
import pandas as pd
import scipy.sparse as sp
import pickle
from datasets import load_from_disk
import numpy as np
import sys

import warnings
warnings.filterwarnings("ignore")

TOKEN_DICTIONARY_FILE = "../models/geneformer/token_dictionary_gc95M.pkl"

def split_ids(ids, subset_size=256):
        subsets = []
        for i in range(0, len(ids), subset_size):
            subset = ids[i:i + subset_size]
            if len(subset) < subset_size:
                subset += ids[:subset_size - len(subset)]
            subsets.append(subset)
        return subsets

def preprocess_classifier_batch(cell_batch, max_len):
    if max_len is None:
        max_len = max([len(i) for i in cell_batch["input_ids"]])

    # load token dictionary (Ensembl IDs:token)
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        gene_token_dict = pickle.load(f)

    def pad_label_example(example):
        example["input_ids"] = np.pad(
            example["input_ids"],
            (0, max_len - len(example["input_ids"])),
            mode="constant",
            constant_values=gene_token_dict.get("<pad>"),
        )
        example["attention_mask"] = (
            example["input_ids"] != gene_token_dict.get("<pad>")
        ).astype(int)
        return example

    padded_batch = cell_batch.map(pad_label_example)
    return padded_batch


# TODO: step1-step3
flag = "step1"

if flag == "step1":
    # 生成分散的数据集
    result = pd.read_csv("./gene_id.csv")
    TOKEN_DICTIONARY_FILE = "../models/geneformer/token_dictionary_gc95M.pkl"
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        gene_token_dict = pickle.load(f)
    vocab = list(gene_token_dict.keys())
    
    directory_h5ad = "./tumor_vs_normal_2917_features_sub/h5ad_files"
    os.makedirs(directory_h5ad, exist_ok=True)
    
    folder_path = '/aaa/gelseywang/buddy1/gelseywang/MIL/Data/Pancancer/atlas_dataset/'
    file_paths = glob.glob(os.path.join(folder_path, '*.h5ad'))
    
    column_names = ["case_id", "slide_id", "label"]
    df = pd.DataFrame(columns=column_names)

    for file_path in file_paths:
        ad = sc.read_h5ad(file_path)
        gene_names = list(ad.var_names)
        ret_genes = result.loc[result['Gene name'].isin(gene_names), 'Gene name'].tolist()
        ad = ad[:, ret_genes].copy()
        id_list = result.loc[result['Gene name'].isin(ret_genes), 'Gene stable ID'].tolist()
        ad.var["ensembl_id"] = id_list
        ad.var.index = ad.var["ensembl_id"]
        intersection = list(set(vocab) & set(ad.var.index))
        ad = ad[:, intersection].copy()

        ad.obs["n_counts"] = ad.X.sum(axis=1)
        # ad.obs["joinid"] = list(range(ad.n_obs))
        print(ad)
        
        print(type(ad.X))
        print(ad.X.max())
        print(ad.X.min())
        # sys.exit()

        patient_list = ad.obs["Patient"].value_counts().index.tolist()
        
        for patient in patient_list:
            ad_patient = ad[ad.obs["Patient"] == patient,:]
            # print(ad_patient.obs["Patient"].value_counts())
            tissue_list = ad_patient.obs["Tissue"].value_counts().index.tolist()
            
            for tissue in tissue_list:
                ad_tissue = ad_patient[ad_patient.obs["Tissue"] == tissue,:]
                # if ad_tissue.obsm['X_pca'].shape[0] < 16:
                #     continue
                
                # 划分子集
                ad_tissue_ids = ad_tissue.obs_names.tolist()
                subsets_list = split_ids(ad_tissue_ids, subset_size=256)
                for idx, subset_ids in enumerate(subsets_list):
                    
                    ad_tissue_subset = ad_tissue[subset_ids,:].copy()
                    
                    directory_bag_h5ad = "./tumor_vs_normal_2917_features_sub/h5ad_files/{}_{}_{}".format(patient, tissue, str(idx))
                    os.makedirs(directory_bag_h5ad, exist_ok=True)
                    ad_tissue_subset.write("./tumor_vs_normal_2917_features_sub/h5ad_files/{}_{}_{}/bag.h5ad".format(patient, tissue, str(idx)))

                    if 'tumor' in ad_tissue_subset.obs["cnv_status"].value_counts().index.tolist():
                        df = df.append({"case_id": patient, "slide_id": "{}_{}_{}".format(patient, tissue, str(idx)), "label": "tumor_tissue"}, ignore_index=True)
                    else:
                        df = df.append({"case_id": patient, "slide_id": "{}_{}_{}".format(patient, tissue, str(idx)), "label": "normal_tissue"}, ignore_index=True)
                        
                    print("slide_id: {}_{}_{}".format(patient, tissue, str(idx)))
    
    df.to_csv("./tumor_vs_normal_dummy_clean.csv", index=False)
    
    # 生成geneformer的数据集
    import sys
    sys.path.append("..")
    from utils.tokenizer import TranscriptomeTokenizer
    
    root_h5ad_dir = "./tumor_vs_normal_2917_features_sub/h5ad_files"
    
    root_token_dir = "./tumor_vs_normal_2917_features_sub/tokenized_data"
    os.makedirs(root_token_dir, exist_ok=True)
    
    for entry in os.listdir(root_h5ad_dir):
        h5ad_path = os.path.join(root_h5ad_dir, entry)
        print(entry)
        # token化
        tokenizer = TranscriptomeTokenizer(custom_attr_name_dict={"Celltype": "Celltype", "Cancer type": "Cancer type", "cnv_status": "cnv_status"})
        tokenizer.tokenize_data(
            data_directory=h5ad_path,
            output_directory=root_token_dir,
            output_prefix=entry,
            file_format="h5ad",
        )

    # 生成scMIL的数据集
    directory_input_id = "./tumor_vs_normal_2917_features_sub/input_id_files"
    os.makedirs(directory_input_id, exist_ok=True)
    directory_input_attn = "./tumor_vs_normal_2917_features_sub/input_attn_files"
    os.makedirs(directory_input_attn, exist_ok=True)
    
    root_token_dir = "./tumor_vs_normal_2917_features_sub/tokenized_data"
    
    for entry in os.listdir(root_token_dir):
        token_path = os.path.join(root_token_dir, entry)
        print(entry[:-8])
        
        evalset=load_from_disk(token_path)
        max_evalset_len = max(evalset.select([i for i in range(len(evalset))])["length"])
        print(max_evalset_len)
        
        padded_batch = preprocess_classifier_batch(evalset, max_evalset_len)

        max_len = 1024
        input_data_batch = torch.tensor(padded_batch["input_ids"])[:, :max_len]
        attn_msk_batch = torch.tensor(padded_batch["attention_mask"])[:, :max_len]
        
        print(input_data_batch.size())
        torch.save(input_data_batch, "./tumor_vs_normal_2917_features_sub/input_id_files/{}.pt".format(entry[:-8]))
        print(attn_msk_batch.size())
        torch.save(attn_msk_batch, "./tumor_vs_normal_2917_features_sub/input_attn_files/{}.pt".format(entry[:-8]))

elif flag == "step2":
    # 过滤样本不足的bag
    del_list = []
    root_id_dir = "./tumor_vs_normal_2917_features_sub/input_id_files"
    for entry in os.listdir(root_id_dir):
        id_path = os.path.join(root_id_dir, entry)
        input_data = torch.load(id_path)
        if input_data.size(0) < 17 :
            # print(id_path)
            del_list.append(entry[:-3])
            
    print(len(del_list))
    print(del_list)
    
    df = pd.read_csv("./tumor_vs_normal_dummy_clean.csv")
    filtered_df = df[~df['slide_id'].isin(del_list)]
    filtered_df.to_csv("./tumor_vs_normal_dummy_clean_sub.csv", index=False)
    
elif flag == "step3":
    # 生成多分类数据集
    tumor_normal_df = pd.read_csv("./tumor_vs_normal_dummy_clean_sub.csv")
    tumor_df = tumor_normal_df[tumor_normal_df["label"] == "tumor_tissue"]
    slide_ids = tumor_df["slide_id"].tolist()
    
    root_h5ad_dir = "./tumor_vs_normal_2917_features_sub/h5ad_files"
    
    column_names = ["case_id", "slide_id", "label"]
    df = pd.DataFrame(columns=column_names)
    
    for slide_id in slide_ids:
        h5ad_path = os.path.join(root_h5ad_dir, slide_id + "/bag.h5ad")
        adata = sc.read(h5ad_path)
        adata = adata[adata.obs["cnv_status"] == "tumor",:]
        case_id = tumor_df[tumor_df["slide_id"] == slide_id]["case_id"].tolist()[0]
        df = df.append({"case_id": case_id, "slide_id": slide_id, "label": adata.obs["Cancer type"].value_counts().index.tolist()[0]}, ignore_index=True)
        
    df.to_csv("./tumor_subtyping_dummy_clean_sub.csv", index=False)
    