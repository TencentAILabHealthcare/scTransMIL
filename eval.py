from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *
import sys

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')

parser.add_argument('--fold_list', type=list, default=[0], help='single fold to evaluate')

parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--embed_dim', type=int, default=1024)
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(
                            csv_path = 'celldata_pancancer/tumor_vs_normal_dummy_clean_sub.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_2917_features_sub'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=29
    dataset = Generic_MIL_Dataset(csv_path = 'celldata_pancancer/tumor_subtyping_dummy_clean_sub.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_2917_features_sub'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'GBM': 0,
                                        'BRCA': 1,
                                        'LC': 2,
                                        'OV': 3,
                                        'PAAD': 4,
                                        'HNSC': 5,
                                        'UVM': 6,
                                        'NHL': 7,
                                        'SARC-OST': 8,
                                        'HCC': 9,
                                        'RCC': 10,
                                        'CRC': 11,
                                        'THCA': 12,
                                        'LGG': 13,
                                        'CHOL': 14,
                                        'SARC-SYN': 15,
                                        'ALL': 16,
                                        'NB': 17,
                                        'NET': 18,
                                        'SARC-RHAB': 19,
                                        'BLCA': 20,
                                        'PRAD': 21,
                                        'MM': 22,
                                        'MEL': 23,
                                        'SARC-EWING': 24,
                                        'UCEC': 25,
                                        'SSCC': 26,
                                        'STAD': 27,
                                        'CLL': 28
                                        },
                            patient_strat= False,
                            ignore=[])

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    # folds = range(start, end)
    folds = args.fold_list
    # print(folds)
    # sys.exit()
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    
    ckpt_idx = 0
    
    for ckpt_idx in range(len(ckpt_paths)):
        print("######################################################")
        print(args.split)
        
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        
        model, test_error, auc, df, instance_results  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}_{}.csv'.format(folds[ckpt_idx], args.split)), index=False)
        
        if datasets_id[args.split] < 0:
            print(len(list(instance_results.keys())))
            with open(os.path.join(args.save_dir, 'instance_results_fold_{}.pkl'.format(folds[ckpt_idx])), 'wb') as f:
                pickle.dump(instance_results, f)