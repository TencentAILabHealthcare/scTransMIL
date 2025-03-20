import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            # nn.Linear(L, D),
            # nn.Tanh()
            nn.Identity(),
            ]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            # nn.Linear(L, D),
            # nn.Tanh(),
            nn.Identity(),
            ]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        # print("x: ", x.size())
        a = self.attention_a(x)
        b = self.attention_b(x)
        # print("gate: ", b.size())
        
        # import sys
        # sys.exit("code0")
        
        A_raw = a.mul(b)
        A = self.attention_c(A_raw)  # N x n_classes
        
        # return A, x
        return A, A_raw

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class MIL_fc(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        # instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        # super().__init__()
        # self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        # size = self.size_dict[size_arg]
        # fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        # if gate:
        #     attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        # else:
        #     attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        # fc.append(attention_net)
        # self.attention_net = nn.Sequential(*fc)
        # self.classifiers = nn.Linear(size[1], n_classes)
        # instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        # self.instance_classifiers = nn.ModuleList(instance_classifiers)
        # self.k_sample = k_sample
        # self.instance_loss_fn = instance_loss_fn
        # self.n_classes = n_classes
        # self.subtyping = subtyping
        
        # instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=2000):
        # super().__init__()
        # self.size_dict = {"small": [embed_dim, 256, 256], "big": [embed_dim, 512, 512]}
        # size = self.size_dict[size_arg]
        # # fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        # if gate:
        #     attention_net = Attn_Net_Gated(L = size[0], D = size[0], dropout = dropout, n_classes = 1)
        # else:
        #     attention_net = Attn_Net(L = size[0], D = size[0], dropout = dropout, n_classes = 1)
        # # fc.append(attention_net)
        # self.attention_net = attention_net
        # self.classifiers = nn.Linear(size[0], n_classes)
        # instance_classifiers = [nn.Linear(size[0], 2) for i in range(n_classes)]
        # self.instance_classifiers = nn.ModuleList(instance_classifiers)
        # self.k_sample = k_sample
        # self.instance_loss_fn = instance_loss_fn
        # self.n_classes = n_classes
        # self.subtyping = subtyping
        
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=2000):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 256, 256], "big": [embed_dim, 512, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        # print("h: ", h.size())
        A, h = self.attention_net(h)  # NxK     
        # print(h.size())
        # print(A.size()) # A shape: N x class_num
        # print("模型内部: ", A)  
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        # print(A.size())

        if instance_eval:
            # total_inst_loss = 0.0
            # all_preds = []
            # all_targets = []
            # inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            # for i in range(len(self.instance_classifiers)):
            #     inst_label = inst_labels[i].item()
            #     classifier = self.instance_classifiers[i]
            #     if inst_label == 1: #in-the-class:
            #         instance_loss, preds, targets = self.inst_eval(A, h, classifier)
            #         all_preds.extend(preds.cpu().numpy())
            #         all_targets.extend(targets.cpu().numpy())
            #     else: #out-of-the-class
            #         if self.subtyping:
            #             instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
            #             all_preds.extend(preds.cpu().numpy())
            #             all_targets.extend(targets.cpu().numpy())
            #         else:
            #             continue
            #     total_inst_loss += instance_loss

            # if self.subtyping:
            #     total_inst_loss /= len(self.instance_classifiers)
            print("###################instance_eval#######################step1")
        
        # print("A: ", A.size())
        # print("h: ", h.size())
        
        top_instance_idx = torch.topk(A, 1, dim = 1)[1].view(1,)
        
        M = torch.index_select(h, dim=0, index=top_instance_idx)
        
        # M = torch.mm(A, h)
        
        # print("M: ", M.size())
        
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        
        # print("logits: ", logits.size())
        # print("Y_hat: ", Y_hat.size())
        # print("Y_prob: ", Y_prob.size())
        
        if instance_eval:
            # results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            # 'inst_preds': np.array(all_preds)}
            print("###################instance_eval#######################step2")
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class MIL_fc_mc(MIL_fc):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        nn.Module.__init__(self)
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h) 

        logits = torch.empty(1, self.n_classes).float().to(M.device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

class MIL_fc_old(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1,
                 embed_dim=1024):
        super().__init__()
        assert n_classes == 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifier=  nn.Linear(size[1], n_classes)
        self.top_k=top_k

    def forward(self, h, return_features=False):
        h = self.fc(h)
        logits  = self.classifier(h) # K x 2
        
        y_probs = F.softmax(logits, dim = 1)
        y_probs_return = torch.transpose(F.softmax(logits, dim = 1)[:,1].view(-1,1), 1, 0)
        
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs_return, results_dict


class MIL_fc_mc_old(nn.Module):
    def __init__(self, size_arg = "small", dropout = 0., n_classes = 2, top_k=1, embed_dim=1024):
        super().__init__()
        assert n_classes > 2
        self.size_dict = {"small": [embed_dim, 512]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1
    
    def forward(self, h, return_features=False):       
        h = self.fc(h)
        logits = self.classifiers(h)

        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]

        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]
        
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


        
