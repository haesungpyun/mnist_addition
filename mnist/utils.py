import torch
import pandas as pd
import itertools
from collections import defaultdict
from torch import nn


def get_loss_function(predict_method):
    if predict_method == "inverse_augment":
        """  Make label dictionary storing all permutations corresponding to summation"""

        num = list(range(10))   
        num = num * 2
        per_list = sorted((set(itertools.permutations(num, 2, ))), key=lambda x: x)

        assert len(per_list) == 100

        label_dic = defaultdict(list)
        for i in per_list:
            label_dic[sum(i)].append(i)

        label_dic = pd.DataFrame(data=label_dic.values(), index=label_dic.keys()).T

        assert 100 == (~label_dic.isna()).sum().sum()

        def custom_loss(pred, label):

            return nn.NLLLoss(reduction="none")(nn.LogSoftmax(dim=-1)(pred), label)

        loss_function = custom_loss
        
        return loss_function, label_dic
    else: 
        loss_function = nn.CrossEntropyLoss()
    
        return loss_function, None
    

def accuracy(pred, target, denom=1):
    pred_num = torch.argmax(pred, dim=-1).float() / denom
    return (pred_num == target).float().mean()

def accuracy_sum(pred1,pred2, target):
    pred_num1 = torch.argmax(pred1, dim=-1)
    pred_num2 = torch.argmax(pred2,dim=-1)
    pred_sum = pred_num1 + pred_num2
    return (pred_sum == target).float().mean()   
    
    
def calculate_loss_acc(h1, h2, label, classifier, loss_function, label_dic, predict_method: str = "add_hs", test=False):
    if predict_method == "combination":
        if test:
            logit = classifier(h1)

            loss = loss_function(logit, label).mean()

            acc = accuracy(logit, label)

            return loss, acc    
        
        logit1 = classifier(h1)
        logit2 = classifier(h2)
        
        prob1 = nn.Softmax(dim=-1)(logit1).unsqueeze(-1)
        prob2 = nn.Softmax(dim=-1)(logit2).unsqueeze(1)
        
        batch_matrix = prob1 @ prob2
    
        anti_diag = [
            torch.stack(
                        [torch.sum(torch.diag(torch.fliplr(mat), diag)) 
                         for diag in range(len(mat)-1, -len(mat), -1)
                        ]
                        ) 
                for mat in batch_matrix ]
        
        batch_preds = torch.stack(anti_diag)
        
        loss = loss_function(batch_preds, label)
        
        acc = accuracy(batch_preds, label)
        
        return loss, acc
        
    elif predict_method == "add_hs":

        hidden = (h1 + h2)

        logit = classifier(hidden)
        
        loss = loss_function(logit, label)

        acc = accuracy(logit, label)        
    
        if test:
            acc = accuracy(logit, label, 2)        
            return loss, acc
        
        return loss, acc
    
    elif predict_method == "mean_hs":
        
        hidden = (h1 + h2) / 2

        logit = classifier(hidden)
        
        loss = loss_function(logit, label)

        acc = accuracy(logit, label)        
    
        if test:
            acc = accuracy(logit, label, 2)        
            return loss, acc
        
        return loss, acc
    
    elif predict_method == "concat_hs":
        # if test:
        #     label = label *2
        
        # hidden = (h1 + h2) / 2
        # hidden = torch.cat((h1, h2), -1)
        
        hidden = torch.cat((h1, h2), 1)
        hidden = hidden.view(-1, 16*4*4*2)

        logit = classifier(hidden)
        
        loss = loss_function(logit, label)

        acc = accuracy(logit, label)        
    
        if test:
            pred = torch.argmax(logit, dim=-1)
            pred = pred.float() / 2
            acc = (pred == label).float().mean()   
            return loss, acc
        
        return loss, acc
    
    elif predict_method == "add_logits":
        if test:
            label = label *2
        
        logit1 = classifier(h1)
        logit2 = classifier(h2)
        
        logit = (logit1 + logit2) / 2
        
        loss = loss_function(logit, label)

        acc = accuracy(logit, label)        

        return loss, acc
        
    elif predict_method == "inverse_augment":
        
        if test:
            logit = classifier(h1)
            
            loss = loss_function(logit, label).mean()
            
            acc = accuracy(logit, label)
            
            return loss, acc
        
        logit1 = classifier(h1)
        logit2 = classifier(h2)
        
        loss, acc = 0, 0
        
        assert label_dic is not None
        
        for lo1, lo2, (sum, combinations) in zip(logit1, logit2, label_dic[label].items()):

            label1 = torch.tensor(list(zip(*combinations.dropna().values))[0])
            label2 = torch.tensor(list(zip(*combinations.dropna().values))[1])
            
            lo1 = lo1.expand(label1.shape[0], -1)
            lo2 = lo2.expand(label2.shape[0], -1)
            
            loss += (loss_function(lo1, label1) + loss_function(lo2, label2)).mean()
            
            acc += accuracy_sum(lo1, lo2, sum)
            
        loss /= len(label)
        acc /= len(label)
        
        return loss, acc
        
    else:
        raise ValueError (f"Method only have 3 options : ['mean_hs', 'add_hs', 'add_logits', 'combination', 'inverse_augment', 'concat_hs'], but {predict_method} is given")