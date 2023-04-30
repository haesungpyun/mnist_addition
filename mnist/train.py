import torch
from tqdm.auto import tqdm
from torch import nn
import wandb

from utils import calculate_loss_acc 


def train_epoch(epoch: int, 
                encoder: nn.Module,
                classifier:nn.Module,
                dataloader:torch.utils.data.DataLoader, 
                loss_function: nn.Module, 
                optimizer: torch.optim.Optimizer, 
                lr_scheduler,
                predict_method: str,
                label_dic = None,
               ):
    
    wandb.define_metric("Train/step")
    wandb.define_metric("Train/*", step_metric="Train/step")
    
    total_loss, total_acc = 0.0, 0.0
    
    with tqdm(enumerate(dataloader), desc=f"Training Epoch {epoch}", total=len(dataloader)) as train_bar:
        for tri, batch  in train_bar:
            
            encoder.train()
            classifier.train()
            optimizer.zero_grad()

            num1, num2, label = batch

            h1 = encoder(num1)
            h2 = encoder(num2)

            tr_loss, tr_acc = calculate_loss_acc(h1, h2, label, 
                                                classifier, 
                                                loss_function, 
                                                label_dic, 
                                                predict_method)
            
            tr_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += tr_loss.item()
            total_acc += tr_acc.item()
            
            train_bar.set_description(f"Train Step {tri} || Train ACC {tr_acc: .4f} | Train Loss {tr_loss.item(): .4f}")
            
            log_dict = {"Train/step": tri + epoch*len(dataloader),
                        "Train/Accuracy": tr_acc,
                        "Train/Loss": tr_loss}
                
            wandb.log(log_dict)
            
    return total_loss/len(dataloader), total_acc / len(dataloader)