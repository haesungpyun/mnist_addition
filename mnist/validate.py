import torch
from tqdm.auto import tqdm
from torch import nn
import wandb

from utils import calculate_loss_acc, accuracy

def valid_epoch(epoch: int, 
                encoder: nn.Module,
                classifier:nn.Module,
                dataloader:torch.utils.data.DataLoader, 
                loss_function: nn.Module, 
                predict_method: str,
                label_dic = None                
               ):
    
    wandb.define_metric("Valid/step")
    wandb.define_metric("Valid/*", step_metric="Valid/step")
        
    total_loss, total_acc = 0.0, 0.0
    
    with torch.no_grad():
        with tqdm(enumerate(dataloader), desc=f"Val Epoch {epoch}", total=len(dataloader)) as val_bar:
            for vli, batch  in val_bar:

                num1, num2, label = batch

                h1 = encoder(num1)
                h2 = encoder(num2)

                vl_loss, vl_acc = calculate_loss_acc(h1, h2, label,
                                                    classifier, 
                                                    loss_function, 
                                                    label_dic, 
                                                    predict_method)

                total_loss += vl_loss.item()
                total_acc += vl_acc.item()

                val_bar.set_description(f"Val Step {vli} || Val ACC {vl_acc: .4f} | Val Loss {vl_loss: .4f}")
                
                log_dict = {"Valid/step": vli + epoch*len(dataloader),
                            "Valid/Accuracy": vl_acc,
                            "Valid/Loss": vl_loss}
                
                wandb.log(log_dict)
                
    return total_loss/len(dataloader), total_acc / len(dataloader)               
           


def test(encoder: nn.Module,
        classifier:nn.Module,
        dataloader:torch.utils.data.DataLoader, 
        loss_function: nn.Module,
        predict_method: str,
        label_dic = None                
    ):
    
    wandb.define_metric("Test/step")
    wandb.define_metric("Test/*", step_metric="Test/step")
        
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        with tqdm(enumerate(dataloader), desc="Test", total=len(dataloader)) as test_bar:
            for tti, batch  in test_bar:

                num, _, label = batch

                hs = encoder(num)
                
                tt_loss, tt_acc = calculate_loss_acc(hs, hs, label,
                                                    classifier, 
                                                    loss_function, 
                                                    label_dic, 
                                                    predict_method,
                                                    True)

                total_loss += tt_loss.item()
                total_acc += tt_acc.item()
                
                test_bar.set_description(f"Test Step {tti} || Test ACC {tt_acc: .4f} | Test Loss {tt_loss: .4f}")
                
                log_dict = {"Test/step": tti,
                            "Test/Accuracy": tt_acc,
                            "Test/Loss": tt_loss}
                
                wandb.log(log_dict)

    return total_loss/len(dataloader), total_acc / len(dataloader) 