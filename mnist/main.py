import torch
from tqdm.auto import tqdm
from transformers import get_scheduler
import wandb

from dataset import get_loader
from model import (MNIST_Encoder, Classifier, get_models)
from utils import get_loss_function
from train import train_epoch
from validate import valid_epoch, test


def main(num_epoch, batch_size, predict_method, lr, weight_decay):
    
    train_loader, valid_loader, test_loader = get_loader(batch_size)
    
    loss_function, label_dic = get_loss_function(predict_method)
    
    encoder, classifier = get_models(predict_method)

    wandb.watch((encoder, classifier))
    
    optimizer = torch.optim.AdamW(params=[{"params":encoder.parameters(), "params":classifier.parameters()}],
                                           lr=lr, 
                                           weight_decay=weight_decay
                                 )
    
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, 
                                 num_warmup_steps=int(len(train_loader)*num_epoch*0.1),
                                 num_training_steps=len(train_loader)*num_epoch
                                )
                                           
    with tqdm(range(num_epoch), desc="Total Epoch", total=num_epoch) as total_bar:
    
        for epoch in total_bar:
            
            train_loss, train_acc = train_epoch(epoch, 
                                                encoder, 
                                                classifier, 
                                                train_loader, 
                                                loss_function, 
                                                optimizer, 
                                                lr_scheduler, 
                                                predict_method,
                                                label_dic)
            
            valid_loss, valid_acc = valid_epoch(epoch, 
                                                encoder,
                                                classifier,
                                                valid_loader,
                                                loss_function,
                                                predict_method,
                                                label_dic)
            
            test_loss, test_acc = test(encoder,
                                        classifier,
                                        test_loader,
                                        loss_function,
                                        predict_method,
                                        label_dic)
                            
            total_bar.set_description(f"Epoch {epoch} |||| Train ACC {train_acc:.4f} \
                                        Train Epoch Loss {train_loss:.4f} || \
                                        Valid Epoch ACC {valid_acc:.4f} \
                                        Valid Epoch Loss {valid_loss:.4f}")
            
            wandb.log({"Epoch/Epoch": epoch,
                       "Total_ACC/Train Epoch ACC": train_acc,
                       "Total_Loss/Train Epoch Loss": train_loss,
                       "Total_ACC/Valid Epoch ACC ": valid_acc,
                       "Total_Loss/Valid Epoch Loss": valid_loss,
                       "Total_ACC/Test Epoch ACC ": test_acc,
                       "Total_Loss/Test Epoch Loss": test_loss,
                        })

        test_loss, test_acc = test(encoder,
                                    classifier,
                                    test_loader,
                                    loss_function,
                                    predict_method,
                                    label_dic)
        
        wandb.log({"Total_ACC/Test Accuracy": test_acc,
                    "Total_Loss/Test Loss": test_loss})
        
            
if __name__ == "__main__":
    import json
    from random import random
    
    with open('./config.json') as f:
        config = json.load(f)
    
    num_epoch = config.get('num_epoch')
    batch_size = config.get('batch_size')
    predict_method = config.get('predict_method')     # ['mean_hs', 'add_hs', 'add_logits', 'combination', 'inverse_augment', 'concat_hs']
    lr =config.get('lr')
    weight_decay = config.get('weight_decay')
    
    print(config)
    
    wandb.init(project="MNIST addition", config=config, id=predict_method+str(random()))
    
    # wandb.config = {"epoch": num_epoch, 
    #                 'batch_size': batch_size, 
    #                 'learning_rate': lr, 
    #                 'weight_decay': weight_decay
    #                }
    
    main(num_epoch=num_epoch, batch_size=batch_size, predict_method=predict_method, lr=lr, weight_decay=weight_decay)