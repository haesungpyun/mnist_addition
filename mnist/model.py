from torch import nn

class MNIST_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # nn.Conv2d(1, 6, 5),
            # nn.BatchNorm2d(6),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, 2), # 6 24 24 -> 6 12 12
            # nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8
            # nn.BatchNorm2d(16),
            # nn.ReLU(True),
            # nn.MaxPool2d(2, 2), # 16 8 8 -> 16 4 4
            
            # in_channels, out_channels, kernel_size
            nn.Conv2d(1, 6, 5), # 6 24 24 
            nn.MaxPool2d(2, 2), # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2), # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        )
        
        for m in self.encoder:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.encoder(x)
        # x = x.view(-1, 16 * 4 * 4) to compare with submitted code

        return x    
    
class Classifier(nn.Module):
    def __init__(self, N=10, input_dim=16*4*4):
        super().__init__()
        self.classifier =  nn.Sequential(
            nn.Linear(input_dim, 120),
            nn.ReLU(),
            nn.Linear(120, N)
        )
        
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        
    def forward(self, x):
        return self.classifier(x)    
 
   
def get_models(predict_method):
    encoder = MNIST_Encoder()
    
    if predict_method in ["inverse_augment", "combination"]:
        classifier = Classifier(N=10)
    
    elif predict_method in ['add_hs', 'add_logits', 'mean_hs']:
        classifier = Classifier(N=19)
        
    elif predict_method in ['concat_hs']:
        classifier = Classifier(N=19, input_dim=16*4*4*2)
        
    else:
        raise ValueError (f"Method only have 3 options : ['mean_hs', 'add_hs', 'add_logits', 'combination', 'inverse_augment', 'concat_hs'], but {predict_method} is given")
         
    return encoder, classifier