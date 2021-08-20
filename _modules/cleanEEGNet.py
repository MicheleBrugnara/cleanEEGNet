from pytorch_lightning import LightningModule
from _modules.CNN_chocolate import ConvNet
from loss import custom_loss
import torchmetrics
import params as p
import torch
from torch import nn
from torch.autograd import Variable


class cleanEEGNet(LightningModule):
    def __init__(self):
        
        super().__init__()
        self.f1 = torchmetrics.classification.f_beta.F1()
        self.model = ConvNet()
        #self.mu = Variable(torch.rand(1,dtype=float).to(p.device), requires_grad=True)

    def forward(self, x):
        output = torch.zeros(x.shape[0],x.shape[2]).to(p.device) # (n_batches, n_channels)
        shadow = torch.zeros(x.shape[2]).to(p.device)
        for i_b, batch in enumerate(x):
            for i_e, epoch in enumerate(batch):
                output[i_b,:] += self.model.forward(epoch.view(1,1,epoch.shape[0],epoch.shape[1])) 
                
        output /= x.shape[1]
        return output
    
    def loss_fn(self, y_hat, y_target):
        loss = custom_loss()
        return loss(y_hat, y_target)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                    lr=p.lr,
                                    weight_decay=p.weigth_decay)

        return optimizer

    def training_step(self, batch, batch_idx):
        x, label = batch
        label = label[:,:,1]
        output = self(x.float())       
        loss = self.loss_fn(output, label.int())
        #print("output: ", torch.sigmoid(output), "labels: ", label)

        pred = torch.round(torch.sigmoid(output))
        f1 = self.f1(torch.flatten(pred), torch.flatten(label).int())

        self.log('train_loss', loss)
        self.log('train_f1', f1)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        label = label[:,:,1]
        output = self(x.float())

        loss = self.loss_fn(output, label.int())

        pred = torch.round(torch.sigmoid(output))
        f1 = self.f1(torch.flatten(pred), torch.flatten(label).int())

        self.log('val_loss', loss)
        self.log('val_f1', f1)

        return loss, f1

    def test_step(self, batch, batch_idx):
        x, label = batch
        label = label[:,:,1]
        output = self(x.float())

        loss = self.loss_fn(output, label.int())

        pred = torch.round(torch.sigmoid(output))
        f1 = self.f1(torch.flatten(pred), torch.flatten(label).int())

        self.log('val_loss', loss)
        self.log('val_f1', f1)

        return loss, f1