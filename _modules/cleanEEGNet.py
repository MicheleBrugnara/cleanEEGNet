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
        self.num_features = 62
        self.rnn = torch.nn.GRU(62,self.num_features,batch_first = True)


    def forward(self, x):
        conv_output = torch.zeros(x.shape[0],x.shape[1],x.shape[2]).to(p.device) # (n_batches, n_epochs ,n_channels)
        h = torch.rand(1,x.shape[0],self.num_features).to(p.device)
        for i_b, batch in enumerate(x):
            for i_e, epoch in enumerate(batch):
                conv_output[i_b,i_e,:] = self.model.forward(epoch.view(1,1,epoch.shape[0],epoch.shape[1]))

        output, hn = self.rnn(conv_output,h)
        #print(hn,hn.shape)
        return hn[0,:,:]
    
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
        print("output: ", torch.sigmoid(output), "labels: ", label)

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