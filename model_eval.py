from functools import partial
import pytorch_lightning as pl
from _modules.cleanEEGNet import cleanEEGNet
import _modules.params as p
from _modules.datamodule import EEGDataModule, EEGDataset
import torch
from scipy.stats import zscore
import numpy as np
import torchmetrics
import matplotlib.pyplot as plt

light_mod = cleanEEGNet().load_from_checkpoint("/home/mbrugnara/cleanEEGNet/best model/models-epoch=15-valid_loss=0.00-v1.ckpt").to(p.device)
 
print(light_mod.model)

f1_comp = torchmetrics.classification.f_beta.F1(num_classes = 1, average = 'macro')

light_mod.eval()
light_mod.freeze()
f1 = []
test_path = "/data/disk0/volkan/mbrugnara/test/"

tests = EEGDataset(test_path)

lin_range = np.arange(0, 1, 0.05)

for mu in lin_range:
    avg_f1 = []
    for test in tests:
        x, label = test
        output = torch.zeros(x.shape[1]).to(p.device) 
        shadow = torch.zeros(x.shape[1]).to(p.device)
        t_mu = torch.from_numpy(np.asarray(mu)).to(p.device)
        for i_e, epoch in enumerate(x):
            input  = torch.from_numpy(zscore(epoch.cpu()))
            input = input.to(p.device)
            input = input.view(1,1,input.shape[0],input.shape[1])
            partial_output = light_mod(input)
            output = (t_mu * partial_output + (1 - t_mu) * output)
        
        round_output = torch.round(torch.sigmoid(output))
        temp_f1 = f1_comp(round_output[0].cpu(), label[:,0].int())
        avg_f1.append(temp_f1)
    avg_f1 = sum(avg_f1)/len(avg_f1)
    print(mu,avg_f1) 
    f1.append(avg_f1)

plt.plot(lin_range,f1)
plt.savefig('f1_mu.png')
