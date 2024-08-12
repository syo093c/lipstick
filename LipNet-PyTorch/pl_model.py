from ema import EMAOptimizer
from torch import nn
from torch import optim
from transformers import get_cosine_schedule_with_warmup
from transformers import get_polynomial_decay_schedule_with_warmup
import lightning as L
from lightning.pytorch.utilities.grads import grad_norm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import gc
import ipdb

from dataset import MyDataset
def ctc_decode(y):
    result = []
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]

class WrapperModel(L.LightningModule):
    def __init__(self, model, loss,learning_rate=3e-5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss=loss

        # for score calculation
        self.train_pred=[]
        self.val_pred=[]
        self.train_label=[]
        self.val_label=[]


    def forward(self, input):
        output = self.model(input)
        return output

    def training_step(self, i):
        vid = i.get('vid')
        txt = i.get('txt')
        vid_len = i.get('vid_len')
        txt_len = i.get('txt_len')
        y = self.forward(vid)
        loss = self.loss(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
        self.log("train/loss", loss)

        pred_txt = ctc_decode(y)
        truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]

        self.train_pred.append(pred_txt)
        self.train_label.append(truth_txt)
        
        return loss

    def on_before_optimizer_step(self, optimizer):
        # log grad norm
        grad_norm_dict=grad_norm(self,norm_type=2)
        grad=grad_norm_dict['grad_2.0_norm_total']
        self.log("train/grad_norm",grad)

    def configure_optimizers(self):
        train_steps = self.trainer.max_steps # note: set at trainer init
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate,betas=(0.9, 0.999), weight_decay=0.05)
        optimizer= EMAOptimizer(optimizer=optimizer,device=torch.device('cuda'))
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            #num_warmup_steps=int(train_steps* 0.03/self.trainer.accumulate_grad_batches),
            num_warmup_steps=int(100),
            num_training_steps=int(train_steps/self.trainer.accumulate_grad_batches),
        )
        return [optimizer], [
            {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        ]

    def validation_step(self, i):
        #input = i["data"]
        #label = i["label"]
        #output = self.forward(input)

        #loss = self.loss(input=output, target=label)
        #self.log("valid/loss", loss)

        #self.val_pred.append(output.detach().cpu())
        #self.val_label.append(label.detach().cpu())

        vid = i.get('vid')
        txt = i.get('txt')
        vid_len = i.get('vid_len')
        txt_len = i.get('txt_len')
        y = self.forward(vid)
        loss = torch.tensor(self.loss(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy())
        self.log("valid/loss", loss)
        pred_txt = ctc_decode(y)
        truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]

        self.val_pred.append(pred_txt)
        self.val_label.append(truth_txt)
    
    def on_train_epoch_end(self):
        print(self.train_pred[:5])
        print(self.train_label[:5])
        
        wer = []
        cer = []
        for i in range(len(self.train_label)):
            wer.extend(MyDataset.wer(self.train_pred[i], self.train_label[i])) 
            cer.extend(MyDataset.cer(self.train_pred[i], self.train_label[i])) 
        wer_score = torch.tensor(np.array(wer).mean())
        cer_score = torch.tensor(np.array(cer).mean())

        self.train_pred.clear()
        self.train_label.clear()

        gc.collect()
        self.log("score/wer_score", wer_score)
        self.log("score/cer_score", cer_score)

    def on_validation_epoch_end(self):
        print(self.val_pred[:5])
        print(self.val_label[:5])
        
        wer = []
        cer = []
        for i in range(len(self.val_label)):
            wer.extend(MyDataset.wer(self.val_pred[i], self.val_label[i])) 
            cer.extend(MyDataset.cer(self.val_pred[i], self.val_label[i])) 
        wer_score = torch.tensor(np.array(wer).mean())
        cer_score = torch.tensor(np.array(cer).mean())

        self.val_pred.clear()
        self.val_label.clear()

        gc.collect()
        self.log("score/wer_score", wer_score)
        self.log("score/cer_score", cer_score)