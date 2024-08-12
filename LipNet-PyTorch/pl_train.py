from pl_model import WrapperModel
import lightning as L

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

from dataset import MyDataset

from model import LipNet
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichModelSummary
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger

import os
import random
import numpy as np

def main():
    opt = __import__('options')
    train_dataset = MyDataset(opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train')
    val_dataset = MyDataset(opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'test')
    train_dataloader=DataLoader(dataset=train_dataset,batch_size=64,num_workers=10)
    val_dataloader=DataLoader(dataset=val_dataset,batch_size=64,num_workers=10)
    crit = nn.CTCLoss()

    loss_checkpoint_callback = ModelCheckpoint(
        verbose=True,
        filename=f"val_loss-" + "epoch_{epoch}-val_loss_{valid/loss:.4f}-wer_{score/wer_score:.4f}",
        monitor="valid/loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        save_weights_only=True,
        auto_insert_metric_name=False,
    )

    debug=False
    ep=int(400)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=3)

    if not debug:
        #logger = WandbLogger(project="lipstick", name="exp001")
        logger = WandbLogger(project="lipstick")
    else:
        logger = TensorBoardLogger("lipstick",name='exp001')
    
    
    model = LipNet()
    model = model.cuda()
    wrapper_model=WrapperModel(model=model,loss=crit)

    trainer = L.Trainer(max_epochs=ep, max_steps=ep*len(train_dataloader),precision="32-true", logger=logger, callbacks=[lr_monitor,loss_checkpoint_callback],log_every_n_steps=10,accumulate_grad_batches=1,gradient_clip_val=1)
    
    trainer.fit(model=wrapper_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    #torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    seed_everything(42)
    main()
