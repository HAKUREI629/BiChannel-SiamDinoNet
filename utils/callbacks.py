import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
#from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import Callback

class LossHistory():
    def __init__(self, log_dir, model=None, input_shape=None):
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = os.path.join(log_dir, "loss_" + str(time_str))
        self.losses     = []
        self.val_loss   = []
        self.val_acc    = []
        self.val_pre    = []
        self.val_rec    = []
        self.val_f1     = []

        
        os.makedirs(self.log_dir)
        #self.writer     = SummaryWriter(self.log_dir)
        # try:
        #     dummy_input     = torch.randn(2, 2, 3, input_shape[0], input_shape[1])
        #     self.writer.add_graph(model, dummy_input)
        # except:
        #     pass

    def append_loss(self, epoch, loss, val_loss, val_acc, val_pre, val_rec, val_f1):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.val_pre.append(val_pre)
        self.val_rec.append(val_rec)
        self.val_f1.append(val_f1)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_acc.txt"), 'a') as f:
            f.write(str(val_acc))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_pre.txt"), 'a') as f:
            f.write(str(val_pre))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_rec.txt"), 'a') as f:
            f.write(str(val_rec))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_f1.txt"), 'a') as f:
            f.write(str(val_f1))
            f.write("\n")

        #self.writer.add_scalar('loss', loss, epoch)
        #self.writer.add_scalar('val_loss', val_loss, epoch)
        #self.writer.add_scalar('val_acc', val_acc, epoch)
        #self.writer.add_scalar('val_pre', val_pre, epoch)
        #self.writer.add_scalar('val_rec', val_rec, epoch)
        #self.writer.add_scalar('val_f1', val_f1, epoch)
        self.loss_plot()
        self.acc_plot()
        self.pre_plot()
        self.rec_plot()
        self.f1_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")
    
    def acc_plot(self):
        iters = range(len(self.val_acc))

        plt.figure()
        plt.plot(iters, self.val_acc, 'coral', linewidth = 2, label='val accuracy')
        try:
            if len(self.val_acc) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.val_acc, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val accuracy')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_val_acc.png"))

        plt.cla()
        plt.close("all")
    
    def pre_plot(self):
        iters = range(len(self.val_pre))

        plt.figure()
        plt.plot(iters, self.val_pre, 'coral', linewidth = 2, label='val precision')
        try:
            if len(self.val_pre) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.val_pre, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val precision')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_val_pre.png"))

        plt.cla()
        plt.close("all")
    
    def rec_plot(self):
        iters = range(len(self.val_rec))

        plt.figure()
        plt.plot(iters, self.val_rec, 'coral', linewidth = 2, label='val recall')
        try:
            if len(self.val_rec) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.val_rec, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val recall')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_val_rec.png"))

        plt.cla()
        plt.close("all")
    
    def f1_plot(self):
        iters = range(len(self.val_f1))

        plt.figure()
        plt.plot(iters, self.val_f1, 'coral', linewidth = 2, label='val F1 Score')
        try:
            if len(self.val_f1) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.val_f1, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val F1 Score')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_val_f1.png"))

        plt.cla()
        plt.close("all")

class HistoryLogger(Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.loss_history = LossHistory(save_dir)
        self.metrics = []

    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        # 假设我们想保存训练损失和验证损失
        loss = metrics.get('train_loss', None)
        val_loss = metrics.get('val_loss', None)
        val_acc = metrics.get('val_acc', None)
        epoch_metrics = {'train_loss': loss, 'val_loss': val_loss}
        self.metrics.append(epoch_metrics)

        self.loss_history.append_loss(trainer.current_epoch, loss, val_loss, val_acc)
        
