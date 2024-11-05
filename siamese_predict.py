import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

import argparse

from nets.siamese_dino import SiameseDinoResNet, SiameseDinoTFAResNet, SiameseDinoTFAResNetV2

def mean2(x):
    y = np.sum(x) / np.size(x)
    return y
 
def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)
 
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r


class SiameseV2(object):
    _defaults = {
        "model_path"        : '/wangyunhao/cds/wangyunhao/code/Siamese-pytorch-allvalid/logs/melfbank_10s_50ms_100_8khz_50overlap_wav_dinotfav2_save_2024_04_19_12_27_02/best_acc0.8831098474973931_epoch_weights.pth',
        "input_shape"       : [399, 300],
        "letterbox_image"   : False,
        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Siamese
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
    
    def change_model(self, path):
        self.model_path = path
        self.generate()
    
    def generate(self):
        #---------------------------#
        #   载入模型与权值
        #---------------------------#
        print('Loading weights into state dict...')
        device  = torch.device('cuda' if self.cuda else 'cpu')
        model   = SiameseDinoTFAResNetV2(self.input_shape)
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(self.model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        self.net = model.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = self.net.cuda()

    def detect_image(self, image_1, image_2):

        image_1 = np.load(image_1)
        image_2 = np.load(image_2)

        
        photo_1  = np.array(image_1, np.float32)
        photo_2  = np.array(image_2, np.float32)

        with torch.no_grad():

            photo_1 = torch.from_numpy(np.expand_dims(photo_1, 0)).type(torch.FloatTensor)
            photo_2 = torch.from_numpy(np.expand_dims(photo_2, 0)).type(torch.FloatTensor)

            #photo_1 = photo_1.unsqueeze(0)
            #photo_2 = photo_2.unsqueeze(0)
            data = np.stack((photo_1, photo_2), axis=1)
            data = torch.as_tensor(data, dtype=torch.float)
            #data = data.unsqueeze(0)
            
            photo_1 = photo_1.unsqueeze(0)
            photo_2 = photo_2.unsqueeze(0)
            a = photo_1, 1
            base1  = self.base_net(a)
            a = photo_2, 1
            base2  = self.base_net(a)
            #print(base2[0].shape)
            
            base1 = base1[0].detach().cpu().squeeze(0).numpy()
            base2 = base2[0].detach().cpu().squeeze(0).numpy()
            
            if self.cuda:
                data = data.cuda()
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()
                

            output = self.net(data, data)[0]
            output = torch.nn.Sigmoid()(output)

        return output

    def detect_image_snr(self, image_1, image_2, snr):

        image_1 = np.load(image_1)
        image_2 = np.load(image_2)
        
        image_1, _ = add_noise(image_1, snr)
        image_2, _ = add_noise(image_2, snr)


        photo_1  = np.array(image_1, np.float32)
        photo_2  = np.array(image_2, np.float32)

        with torch.no_grad():

            photo_1 = torch.from_numpy(np.expand_dims(photo_1, 0)).type(torch.FloatTensor)
            photo_2 = torch.from_numpy(np.expand_dims(photo_2, 0)).type(torch.FloatTensor)

            #photo_1 = photo_1.unsqueeze(0)
            #photo_2 = photo_2.unsqueeze(0)
            data = np.stack((photo_1, photo_2), axis=1)
            data = torch.as_tensor(data, dtype=torch.float)
            #data = data.unsqueeze(0)
            
            photo_1 = photo_1.unsqueeze(0)
            photo_2 = photo_2.unsqueeze(0)
            a = photo_1, 1
            base1  = self.base_net(a)
            a = photo_2, 1
            base2  = self.base_net(a)
            #print(base2[0].shape)
            
            base1 = base1[0].detach().cpu().squeeze(0).numpy()
            base2 = base2[0].detach().cpu().squeeze(0).numpy()

            
            if self.cuda:
                data = data.cuda()
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            output = self.net(data, data)[0]
            output = torch.nn.Sigmoid()(output)

        return output

def awgn(signal, snr_dB):
    signal_power = np.mean(np.abs(signal) ** 2)  
    snr_linear = 10 ** (snr_dB / 10)  
    noise_power = signal_power / snr_linear  
    noise = np.random.randn(*signal.shape) * np.sqrt(noise_power)  
    noisy_signal = signal + noise  
    return noisy_signal

def add_noise(t_angle, snr):
    if snr is not None:
        db = snr
    else:
        db = 30 + 20 * np.random.rand()

    t_angle_noise = awgn(t_angle, db)
    eps = t_angle_noise - t_angle


    return t_angle_noise, eps

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Input Feature')
    parser.add_argument('--feature1', type=str, default=None, help='feature1 path, should be npy')
    parser.add_argument('--feature2', type=str, default=None, help='feature2 path, should be npy')
    args = parser.parse_args()

    model = SiameseV2()
    model.generate()

    feature1 = args.feature1
    feature2 = args.feature2
    model.detect_image(feature1, feature2)