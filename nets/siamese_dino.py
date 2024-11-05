import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import wraps
from nets.resnet_dino import custom_resnet18, custom_resnet18v2

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average_1(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class SiameseDinoResNet(nn.Module):
    def __init__(self, input_shape, pretrained=False, moving_average_decay=0.99):
        super(SiameseDinoResNet, self).__init__()
        # online_model: student, target_model: teacher
        self.online_model = models.resnet18(pretrained=False)
        self.online_model.load_state_dict(torch.load('resnet_pth/resnet18-5c106cde.pth'))
        self.online_model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.online_model.fc = nn.Linear(in_features=512, out_features=256)

        self.target_model = None

        self.online_predicter = nn.Sequential(nn.Linear(in_features=256, out_features=256), nn.ReLU(), 
                                              nn.BatchNorm1d(num_features=256), nn.Linear(in_features=256, out_features=256))
        
        #self.online_projector = nn.Sequential(nn.Linear(in_features=256, out_features=256), nn.ReLU(), 
        #                                      nn.BatchNorm1d(num_features=256), nn.Linear(in_features=256, out_features=256))
        
        self.target_predicter = None

        self.target_ema_updater = EMA(moving_average_decay)
        
        #flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(256, 256)
        self.fully_connect2 = torch.nn.Linear(256, 1)

        self.use_momentum = True

    @singleton('target_model')
    def _get_target_encoder(self):
        target_model = copy.deepcopy(self.online_model)
        set_requires_grad(target_model, False)
        return target_model

    def reset_moving_average(self):
        del self.target_model
        self.target_model = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_model is not None, 'target encoder has not been created yet'
        update_moving_average_1(self.target_ema_updater, self.target_model, self.online_model)
    
    @singleton('target_predicter')
    def _get_target_predicter(self):
        target_predicter = copy.deepcopy(self.online_predicter)
        set_requires_grad(target_predicter, False)
        return target_predicter

    def reset_moving_average_p(self):
        del self.target_predicter
        self.target_predicter = None

    def update_moving_average_p(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_predicter is not None, 'target encoder has not been created yet'
        update_moving_average_1(self.target_ema_updater, self.target_predicter, self.online_predicter)


    def forward(self, images_ori, crop_images_ori):

        images      = self.online_model(images_ori)
        crop_images = self.online_model(crop_images_ori)   

        images      = self.online_predicter(images)
        crop_images = self.online_predicter(crop_images)
        
        #x1 = self.online_projector(x1)
        #x2 = self.online_projector(x2)

        with torch.no_grad():
            target_model = self._get_target_encoder() if self.use_momentum else self.online_model
            target_predicter = self._get_target_predicter() if self.use_momentum else self.online_predicter
            target_proj_one = target_model(images_ori)
            # target_proj_two = target_model(x2_ori)
            target_proj_one = target_predicter(target_proj_one)
            # target_proj_two = target_predicter(target_proj_two)
            target_proj_one.detach_()
            # target_proj_two.detach_()
        
        images      = (images + target_proj_one.detach()) / 2
        images      = self.fully_connect1(images)
        out_img     = self.fully_connect2(images)

        crop_images = self.fully_connect1(crop_images)
        out_cropimg = self.fully_connect2(crop_images)
    
        # x1 = torch.flatten(x1, 1)
        # x2 = torch.flatten(x2, 1)

        # target_proj_one = torch.flatten(target_proj_one, 1)
        # target_proj_two = torch.flatten(target_proj_two, 1)

        # diff1 = torch.abs(x1 - target_proj_two.detach())
        # diff2 = torch.abs(x2 - target_proj_one.detach())
        # # x = torch.abs(x1 - x2)
        # x = diff1 + diff2

        # x = self.fully_connect1(x)
        # x = self.fully_connect2(x)

        return out_img, out_cropimg

class SiameseDinoTFAResNet(nn.Module):
    def __init__(self, input_shape, pretrained=False, moving_average_decay=0.99):
        super(SiameseDinoTFAResNet, self).__init__()
        # online_model为student，target_model为teacher
        self.online_model = custom_resnet18v2(pretrained=True)
        # self.online_model.load_state_dict(torch.load('resnet_pth/resnet18-5c106cde.pth'))
        self.online_model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.online_model.fc = nn.Linear(in_features=512, out_features=256)

        self.target_model = None

        self.online_predicter = nn.Sequential(nn.Linear(in_features=256, out_features=256), nn.ReLU(), 
                                              nn.BatchNorm1d(num_features=256), nn.Linear(in_features=256, out_features=256))
        
        #self.online_projector = nn.Sequential(nn.Linear(in_features=256, out_features=256), nn.ReLU(), 
        #                                      nn.BatchNorm1d(num_features=256), nn.Linear(in_features=256, out_features=256))
        
        self.target_predicter = None

        self.target_ema_updater = EMA(moving_average_decay)
        
        #flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(256, 256)
        self.fully_connect2 = torch.nn.Linear(256, 1)

        self.use_momentum = True

    @singleton('target_model')
    def _get_target_encoder(self):
        target_model = copy.deepcopy(self.online_model)
        set_requires_grad(target_model, False)
        return target_model

    def reset_moving_average(self):
        del self.target_model
        self.target_model = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_model is not None, 'target encoder has not been created yet'
        update_moving_average_1(self.target_ema_updater, self.target_model, self.online_model)
    
    @singleton('target_predicter')
    def _get_target_predicter(self):
        target_predicter = copy.deepcopy(self.online_predicter)
        set_requires_grad(target_predicter, False)
        return target_predicter

    def reset_moving_average_p(self):
        del self.target_predicter
        self.target_predicter = None

    def update_moving_average_p(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_predicter is not None, 'target encoder has not been created yet'
        update_moving_average_1(self.target_ema_updater, self.target_predicter, self.online_predicter)


    def forward(self, images_ori, crop_images_ori):
        images      = self.online_model(images_ori)
        crop_images = self.online_model(crop_images_ori)   

        images      = self.online_predicter(images)
        crop_images = self.online_predicter(crop_images)
        
        #x1 = self.online_projector(x1)
        #x2 = self.online_projector(x2)

        with torch.no_grad():
            target_model = self._get_target_encoder() if self.use_momentum else self.online_model
            target_predicter = self._get_target_predicter() if self.use_momentum else self.online_predicter
            target_proj_one = target_model(images_ori)
            # target_proj_two = target_model(x2_ori)
            target_proj_one = target_predicter(target_proj_one)
            # target_proj_two = target_predicter(target_proj_two)
            target_proj_one.detach_()
            # target_proj_two.detach_()
        
        images      = (images + target_proj_one.detach()) / 2
        images      = self.fully_connect1(images)
        out_img     = self.fully_connect2(images)

        crop_images = self.fully_connect1(crop_images)
        out_cropimg = self.fully_connect2(crop_images)   
        # x1 = torch.flatten(x1, 1)
        # x2 = torch.flatten(x2, 1)

        # target_proj_one = torch.flatten(target_proj_one, 1)
        # target_proj_two = torch.flatten(target_proj_two, 1)

        # diff1 = torch.abs(x1 - target_proj_two.detach())
        # diff2 = torch.abs(x2 - target_proj_one.detach())
        # # x = torch.abs(x1 - x2)
        # x = diff1 + diff2
        # x = self.fully_connect1(x)
        # x = self.fully_connect2(x)

        return out_img, out_cropimg

class SiameseDinoTFAResNetV2(nn.Module):
    def __init__(self, input_shape, pretrained=False, moving_average_decay=0.99):
        super(SiameseDinoTFAResNetV2, self).__init__()
        self.online_model = custom_resnet18v2(pretrained=True)
        # self.online_model.load_state_dict(torch.load('resnet_pth/resnet18-5c106cde.pth'))
        self.online_model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.online_model.fc = nn.Linear(in_features=512, out_features=256)

        self.target_model = None

        self.online_predicter = nn.Sequential(nn.Linear(in_features=256, out_features=256), nn.ReLU(), 
                                              nn.BatchNorm1d(num_features=256), nn.Linear(in_features=256, out_features=256))
        
        #self.online_projector = nn.Sequential(nn.Linear(in_features=256, out_features=256), nn.ReLU(), 
        #                                      nn.BatchNorm1d(num_features=256), nn.Linear(in_features=256, out_features=256))
        
        self.target_predicter = None

        self.target_ema_updater = EMA(moving_average_decay)
        
        #flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(256, 256)
        self.fully_connect2 = torch.nn.Linear(256, 1)

        self.use_momentum = True

    @singleton('target_model')
    def _get_target_encoder(self):
        target_model = copy.deepcopy(self.online_model)
        set_requires_grad(target_model, False)
        return target_model

    def reset_moving_average(self):
        del self.target_model
        self.target_model = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_model is not None, 'target encoder has not been created yet'
        update_moving_average_1(self.target_ema_updater, self.target_model, self.online_model)
    
    @singleton('target_predicter')
    def _get_target_predicter(self):
        target_predicter = copy.deepcopy(self.online_predicter)
        set_requires_grad(target_predicter, False)
        return target_predicter

    def reset_moving_average_p(self):
        del self.target_predicter
        self.target_predicter = None

    def update_moving_average_p(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_predicter is not None, 'target encoder has not been created yet'
        update_moving_average_1(self.target_ema_updater, self.target_predicter, self.online_predicter)


    def forward(self, images_ori, crop_images_ori):

        images      = self.online_model(images_ori)
        images      = self.online_predicter(images)

        #x1 = self.online_projector(x1)
        #x2 = self.online_projector(x2)

        with torch.no_grad():
            target_model = self._get_target_encoder() if self.use_momentum else self.online_model
            target_predicter = self._get_target_predicter() if self.use_momentum else self.online_predicter
            target_proj_one = target_model(images_ori)
            target_proj_one = target_predicter(target_proj_one)
            target_proj_one = target_proj_one.detach_()
        
        images      = images
        images      = self.fully_connect1(images)
        out_img     = self.fully_connect2(images)

        crop_images = self.online_model(crop_images_ori)   
        crop_images1 = self.online_predicter(crop_images)
        crop_images = self.fully_connect1(crop_images1)
        out_cropimg = self.fully_connect2(crop_images)

        return out_img, out_cropimg, target_proj_one, crop_images1

class SiameseDinoTFAResNetV3(nn.Module):
    def __init__(self, input_shape, pretrained=False, moving_average_decay=0.99):
        super(SiameseDinoTFAResNetV3, self).__init__()
        self.online_model = custom_resnet18v2(pretrained=True)
        # self.online_model.load_state_dict(torch.load('resnet_pth/resnet18-5c106cde.pth'))
        self.online_model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.online_model.fc = nn.Linear(in_features=512, out_features=256)

        self.target_model = None

        self.online_predicter = nn.Sequential(nn.Linear(in_features=256, out_features=256), nn.ReLU(), 
                                              nn.BatchNorm1d(num_features=256), nn.Linear(in_features=256, out_features=256))
        
        #self.online_projector = nn.Sequential(nn.Linear(in_features=256, out_features=256), nn.ReLU(), 
        #                                      nn.BatchNorm1d(num_features=256), nn.Linear(in_features=256, out_features=256))
        
        self.target_predicter = None

        self.target_ema_updater = EMA(moving_average_decay)
        
        #flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(256, 256)
        self.fully_connect2 = torch.nn.Linear(256, 1)

        self.use_momentum = True

    @singleton('target_model')
    def _get_target_encoder(self):
        target_model = copy.deepcopy(self.online_model)
        set_requires_grad(target_model, False)
        return target_model

    def reset_moving_average(self):
        del self.target_model
        self.target_model = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_model is not None, 'target encoder has not been created yet'
        update_moving_average_1(self.target_ema_updater, self.target_model, self.online_model)
    
    @singleton('target_predicter')
    def _get_target_predicter(self):
        target_predicter = copy.deepcopy(self.online_predicter)
        set_requires_grad(target_predicter, False)
        return target_predicter

    def reset_moving_average_p(self):
        del self.target_predicter
        self.target_predicter = None

    def update_moving_average_p(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_predicter is not None, 'target encoder has not been created yet'
        update_moving_average_1(self.target_ema_updater, self.target_predicter, self.online_predicter)


    def forward(self, images_ori, crop_images_ori):
        images      = self.online_model(images_ori)
        images      = self.online_predicter(images)

        #x1 = self.online_projector(x1)
        #x2 = self.online_projector(x2)

        with torch.no_grad():
            target_model = self._get_target_encoder() if self.use_momentum else self.online_model
            target_predicter = self._get_target_predicter() if self.use_momentum else self.online_predicter
            target_proj_one = target_model(images_ori)
            # target_proj_two = target_model(x2_ori)
            target_proj_one = target_predicter(target_proj_one)
            # target_proj_two = target_predicter(target_proj_two)
            target_proj_one = target_proj_one.detach_()
            # target_proj_two.detach_()
            #teacher_out = self.fully_connect2(self.fully_connect1(target_proj_one))
            #teacher_out = teacher_out.detach_()
        
        images      = images
        images      = self.fully_connect1(images)
        out_img     = self.fully_connect2(images)

        #out_cropimg = []
        #for crop in crop_images_ori:
        crop_images = self.online_model(crop_images_ori)   
        crop_images1 = self.online_predicter(crop_images)
        # crop_images = 0.8 * crop_images + 0.2 * target_proj_one
        crop_images = self.fully_connect1(crop_images1)
        out_cropimg = self.fully_connect2(crop_images)
        #out_cropimg.append(out_crop)
   
        # x1 = torch.flatten(x1, 1)
        # x2 = torch.flatten(x2, 1)

        # target_proj_one = torch.flatten(target_proj_one, 1)
        # target_proj_two = torch.flatten(target_proj_two, 1)

        # diff1 = torch.abs(x1 - target_proj_two.detach())
        # diff2 = torch.abs(x2 - target_proj_one.detach())
        # # x = torch.abs(x1 - x2)
        # x = diff1 + diff2

        # x = self.fully_connect1(x)
        # x = self.fully_connect2(x)

        return out_img, out_cropimg, target_proj_one, crop_images1