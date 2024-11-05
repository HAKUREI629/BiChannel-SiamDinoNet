import random
import os
import cv2
import numpy as np
import torch
import itertools
from PIL import Image
from torch.utils.data.dataset import Dataset

from torchvision import transforms

from .utils import cvtColor, preprocess_input
from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize
from utils.data_augment import SpecAugment, RandomGaussianBlur, GaussNoise, RandTimeShift, RandFreqShift, TimeReversal, Compander
from utils.rnd_resized_crop import RandomResizedCrop_diy, RandomResizeCrop, RandomLinearFader
from utils.audio_processing import mix_energy

class RandomAudioAugmentation:
    def __init__(self, args):
        self.augmentations = [
            RandTimeShift(do_rand_time_shift=True, Tshift=args['da']['Tshift']),
            RandFreqShift(do_rand_freq_shift=True, Fshift=args['da']['Fshift']),
            #RandomResizedCrop_diy(do_randcrop=True, scale=args['da']['rc_scale'],
            #                    ratio=args['da']['rc_ratio']),
            transforms.RandomApply([TimeReversal(do_time_reversal=True)], p=0.5),
            Compander(do_compansion=True, comp_alpha=args['da']['comp_alpha']),
            SpecAugment(do_time_warp=True, W=args['da']['W'],
                        do_freq_mask=True, F=args['da']['F'], m_f=args['da']['m_f'],
                        reduce_mask_range=args['da']['reduce_mask_range'],
                        do_time_mask=True, T=args['da']['T'], m_t=args['da']['m_t'],
                        mask_val=args['da']['mask_val']),
            GaussNoise(stdev_gen=args['da']['awgn_stdev_gen']),
            RandomGaussianBlur(do_blur=True, max_ksize=args['da']['blur_max_ksize'],
                            stdev_x=args['da']['blur_stdev_x']),
            #RandomResizeCrop(),
            RandomLinearFader(),
        ]

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean, std),
            ])


    def __call__(self, sample):
        num_augmentations = random.randint(1, 4)
        chosen_augmentations = random.sample(self.augmentations, num_augmentations)
        
        augmented_sample = sample
        for augmentation in chosen_augmentations:
            augmented_sample = augmentation(augmented_sample)
        
        augmented_sample = self.norm(augmented_sample)
        
        return augmented_sample

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class SiameseDinoDataset(Dataset):
    def __init__(self, input_shape, crop_shape, crop_num, lines, labels, args, valid_path, mixaugment_flag=True, na='train'):
        self.input_shape    = input_shape
        self.crop_shape     = crop_shape
        #print(self.crop_shape)
        self.crop_num       = int(crop_num)
        self.train_lines    = lines
        self.train_labels   = labels
        self.types          = max(labels)
        self.args           = args
        self.na             = na
        
        self.mixaugment_flag   = mixaugment_flag
        self.test_transform    = None
        self.transform         = RandomAudioAugmentation(args)

        self.valid_path        = valid_path

        self.valid_pairs, self.valid_labels = self._get_all_valid_piars(self.valid_path)
        self.pos_index                      = [index for index, element in enumerate(self.valid_labels) if element == 1]
        self.neg_index                      = [index for index, element in enumerate(self.valid_labels) if element == 0]
        self.len               = 65000 if na == 'train' else len(self.pos_index)
        
        self.neg_ran = np.random.permutation(len(self.neg_index))

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        if self.na != 'train':
            pairs1 = self.valid_pairs[self.pos_index[index]]
            label1 = self.valid_labels[self.pos_index[index]]
            pairs2 = self.valid_pairs[self.neg_index[self.neg_ran[index]]]
            label2 = self.valid_labels[self.neg_index[self.neg_ran[index]]]

            pairs_of_images = np.zeros((2, 2, self.input_shape[0], self.input_shape[1]))
            labels          = np.zeros((2,1))
            labels[0]       = label1
            labels[1]       = label2

            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ])

            
            image         = np.load(pairs1[0])
            image = self.test_transform(image)
            pairs_of_images[0, 0, :, :] = image

            image         = np.load(pairs1[1])
            image = self.test_transform(image)
            pairs_of_images[0, 1, :, :] = image

            image         = np.load(pairs2[0])
            image = self.test_transform(image)
            pairs_of_images[1, 0, :, :] = image

            image         = np.load(pairs2[1])
            image = self.test_transform(image)
            pairs_of_images[1, 1, :, :] = image

            random_permutation = np.random.permutation(2)
            labels = labels[random_permutation]
            pairs_of_images[:, :, :, :] = pairs_of_images[random_permutation, :, :, :]

            return pairs_of_images, [], labels, []

        batch_images_path = []
        miximage_path     = []

        c               = random.randint(0, self.types - 1)
        selected_path   = self.train_lines[self.train_labels[:] == c]
        while len(selected_path) < 3:
            c               = random.randint(0, self.types - 1)
            selected_path   = self.train_lines[self.train_labels[:] == c]

        image_indexes = random.sample(range(0, len(selected_path)), 3)

        batch_images_path.append(selected_path[image_indexes[0]])
        batch_images_path.append(selected_path[image_indexes[1]])

        batch_images_path.append(selected_path[image_indexes[2]])

        different_c         = list(range(self.types))
        different_c.pop(c)
        different_c_index   = np.random.choice(range(0, self.types - 1), 1)
        current_c           = different_c[different_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            different_c_index   = np.random.choice(range(0, self.types - 1), 1)
            current_c           = different_c[different_c_index[0]]
            selected_path       = self.train_lines[self.train_labels == current_c]

        image_indexes = random.sample(range(0, len(selected_path)), 1)
        batch_images_path.append(selected_path[image_indexes[0]])

        mix_c               = list(range(self.types))
        mix_c.pop(c)
        mix_c_index         = np.random.choice(range(0, self.types - 1), 1)
        current_c           = mix_c[mix_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            mix_c_index     = np.random.choice(range(0, self.types - 1), 1)
            current_c       = mix_c[mix_c_index[0]]
            selected_path   = self.train_lines[self.train_labels == current_c]
        
        miximage_indexes = random.sample(range(0, len(selected_path)), 1)
        miximage_path.append(selected_path[miximage_indexes[0]])

        mix_c               = list(range(self.types))
        mix_c.pop(c)
        mix_c_index         = np.random.choice(range(0, self.types - 1), 1)
        current_c           = mix_c[mix_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            mix_c_index     = np.random.choice(range(0, self.types - 1), 1)
            current_c       = mix_c[mix_c_index[0]]
            selected_path   = self.train_lines[self.train_labels == current_c]
        
        miximage_indexes = random.sample(range(0, len(selected_path)), 1)
        miximage_path.append(selected_path[miximage_indexes[0]])
        
        pairs_of_images, multi_crop_pairs_images, labels, multi_crop_labels = self._convert_path_list_to_images_and_labels(batch_images_path, miximage_path)
        return pairs_of_images, multi_crop_pairs_images, labels, multi_crop_labels

    def _convert_path_list_to_images_and_labels(self, path_list, miximage_path):
        #-------------------------------------------#
        #   len(path_list)      = 4
        #   len(path_list) / 2  = 2
        #-------------------------------------------#
        number_of_pairs = int(len(path_list) / 2)

        pairs_of_images         = np.zeros((number_of_pairs, 2, self.input_shape[0], self.input_shape[1]))
        multi_crop_pairs_images = [np.zeros((number_of_pairs, 2, self.crop_shape[0], self.crop_shape[1])) for i in range(self.crop_num)]
        labels                  = np.zeros((number_of_pairs, 1))
        multi_crop_labels       = [np.zeros((number_of_pairs, 1)) for i in range(self.crop_num)]

        for pair in range(number_of_pairs):

            # image = Image.open(path_list[pair * 2])
            image = np.load(path_list[pair * 2])

            for i in range(self.crop_num):
                crop_image = self._random_crop_numpy_array(image, self.crop_shape)
                crop_image = self.transform(crop_image)
                multi_crop_pairs_images[i][pair, 0, :, :] = crop_image
                if (pair + 1) % 2 == 0:
                    multi_crop_labels[i][pair] = 0
                else:
                    multi_crop_labels[i][pair] = 1
            
            if self.na == 'train':
                if self.mixaugment_flag:
                    alpha_mix = np.random.uniform(low=0, high=self.args['da']['mix_energy'])
                    miximage = np.load(miximage_path[0])
                    image = mix_energy(image, miximage, alpha_mix)

                image = self.transform(image)
            
            else:
                self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(mean, std),
                ])
                image = self.test_transform(image)

            pairs_of_images[pair, 0, :, :] = image

            # image = Image.open(path_list[pair * 2 + 1])
            image = np.load(path_list[pair * 2 + 1])

            for i in range(self.crop_num):
                crop_image = self._random_crop_numpy_array(image, self.crop_shape)
                crop_image = self.transform(crop_image)
                multi_crop_pairs_images[i][pair, 1, :, :] = crop_image
                if (pair + 1) % 2 == 0:
                    multi_crop_labels[i][pair] = 0
                else:
                    multi_crop_labels[i][pair] = 1

            if self.na:
                if self.mixaugment_flag:
                    alpha_mix = np.random.uniform(low=0, high=self.args['da']['mix_energy'])
                    miximage = np.load(miximage_path[1])
                    image = mix_energy(image, miximage, alpha_mix)
                
                image = self.transform(image)
            
            else:
                self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(mean, std),
                ])
                image = self.test_transform(image)

            pairs_of_images[pair, 1, :, :] = image
                
            if (pair + 1) % 2 == 0:
                labels[pair] = 0
            else:
                labels[pair] = 1

        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[:, :, :, :] = pairs_of_images[random_permutation, :, :, :]

        # random_permutation = np.random.permutation(number_of_pairs)
        for i in range(self.crop_num):
            multi_crop_pairs_images[i][random_permutation, :, :, :] = multi_crop_pairs_images[i][random_permutation, :, :, :]
            multi_crop_labels[i] = multi_crop_labels[i][random_permutation]

        return pairs_of_images, multi_crop_pairs_images, labels, multi_crop_labels

    def _get_all_valid_piars(self, path):
        file_paths = []
        file_pairs = []
        file_eq    = []
        for foldername, subfolders, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith('.npy'):
                    file_paths.append(os.path.join(foldername, filename))
        
        file_pairs = []
        for pair in itertools.combinations(file_paths, 2):
            same_folder = 1 if os.path.dirname(pair[0]) == os.path.dirname(pair[1]) else 0
            file_pairs.append(pair)
            file_eq.append(same_folder)
        
        return file_pairs, file_eq

    def _random_crop_numpy_array(self, image_array, crop_size):

        height, width = image_array.shape[:2]
        crop_height, crop_width = crop_size
        
        crop_height = min(crop_height, height)
        crop_width = min(crop_width, width)
        
        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)
        
        crop_array = image_array[y:y+crop_height, x:x+crop_width]
        
        return crop_array

def dataset_collate_dino(batch):
    images          = []
    crop_images     = []
    labels          = []
    crop_labels     = []
    for pair_imgs, pair_crop_images, pair_labels, pair_crop_labels in batch:
        images.append(pair_imgs)
        labels.append(pair_labels)
        if len(pair_crop_images) > 0:
            for i in range(len(pair_crop_images)):
                crop_images.append(pair_crop_images[i])
                crop_labels.append(pair_crop_labels[i])
            
    images      = torch.from_numpy(np.array(np.concatenate(images, axis=0))).type(torch.FloatTensor)
    labels      = torch.from_numpy(np.array(np.concatenate(labels, axis=0))).type(torch.FloatTensor)
    if len(pair_crop_images) > 0:
        crop_images = torch.from_numpy(np.array(np.concatenate(crop_images, axis=0))).type(torch.FloatTensor)
        crop_labels = torch.from_numpy(np.array(np.concatenate(crop_labels, axis=0))).type(torch.FloatTensor)

    return images, crop_images, labels, crop_labels

class SiameseDinoDatasetV2(Dataset):
    def __init__(self, input_shape, crop_shape, crop_num, lines, labels, args, valid_path, mixaugment_flag=True, na='train'):
        self.input_shape    = input_shape
        self.crop_shape     = crop_shape
        #print(self.crop_shape)
        self.crop_num       = int(crop_num)
        self.train_lines    = lines
        self.train_labels   = labels
        self.types          = max(labels)
        self.args           = args
        self.na             = na
        
        self.mixaugment_flag   = mixaugment_flag
        self.test_transform    = None
        self.transform         = RandomAudioAugmentation(args)

        self.valid_path        = valid_path

        self.valid_pairs, self.valid_labels = self._get_all_valid_piars(self.valid_path)
        self.pos_index                      = [index for index, element in enumerate(self.valid_labels) if element == 1]
        self.neg_index                      = [index for index, element in enumerate(self.valid_labels) if element == 0]
        self.len               = 65000 if na == 'train' else len(self.pos_index)
        
        self.neg_ran = np.random.permutation(len(self.neg_index))

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        if self.na != 'train':
            pairs1 = self.valid_pairs[self.pos_index[index]]
            label1 = self.valid_labels[self.pos_index[index]]
            pairs2 = self.valid_pairs[self.neg_index[self.neg_ran[index]]]
            label2 = self.valid_labels[self.neg_index[self.neg_ran[index]]]

            pairs_of_images = np.zeros((2, 2, self.input_shape[0], self.input_shape[1]))
            labels          = np.zeros((2,1))
            labels[0]       = label1
            labels[1]       = label2

            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ])

            
            image         = np.load(pairs1[0])
            image = self.test_transform(image)
            pairs_of_images[0, 0, :, :] = image

            image         = np.load(pairs1[1])
            image = self.test_transform(image)
            pairs_of_images[0, 1, :, :] = image

            image         = np.load(pairs2[0])
            image = self.test_transform(image)
            pairs_of_images[1, 0, :, :] = image

            image         = np.load(pairs2[1])
            image = self.test_transform(image)
            pairs_of_images[1, 1, :, :] = image

            random_permutation = np.random.permutation(2)
            labels = labels[random_permutation]
            pairs_of_images[:, :, :, :] = pairs_of_images[random_permutation, :, :, :]

            return pairs_of_images, [], labels, []

        batch_images_path = []
        miximage_path     = []

        c               = random.randint(0, self.types - 1)
        selected_path   = self.train_lines[self.train_labels[:] == c]
        while len(selected_path) < 3:
            c               = random.randint(0, self.types - 1)
            selected_path   = self.train_lines[self.train_labels[:] == c]

        image_indexes = random.sample(range(0, len(selected_path)), 3)

        batch_images_path.append(selected_path[image_indexes[0]])
        batch_images_path.append(selected_path[image_indexes[1]])

        batch_images_path.append(selected_path[image_indexes[2]])

        different_c         = list(range(self.types))
        different_c.pop(c)
        different_c_index   = np.random.choice(range(0, self.types - 1), 1)
        current_c           = different_c[different_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            different_c_index   = np.random.choice(range(0, self.types - 1), 1)
            current_c           = different_c[different_c_index[0]]
            selected_path       = self.train_lines[self.train_labels == current_c]

        image_indexes = random.sample(range(0, len(selected_path)), 1)
        batch_images_path.append(selected_path[image_indexes[0]])

        mix_c               = list(range(self.types))
        mix_c.pop(c)
        mix_c_index         = np.random.choice(range(0, self.types - 1), 1)
        current_c           = mix_c[mix_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            mix_c_index     = np.random.choice(range(0, self.types - 1), 1)
            current_c       = mix_c[mix_c_index[0]]
            selected_path   = self.train_lines[self.train_labels == current_c]
        
        miximage_indexes = random.sample(range(0, len(selected_path)), 1)
        miximage_path.append(selected_path[miximage_indexes[0]])

        mix_c               = list(range(self.types))
        mix_c.pop(c)
        mix_c_index         = np.random.choice(range(0, self.types - 1), 1)
        current_c           = mix_c[mix_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            mix_c_index     = np.random.choice(range(0, self.types - 1), 1)
            current_c       = mix_c[mix_c_index[0]]
            selected_path   = self.train_lines[self.train_labels == current_c]
        
        miximage_indexes = random.sample(range(0, len(selected_path)), 1)
        miximage_path.append(selected_path[miximage_indexes[0]])
        
        pairs_of_images, multi_crop_pairs_images, labels, multi_crop_labels = self._convert_path_list_to_images_and_labels(batch_images_path, miximage_path)
        return pairs_of_images, multi_crop_pairs_images, labels, multi_crop_labels

    def _convert_path_list_to_images_and_labels(self, path_list, miximage_path):
        #-------------------------------------------#
        #   len(path_list)      = 4
        #   len(path_list) / 2  = 2
        #-------------------------------------------#
        number_of_pairs = int(len(path_list) / 2)

        pairs_of_images         = np.zeros((number_of_pairs, 2, self.input_shape[0], self.input_shape[1]))
        multi_crop_pairs_images = [np.zeros((number_of_pairs, 2, self.crop_shape[0], self.crop_shape[1])) for i in range(self.crop_num)]
        labels                  = np.zeros((number_of_pairs, 1))
        multi_crop_labels       = [np.zeros((number_of_pairs, 1)) for i in range(self.crop_num)]

        for pair in range(number_of_pairs):

            # image = Image.open(path_list[pair * 2])
            image1 = np.load(path_list[pair * 2])
            image2 = np.load(path_list[pair * 2 + 1])

            for i in range(self.crop_num):
                crop_image1, crop_image2 = self._random_crop_numpy_array(image1, image2, self.crop_shape)
                crop_image1 = self.transform(crop_image1)
                crop_image2 = self.transform(crop_image2)
                multi_crop_pairs_images[i][pair, 0, :, :] = crop_image1
                multi_crop_pairs_images[i][pair, 1, :, :] = crop_image2
                if (pair + 1) % 2 == 0:
                    multi_crop_labels[i][pair] = 0
                else:
                    multi_crop_labels[i][pair] = 1
            
            if self.na == 'train':
                if self.mixaugment_flag:
                    alpha_mix = np.random.uniform(low=0, high=self.args['da']['mix_energy'])
                    miximage = np.load(miximage_path[0])
                    image1 = mix_energy(image1, miximage, alpha_mix)

                image1 = self.transform(image1)
            
            else:
                self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(mean, std),
                ])
                image1 = self.test_transform(image1)

            pairs_of_images[pair, 0, :, :] = image1

            if self.na:
                if self.mixaugment_flag:
                    alpha_mix = np.random.uniform(low=0, high=self.args['da']['mix_energy'])
                    miximage = np.load(miximage_path[1])
                    image2 = mix_energy(image2, miximage, alpha_mix)
                
                image2 = self.transform(image2)
            
            else:
                self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(mean, std),
                ])
                image2 = self.test_transform(image2)

            pairs_of_images[pair, 1, :, :] = image2            
                
            if (pair + 1) % 2 == 0:
                labels[pair] = 0
            else:
                labels[pair] = 1

        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[:, :, :, :] = pairs_of_images[random_permutation, :, :, :]
       
        for i in range(self.crop_num):
            random_permutation = np.random.permutation(number_of_pairs)
            multi_crop_pairs_images[i][random_permutation, :, :, :] = multi_crop_pairs_images[i][random_permutation, :, :, :]
            multi_crop_labels[i] = multi_crop_labels[i][random_permutation]

        return pairs_of_images, multi_crop_pairs_images, labels, multi_crop_labels

    def _get_all_valid_piars(self, path):
        file_paths = []
        file_pairs = []
        file_eq    = []
        for foldername, subfolders, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith('.npy'):
                    file_paths.append(os.path.join(foldername, filename))
        
        file_pairs = []
        for pair in itertools.combinations(file_paths, 2):
            same_folder = 1 if os.path.dirname(pair[0]) == os.path.dirname(pair[1]) else 0
            file_pairs.append(pair)
            file_eq.append(same_folder)
        
        return file_pairs, file_eq

    def _random_crop_numpy_array(self, image_array1, image_array2, crop_size):

        height, width = image_array1.shape[:2]
        crop_height, crop_width = crop_size
        
        crop_height = min(crop_height, height)
        crop_width = min(crop_width, width)
        
        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)

        crop_array1 = image_array1[y:y+crop_height, x:x+crop_width]
        crop_array2 = image_array2[y:y+crop_height, x:x+crop_width]
        
        return crop_array1, crop_array2

def dataset_collate_dinov2(batch):
    images          = []
    crop_images     = []
    labels          = []
    crop_labels     = []

    crop_images_array = [[] for i in range(2)]
    crop_labels_array = [[] for i in range(2)]

    for pair_imgs, pair_crop_images, pair_labels, pair_crop_labels in batch:
        images.append(pair_imgs)
        labels.append(pair_labels)
        if len(pair_crop_images) > 0:
            for i in range(len(pair_crop_images)):
                crop_images_array[i].append(pair_crop_images[i])
                crop_labels_array[i].append(pair_crop_labels[i])
            
    images = torch.from_numpy(np.array(np.concatenate(images, axis=0))).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(np.concatenate(labels, axis=0))).type(torch.FloatTensor)
    if len(pair_crop_images) > 0:
        for i in range(2):
            crop_images_array[i] = torch.from_numpy(np.array(np.concatenate(crop_images_array[i], axis=0))).type(torch.FloatTensor)
            crop_labels_array[i] = torch.from_numpy(np.array(np.concatenate(crop_labels_array[i], axis=0))).type(torch.FloatTensor)

    return images, crop_images_array, labels, crop_labels_array

class SiameseDinoDatasetV3(Dataset):
    def __init__(self, input_shape, crop_shape, crop_num, lines, labels, args, valid_path, mixaugment_flag=True, na='train'):
        self.input_shape    = input_shape
        self.crop_shape     = crop_shape
        #print(self.crop_shape)
        self.crop_num       = int(crop_num)
        self.train_lines    = lines
        self.train_labels   = labels
        self.types          = max(labels)
        self.args           = args
        self.na             = na
        
        self.mixaugment_flag   = mixaugment_flag
        self.test_transform    = None
        self.transform         = RandomAudioAugmentation(args)

        self.valid_path        = valid_path

        self.valid_pairs, self.valid_labels = self._get_all_valid_piars(self.valid_path)
        self.pos_index                      = [index for index, element in enumerate(self.valid_labels) if element == 1]
        self.neg_index                      = [index for index, element in enumerate(self.valid_labels) if element == 0]
        self.len               = 65000 if na == 'train' else len(self.pos_index)
        
        self.neg_ran = np.random.permutation(len(self.neg_index))

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        if self.na != 'train':
            pairs1 = self.valid_pairs[self.pos_index[index]]
            label1 = self.valid_labels[self.pos_index[index]]
            pairs2 = self.valid_pairs[self.neg_index[self.neg_ran[index]]]
            label2 = self.valid_labels[self.neg_index[self.neg_ran[index]]]

            pairs_of_images = np.zeros((2, 2, self.input_shape[0], self.input_shape[1]))
            labels          = np.zeros((2,1))
            labels[0]       = label1
            labels[1]       = label2

            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ])

            
            image         = np.load(pairs1[0])
            image = self.test_transform(image)
            pairs_of_images[0, 0, :, :] = image

            image         = np.load(pairs1[1])
            image = self.test_transform(image)
            pairs_of_images[0, 1, :, :] = image

            image         = np.load(pairs2[0])
            image = self.test_transform(image)
            pairs_of_images[1, 0, :, :] = image

            image         = np.load(pairs2[1])
            image = self.test_transform(image)
            pairs_of_images[1, 1, :, :] = image

            random_permutation = np.random.permutation(2)
            labels = labels[random_permutation]
            pairs_of_images[:, :, :, :] = pairs_of_images[random_permutation, :, :, :]

            return pairs_of_images, [], labels, []

        batch_images_path = []
        miximage_path     = []

        c               = random.randint(0, self.types - 1)
        selected_path   = self.train_lines[self.train_labels[:] == c]
        while len(selected_path) < 3:
            c               = random.randint(0, self.types - 1)
            selected_path   = self.train_lines[self.train_labels[:] == c]

        image_indexes = random.sample(range(0, len(selected_path)), 3)

        batch_images_path.append(selected_path[image_indexes[0]])
        batch_images_path.append(selected_path[image_indexes[1]])

        batch_images_path.append(selected_path[image_indexes[2]])

        different_c         = list(range(self.types))
        different_c.pop(c)
        different_c_index   = np.random.choice(range(0, self.types - 1), 1)
        current_c           = different_c[different_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            different_c_index   = np.random.choice(range(0, self.types - 1), 1)
            current_c           = different_c[different_c_index[0]]
            selected_path       = self.train_lines[self.train_labels == current_c]

        image_indexes = random.sample(range(0, len(selected_path)), 1)
        batch_images_path.append(selected_path[image_indexes[0]])

        mix_c               = list(range(self.types))
        mix_c.pop(c)
        mix_c_index         = np.random.choice(range(0, self.types - 1), 1)
        current_c           = mix_c[mix_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            mix_c_index     = np.random.choice(range(0, self.types - 1), 1)
            current_c       = mix_c[mix_c_index[0]]
            selected_path   = self.train_lines[self.train_labels == current_c]
        
        miximage_indexes = random.sample(range(0, len(selected_path)), 1)
        miximage_path.append(selected_path[miximage_indexes[0]])

        mix_c               = list(range(self.types))
        mix_c.pop(c)
        mix_c_index         = np.random.choice(range(0, self.types - 1), 1)
        current_c           = mix_c[mix_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            mix_c_index     = np.random.choice(range(0, self.types - 1), 1)
            current_c       = mix_c[mix_c_index[0]]
            selected_path   = self.train_lines[self.train_labels == current_c]
        
        miximage_indexes = random.sample(range(0, len(selected_path)), 1)
        miximage_path.append(selected_path[miximage_indexes[0]])
        
        pairs_of_images, multi_crop_pairs_images, labels, multi_crop_labels = self._convert_path_list_to_images_and_labels(batch_images_path, miximage_path)
        return pairs_of_images, multi_crop_pairs_images, labels, multi_crop_labels

    def _convert_path_list_to_images_and_labels(self, path_list, miximage_path):
        #-------------------------------------------#
        #   len(path_list)      = 4
        #   len(path_list) / 2  = 2
        #-------------------------------------------#
        number_of_pairs = int(len(path_list) / 2)

        pairs_of_images         = np.zeros((number_of_pairs, 2, self.input_shape[0], self.input_shape[1]))
        multi_crop_pairs_images = np.zeros((number_of_pairs, 2, self.input_shape[0], self.input_shape[1]))
        labels                  = np.zeros((number_of_pairs, 1))
        multi_crop_labels       = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):

            # image = Image.open(path_list[pair * 2])
            image1 = np.load(path_list[pair * 2])
            image2 = np.load(path_list[pair * 2 + 1])

            for i in range(1):
                #crop_image1, crop_image2 = self._random_crop_numpy_array(image1, image2, self.crop_shape)
                crop_image1 = self.transform(image1)
                crop_image2 = self.transform(image2)
                multi_crop_pairs_images[pair, 0, :, :] = crop_image1
                multi_crop_pairs_images[pair, 1, :, :] = crop_image2
                if (pair + 1) % 2 == 0:
                    multi_crop_labels[pair] = 0
                else:
                    multi_crop_labels[pair] = 1
            
            if self.na == 'train':
                if self.mixaugment_flag:
                    alpha_mix = np.random.uniform(low=0, high=self.args['da']['mix_energy'])
                    miximage = np.load(miximage_path[0])
                    image1 = mix_energy(image1, miximage, alpha_mix)

                image1 = self.transform(image1)
            
            else:
                self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(mean, std),
                ])
                image1 = self.test_transform(image1)

            pairs_of_images[pair, 0, :, :] = image1

            if self.na:
                if self.mixaugment_flag:
                    alpha_mix = np.random.uniform(low=0, high=self.args['da']['mix_energy'])
                    miximage = np.load(miximage_path[1])
                    image2 = mix_energy(image2, miximage, alpha_mix)
                
                image2 = self.transform(image2)
            
            else:
                self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(mean, std),
                ])
                image2 = self.test_transform(image2)

            pairs_of_images[pair, 1, :, :] = image2            
                
            if (pair + 1) % 2 == 0:
                labels[pair] = 0
            else:
                labels[pair] = 1

        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[:, :, :, :] = pairs_of_images[random_permutation, :, :, :]
       
        for i in range(1):
            random_permutation = np.random.permutation(number_of_pairs)
            multi_crop_pairs_images[random_permutation, :, :, :] = multi_crop_pairs_images[random_permutation, :, :, :]
            multi_crop_labels = multi_crop_labels[random_permutation]

        return pairs_of_images, multi_crop_pairs_images, labels, multi_crop_labels

    def _get_all_valid_piars(self, path):
        file_paths = []
        file_pairs = []
        file_eq    = []
        for foldername, subfolders, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith('.npy'):
                    file_paths.append(os.path.join(foldername, filename))
        
        file_pairs = []
        for pair in itertools.combinations(file_paths, 2):
            same_folder = 1 if os.path.dirname(pair[0]) == os.path.dirname(pair[1]) else 0
            file_pairs.append(pair)
            file_eq.append(same_folder)
        
        return file_pairs, file_eq

    def _random_crop_numpy_array(self, image_array1, image_array2, crop_size):

        height, width = image_array1.shape[:2]
        crop_height, crop_width = crop_size
        
        crop_height = min(crop_height, height)
        crop_width = min(crop_width, width)
        
        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)
        
        crop_array1 = image_array1[y:y+crop_height, x:x+crop_width]
        crop_array2 = image_array2[y:y+crop_height, x:x+crop_width]
        
        return crop_array1, crop_array2

def dataset_collate_dinov3(batch):
    images          = []
    crop_images     = []
    labels          = []
    crop_labels     = []

    crop_images_array = []
    crop_labels_array = []

    for pair_imgs, pair_crop_images, pair_labels, pair_crop_labels in batch:
        images.append(pair_imgs)
        labels.append(pair_labels)
        crop_images_array.append(pair_crop_images)
        crop_labels_array.append(pair_crop_labels)
            
    images = torch.from_numpy(np.array(np.concatenate(images, axis=0))).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(np.concatenate(labels, axis=0))).type(torch.FloatTensor)
    crop_images_array = torch.from_numpy(np.array(np.concatenate(crop_images_array, axis=0))).type(torch.FloatTensor)
    crop_labels_array = torch.from_numpy(np.array(np.concatenate(crop_labels_array, axis=0))).type(torch.FloatTensor)
    
    return images, crop_images_array, labels, crop_labels_array


class SiameseDinoDatasetV4(Dataset):
    def __init__(self, input_shape, crop_shape, crop_num, lines, labels, args, valid_path, mixaugment_flag=True, na='train'):
        self.input_shape    = input_shape
        self.crop_shape     = crop_shape
        #print(self.crop_shape)
        self.crop_num       = int(crop_num)
        self.train_lines    = lines
        self.train_labels   = labels
        self.types          = max(labels)
        self.args           = args
        self.na             = na
        
        self.mixaugment_flag   = mixaugment_flag
        self.test_transform    = None
        self.transform         = RandomAudioAugmentation(args)

        self.valid_path        = valid_path

        self.valid_pairs, self.valid_labels = self._get_all_valid_piars(self.valid_path)
        self.pos_index                      = [index for index, element in enumerate(self.valid_labels) if element == 1]
        self.neg_index                      = [index for index, element in enumerate(self.valid_labels) if element == 0]
        self.len               = 65000 if na == 'train' else len(self.pos_index)
        
        self.neg_ran = np.random.permutation(len(self.neg_index))

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        if self.na != 'train':
            pairs1 = self.valid_pairs[self.pos_index[index]]
            label1 = self.valid_labels[self.pos_index[index]]
            pairs2 = self.valid_pairs[self.neg_index[self.neg_ran[index]]]
            label2 = self.valid_labels[self.neg_index[self.neg_ran[index]]]

            pairs_of_images = np.zeros((2, 2, self.input_shape[0], self.input_shape[1]))
            labels          = np.zeros((2,1))
            labels[0]       = label1
            labels[1]       = label2

            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean, std),
            ])

            
            image         = np.load(pairs1[0])
            image = self.test_transform(image)
            pairs_of_images[0, 0, :, :] = image

            image         = np.load(pairs1[1])
            image = self.test_transform(image)
            pairs_of_images[0, 1, :, :] = image

            image         = np.load(pairs2[0])
            image = self.test_transform(image)
            pairs_of_images[1, 0, :, :] = image

            image         = np.load(pairs2[1])
            image = self.test_transform(image)
            pairs_of_images[1, 1, :, :] = image

            random_permutation = np.random.permutation(2)
            labels = labels[random_permutation]
            pairs_of_images[:, :, :, :] = pairs_of_images[random_permutation, :, :, :]

            return pairs_of_images, [], labels, []

        batch_images_path = []
        miximage_path     = []

        c               = random.randint(0, self.types - 1)
        selected_path   = self.train_lines[self.train_labels[:] == c]
        while len(selected_path) < 3:
            c               = random.randint(0, self.types - 1)
            selected_path   = self.train_lines[self.train_labels[:] == c]

        image_indexes = random.sample(range(0, len(selected_path)), 3)

        batch_images_path.append(selected_path[image_indexes[0]])
        batch_images_path.append(selected_path[image_indexes[1]])

        batch_images_path.append(selected_path[image_indexes[2]])

        different_c         = list(range(self.types))
        different_c.pop(c)
        different_c_index   = np.random.choice(range(0, self.types - 1), 1)
        current_c           = different_c[different_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            different_c_index   = np.random.choice(range(0, self.types - 1), 1)
            current_c           = different_c[different_c_index[0]]
            selected_path       = self.train_lines[self.train_labels == current_c]

        image_indexes = random.sample(range(0, len(selected_path)), 1)
        batch_images_path.append(selected_path[image_indexes[0]])

        mix_c               = list(range(self.types))
        mix_c.pop(c)
        mix_c_index         = np.random.choice(range(0, self.types - 1), 1)
        current_c           = mix_c[mix_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            mix_c_index     = np.random.choice(range(0, self.types - 1), 1)
            current_c       = mix_c[mix_c_index[0]]
            selected_path   = self.train_lines[self.train_labels == current_c]
        
        miximage_indexes = random.sample(range(0, len(selected_path)), 1)
        miximage_path.append(selected_path[miximage_indexes[0]])

        mix_c               = list(range(self.types))
        mix_c.pop(c)
        mix_c_index         = np.random.choice(range(0, self.types - 1), 1)
        current_c           = mix_c[mix_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            mix_c_index     = np.random.choice(range(0, self.types - 1), 1)
            current_c       = mix_c[mix_c_index[0]]
            selected_path   = self.train_lines[self.train_labels == current_c]
        
        miximage_indexes = random.sample(range(0, len(selected_path)), 1)
        miximage_path.append(selected_path[miximage_indexes[0]])
        
        pairs_of_images, multi_crop_pairs_images, labels, multi_crop_labels = self._convert_path_list_to_images_and_labels(batch_images_path, miximage_path)
        return pairs_of_images, multi_crop_pairs_images, labels, multi_crop_labels

    def _convert_path_list_to_images_and_labels(self, path_list, miximage_path):
        #-------------------------------------------#
        #   len(path_list)      = 4
        #   len(path_list) / 2  = 2
        #-------------------------------------------#
        number_of_pairs = int(len(path_list) / 2)

        pairs_of_images         = np.zeros((number_of_pairs, 2, self.input_shape[0], self.input_shape[1]))
        multi_crop_pairs_images = np.zeros((number_of_pairs, 2, self.input_shape[0], self.input_shape[1]))
        labels                  = np.zeros((number_of_pairs, 1))
        multi_crop_labels       = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):

            # image = Image.open(path_list[pair * 2])
            image1 = np.load(path_list[pair * 2])
            image2 = np.load(path_list[pair * 2 + 1])

            for i in range(1):
                #crop_image1, crop_image2 = self._random_crop_numpy_array(image1, image2, self.crop_shape)
                crop_image1 = self.transform(image1)
                crop_image2 = self.transform(image2)
                multi_crop_pairs_images[pair, 0, :, :] = crop_image1
                multi_crop_pairs_images[pair, 1, :, :] = crop_image2
                if (pair + 1) % 2 == 0:
                    multi_crop_labels[pair] = 0
                else:
                    multi_crop_labels[pair] = 1
            
            if self.na == 'train':
                if self.mixaugment_flag:
                    alpha_mix = np.random.uniform(low=0, high=self.args['da']['mix_energy'])
                    miximage = np.load(miximage_path[0])
                    image1 = mix_energy(image1, miximage, alpha_mix)

                image1 = self.transform(image1)
            
            else:
                self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(mean, std),
                ])
                image1 = self.test_transform(image1)

            pairs_of_images[pair, 0, :, :] = image1

            if self.na:
                if self.mixaugment_flag:
                    alpha_mix = np.random.uniform(low=0, high=self.args['da']['mix_energy'])
                    miximage = np.load(miximage_path[1])
                    image2 = mix_energy(image2, miximage, alpha_mix)
                
                image2 = self.transform(image2)
            
            else:
                self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(mean, std),
                ])
                image2 = self.test_transform(image2)

            pairs_of_images[pair, 1, :, :] = image2            
                
            if (pair + 1) % 2 == 0:
                labels[pair] = 0
            else:
                labels[pair] = 1

        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[:, :, :, :] = pairs_of_images[random_permutation, :, :, :]
       
        for i in range(1):
            random_permutation = np.random.permutation(number_of_pairs)
            multi_crop_pairs_images[random_permutation, :, :, :] = multi_crop_pairs_images[random_permutation, :, :, :]
            multi_crop_labels = multi_crop_labels[random_permutation]

        return pairs_of_images, multi_crop_pairs_images, labels, multi_crop_labels

    def _get_all_valid_piars(self, path):
        file_paths = []
        file_pairs = []
        file_eq    = []
        for foldername, subfolders, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith('.npy'):
                    file_paths.append(os.path.join(foldername, filename))
        
        file_pairs = []
        for pair in itertools.combinations(file_paths, 2):
            same_folder = 1 if os.path.dirname(pair[0]) == os.path.dirname(pair[1]) else 0
            file_pairs.append(pair)
            file_eq.append(same_folder)
        
        return file_pairs, file_eq

    def _random_crop_numpy_array(self, image_array1, image_array2, crop_size):

        height, width = image_array1.shape[:2]
        crop_height, crop_width = crop_size
        
        crop_height = min(crop_height, height)
        crop_width = min(crop_width, width)
        
        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)
        
        crop_array1 = image_array1[y:y+crop_height, x:x+crop_width]
        crop_array2 = image_array2[y:y+crop_height, x:x+crop_width]
        
        return crop_array1, crop_array2

def dataset_collate_dinov4(batch):
    images          = []
    crop_images     = []
    labels          = []
    crop_labels     = []

    crop_images_array = []
    crop_labels_array = []

    for pair_imgs, pair_crop_images, pair_labels, pair_crop_labels in batch:
        images.append(pair_imgs)
        labels.append(pair_labels)
        crop_images_array.append(pair_crop_images)
        crop_labels_array.append(pair_crop_labels)
            
    images = torch.from_numpy(np.array(np.concatenate(images, axis=0))).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(np.concatenate(labels, axis=0))).type(torch.FloatTensor)
    crop_images_array = torch.from_numpy(np.array(np.concatenate(crop_images_array, axis=0))).type(torch.FloatTensor)
    crop_labels_array = torch.from_numpy(np.array(np.concatenate(crop_labels_array, axis=0))).type(torch.FloatTensor)
    
    return images, crop_images_array, labels, crop_labels_array