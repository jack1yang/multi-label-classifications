import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json

import numpy as np
from torchvision import transforms
import pickle


class ChestXrayDataSet(Dataset):
    def __init__(self,
                 image_dir,
                 file_list, # classification labels format like: img_name, its_class
                 #vocabulary,
                 transforms=None):
        self.image_dir = image_dir
        self.file_names, self.labels = self.__load_label_list(file_list)
        #self.vocab = vocabulary
        self.transform = transforms

    def __load_label_list(self, file_list):
        """
        for chexpert dataset.
        """
        # according to github.com/simongrest/chexpert-entries/blob/master/replicating_chexpert.ipynb
        # we treat uncertainly label as positive/negative 
        # Atelectasis 1,  Cardiomegaly 0, Consolidation 0, Edema 1, Pleural Effusion 0.
        # mapping_dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
        #                 {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}]
        
        labels = []
        filename_list = []
        with open(file_list, 'r' ) as f:
            for line in f:
                line = line.strip().split()
                image_name = line[0]
                filename_list.append(image_name)
                label = [int(i) for i in line[1:]]
                labels.append(label)
        return filename_list, labels

     

    def __getitem__(self, index):
        image_name = self.file_names[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')

  
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
            
        # try:
        #     text = self.caption[image_name]
        # except Exception as err:
        #     text = 'normal. '

        # target = list()
        # max_word_num = 0
        # for i, sentence in enumerate(text.split('. ')):
        #     if i >= self.s_max:
        #         break
        #     sentence = sentence.split()
        #     if len(sentence) == 0 or len(sentence) == 1 or len(sentence) > self.n_max:
        #         continue
        #     tokens = list()
        #     tokens.append(self.vocab('<start>'))
        #     tokens.extend([self.vocab(token) for token in sentence])
        #     tokens.append(self.vocab('<end>'))
        #     if max_word_num < len(tokens):
        #         max_word_num = len(tokens)
        #     target.append(tokens)
        # sentence_num = len(target)

        #return image_frontal, image_lateral, image_prefix, label, target, sentence_num, max_word_num
        return image, image_name, label

    def __len__(self):
        return len(self.file_names)


def collate_fn(data):
    images, image_name, label = zip(*data)
    images = torch.stack(images, 0)
    return images, image_name, np.array(label)

    # image_frontal, image_lateral, image_prefix, label, captions, sentence_num, max_word_num = zip(*data)
    # image_frontal = torch.stack(image_frontal, 0)
    # image_lateral = torch.stack(image_lateral, 0) 

    # max_sentence_num = max(sentence_num)
    # max_word_num = max(max_word_num)

    # targets = np.zeros((len(captions), max_sentence_num + 1, max_word_num))
    # prob = np.zeros((len(captions), max_sentence_num + 1))

    # for i, caption in enumerate(captions):
    #     for j, sentence in enumerate(caption):
    #         targets[i, j, :len(sentence)] = sentence[:]
    #         prob[i][j] = len(sentence) > 0

    # return images, image_prefix, torch.Tensor(label), targets, prob


def get_loader(image_dir,
               #caption_json,
               file_list,
               #vocabulary,
               transform,
               batch_size,
               #s_max=10,
               #n_max=50,
               shuffle=False):
    dataset = ChestXrayDataSet(image_dir=image_dir,
                               #caption_json=caption_json,
                               file_list=file_list,
                               #vocabulary=vocabulary,
                               #s_max=s_max,
                               #n_max=n_max,
                               transforms=transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
					      #num_workers = 4,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    #vocab_path = '../data/vocab.pkl'
    image_dir = '/home/ysk/dataset/dataset/'
    #caption_json = '../data/example_captions.json'
    file_list = '/CheXpert-v1.0-small/valid.csv'
    batch_size = 1
    resize = 600
    crop_size = 500

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # with open(vocab_path, 'rb') as f:
    #     vocab = pickle.load(f)

    data_loader = get_loader(image_dir=image_dir,
                             #caption_json=caption_json,
                             file_list=file_list,
                             #vocabulary=vocab,
                             transform=transform,
                             batch_size=batch_size,
                             shuffle=False)

    # for i, (image, image_id, label, target, prob) in enumerate(data_loader):
    #     print(image.shape)
    #     print(image_id)
    #     print(label)
    #     print(target)
    #     print(prob)
    #     break
    for i, (image, image_name, label) in enumerate(data_loader):
        print(image.shape)
        print(image_name)
        print(label)
