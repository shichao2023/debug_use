import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
from annotator_network_cifar10 import Net
import torch
import torch.nn as nn
import math
from scipy.special import softmax


IMAGE_SIZE = 32

dataTransform = transforms.Compose([
    
    #transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), #to [0.0, 1.0], H×W×C -> C×H×W
    #transforms.ToPILImage(),
    # transforms.Resize(IMAGE_SIZE),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
])
    
class Cus_Dataset(data.Dataset):
    def __init__(self, mode, data_set, begin_ind, end_ind, begin_ind1=None, end_ind1=None, new_model=None, st_after_weak=None):
        self.mode = mode
        self.total_img = data_set[0]
        self.list_img = []
        self.list_diff = []
        self.total_label = data_set[1]
        self.list_label = []
        self.total_size = data_set[2]
        self.list_w_label = []
        self.c_label = []
        self.st_label = []
        self.data_size = 0
        self.transform = dataTransform
        self.n_class = 10

        if self.mode == 'train': #used for training the weak annotator, and finetune
            
            self.data_size = end_ind-begin_ind
            self.list_img = self.total_img[begin_ind: end_ind]
            self.list_label = self.total_label[begin_ind: end_ind]

                
        elif self.mode == 'val': #val data

            self.data_size = end_ind-begin_ind
            self.list_img = self.total_img[begin_ind: end_ind]
            self.list_label = self.total_label[begin_ind: end_ind]

        elif self.mode == 'test':
            pass
            
        elif self.mode == 'weak': #weak data

            self.data_size = end_ind-begin_ind
            self.list_img = self.total_img[begin_ind: end_ind]

            device = torch.device("cuda")
            model_file = './model/model.pth'
            model = Net().to(device)
            #model = nn.DataParallel(model)
            model.load_state_dict(torch.load(model_file))
            model.eval()
            
            batch_size = 100
            for i in range(int(len(self.list_img)/batch_size)):
                if (i == int(len(self.list_img)/batch_size) -1) and (i * batch_size < len(self.list_img)):
                    imgs = self.list_img[(i)*batch_size:]
                else:
                    imgs = self.list_img[(i)*batch_size:(i+1)*batch_size]

                imgs = np.asarray([self.transform(img).numpy() for img in imgs])
                imgs = torch.tensor(imgs).to(device)
                
                out = model(imgs)
                out = F.softmax(out, dim=1)
                out = out.data.cpu().numpy()
                #self.list_label.append(np.argmax(out))
                self.list_label += [out[j] for j in range(out.shape[0])]

        elif self.mode == 'diff': #calulate soft label difference 

            self.data_size = end_ind-begin_ind
            self.list_img = self.total_img[begin_ind: end_ind]
            self.list_label = self.total_label[begin_ind: end_ind]
            

            device = torch.device("cuda")
            model_file = './model/model.pth'
            model = Net().to(device)
            #model = nn.DataParallel(model)
            model.load_state_dict(torch.load(model_file))
            model.eval()

            batch_size = 100
            for i in range(int(len(self.list_img)/batch_size)):
                if (i == int(len(self.list_img)/batch_size) -1) and (i * batch_size < len(self.list_img)):
                    imgs = self.list_img[(i)*batch_size:]
                else:
                    imgs = self.list_img[(i)*batch_size:(i+1)*batch_size]
                imgs = np.asarray([self.transform(img).numpy() for img in imgs])
                imgs = torch.tensor(imgs).to(device)
                
                out = model(imgs)
                out = F.softmax(out, dim=1)
                out = out.data.cpu().numpy()
                
                #self.list_label.append(np.argmax(out))
                
                #self.list_diff += [np.eye(self.n_class)[np.argmax(self.list_label[batch_size*i+j])] - out[j] for j in range(out.shape[0])]
                self.list_diff += [self.list_label[batch_size*i+j] - out[j] for j in range(out.shape[0])]
                self.list_w_label += [out[j] for j in range(out.shape[0])]

        elif self.mode == 'new_with_st_index':

            self.data_size = end_ind - begin_ind + end_ind1 - begin_ind1
            self.list_img = []
            self.list_label = []
            self.c_label = []
            self.st_label = []
            # self.st_data = st_after_weak_set[0]
            device = torch.device("cuda")
            model = new_model.to(device)

            #model_file = './model/model_new.pth'
            #model.load_state_dict(torch.load(model_file))
            #model = nn.DataParallel(model)
            model.eval()
            
            model_file = './model/model.pth'
            model_annotator = Net().to(device)
            #model = nn.DataParallel(model)
            model_annotator.load_state_dict(torch.load(model_file))
            model_annotator.eval()
            
            #cifar10 does not have domain discrepancy, we pretend it has here by dividing it into the front and back parts.
            list_img_s = self.total_img[begin_ind: end_ind]
            list_img_t = st_after_weak[0][begin_ind1:end_ind1]



            print("predicting the weak soft label using annotator with weak data")
            batch_size = 100
            self.list_img = list_img_s
            
            for i in range(int(len(list_img_s)/batch_size)):
                if (i == int(len(list_img_s)/batch_size) -1) and (i * batch_size < len(list_img_s)):
                    imgs = list_img_s[(i)*batch_size:]
                else:
                    imgs = list_img_s[(i)*batch_size:(i+1)*batch_size]
                
                imgs = np.asarray([self.transform(img).numpy() for img in imgs])
                imgs = torch.tensor(imgs).to(device)
                
                out = model_annotator(imgs)
                out = F.softmax(out, dim=1)
                
                
                _, diff = model(imgs, out)

                out = out.data.cpu().numpy()
                diff = diff.data.cpu().numpy()
                
                #if we calculated the confidence of the difference, it would be
                #self.list_label += [softmax(out[j] + diff[j]\
                #    *(0.1/((np.linalg.norm(x=diff[j], ord=2)/math.pow(10.0,0.5)) + 0.1))\
                #     , axis=0) for j in range(out.shape[0])]
                #otherwise
                #self.list_label += [  softmax(out[j] + diff[j], axis=0),  for j in range(out.shape[0])]
                for j in range(out.shape[0]):
                    l1 = softmax(out[j] + diff[j], axis=0)
                    self.list_label.append(l1)
                    self.c_label.append(np.argmax(l1))
                    self.st_label.append(0)
            
            
            
            print("adding strong data")
            reconstruct_strong_data = True

            if reconstruct_strong_data:
                batch_size = 100
                self.list_img = np.concatenate((self.list_img, list_img_t), axis = 0)
                for i in range(int(len(list_img_t)/batch_size)):
                    if (i == int(len(list_img_t)/batch_size) -1) and (i * batch_size < len(list_img_t)):
                        imgs = list_img_t[(i)*batch_size:]
                    else:
                        imgs = list_img_t[(i)*batch_size:(i+1)*batch_size]
                    imgs = np.asarray([self.transform(img).numpy() for img in imgs])
                    imgs = torch.tensor(imgs).to(device)
                    
                    out = model_annotator(imgs)
                    out = F.softmax(out, dim=1)
                    #out = out.data.cpu().numpy()
                    
                    _, diff = model(imgs, out)
                    
                    out = out.data.cpu().numpy()
                    diff = diff.data.cpu().numpy()
                    
                    for j in range(out.shape[0]):
                        l1 = softmax(out[j] + diff[j], axis=0)
                        self.list_label.append(l1)
                        self.c_label.append(np.argmax(l1))
                        self.st_label.append(1)
            else:
                self.list_img = np.concatenate((self.list_img, list_img_t), axis = 0)
                for j in range(len(list_img_t)):
                    l1 = self.total_label[begin_ind1: end_ind1][j]
                    self.list_label.append(l1)
                    self.c_label.append(np.argmax(l1))
                    self.st_label.append(1)
            
            #self.list_img = [np.asarray(img).astype(float) for img in self.list_img]
        else:
            print('Undefined Dataset!')
            
        

    def __getitem__(self, item):
        if self.mode == 'train':
            #img = Image.open(self.list_img[item])
            img = self.list_img[item]
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor(label)
        elif self.mode == 'val':
            #img = Image.open(self.list_img[item])
            img = self.list_img[item]
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor([label])
        elif self.mode == 'test':
            pass
        elif self.mode == 'weak':
            #img = Image.open(self.list_img[item])
            img = self.list_img[item]
            label = self.list_label[item]
            #return self.transform(Image.fromarray(np.uint8(img))), torch.tensor(label)
            return self.transform(img), torch.tensor(label)
        elif self.mode == 'diff':
            #img = Image.open(self.list_img[item])
            img = self.list_img[item]
            diff = self.list_diff[item]
            label = self.list_label[item]
            w_label = self.list_w_label[item]
            return self.transform(img), torch.tensor(diff), torch.tensor(w_label)
        elif self.mode == 'new_with_st_index':
            #img = Image.open(self.list_img[item])
            img = self.list_img[item]
            label = self.list_label[item]
            c_label = self.c_label[item]
            st_label = self.st_label[item]
            return self.transform(img), torch.tensor(label), torch.tensor(c_label), torch.tensor(st_label)
        else:
            print('None')

    def __len__(self):
        return self.data_size
