import json
import numpy as np
from PIL import Image
# from plot_utils import PlotMaps, PlotHeatmaps
import os
import random
import json
import torch
import numpy as np
from PIL import Image
from random import sample
import torch.nn.functional as F
import torch.utils.data as data
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datasets.base import BaseDataset, generateGTAnnot
import time
import h5py
import torchvision.transforms as transforms

class DataPreprocess():
    def __init__(self):
        self.r = 64
        self.w = 64
        self.h = 8

        self.sampling_ratio = 1
        self.duration = 600
        self.numFrames = 8
        self.numGroupFrames = 8
        self.numChirps = 16
        
        self.numKeypoints = 14
        self.dirRoot = '../jupyter/hupr/radar_processed/'
        self.idxToJoints = ["R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", 
         "L_Ankle", "Neck", "Head", "L_Shoulder", "L_Elbow", 
         "L_Wrist", "R_Shoulder", "R_Elbow", "R_Wrist"]
        self.random = True # 

        # # shu: comment out the generating gt
        # generateGTAnnot(cfg, phase)
        self.phase = 'test' #'train'
        self.gtFile = os.path.join(self.dirRoot, '%s_gt.json' % self.phase)
        self.coco = COCO(self.gtFile)
        self.imageIds = self.coco.getImgIds()
        # print(self.imageIds)
        self.VRDAEPaths_hori = []
        self.VRDAEPaths_vert = []
        self.folder = []
        self.file_names = []
        for name in self.imageIds:
            namestr = '%09d' % name
            self.folder.append(int(namestr[:4]))
            self.file_names.append(int(namestr[-4:]))
            self.VRDAEPaths_hori.append(os.path.join(self.dirRoot, 'single_%d/hori/%09d.npy'%(int(namestr[:4]), int(namestr[-4:]))))
            self.VRDAEPaths_vert.append(os.path.join(self.dirRoot, 'single_%d/vert/%09d.npy'%(int(namestr[:4]), int(namestr[-4:]))))
        # print('VRDAEPaths_hori', len(self.VRDAEPaths_hori))
        # print('VRDAEPaths_vert', len(self.VRDAEPaths_vert))
        print(len(self.folder))
        self.max_index = len(self.VRDAEPaths_hori)//self.sampling_ratio
        # self.annots = self._load_coco_keypoint_annotations()
    
    def getTransformFunc(self, phase):
        # print('transforming')
        transformFunc = transforms.Compose([
            transforms.ToTensor(),
            Normalize()
        ])
        return transformFunc   

    def preprocess(self, index):
        # index = 5
    #     index = index * self.sampling_ratio
        padSize = index % self.duration

        # 8 frames altogether, start with the index of the first frame, with the picked one in the middle
        idx = index - self.numGroupFrames//2 - 1

        VRDAEmaps_hori = torch.zeros((self.numGroupFrames, self.numFrames, 2, self.r, self.w, self.h))
        VRDAEmaps_vert = torch.zeros((self.numGroupFrames, self.numFrames, 2, self.r, self.w, self.h))
        

    #     # index: the middle frame, idx: the current frame in the numGroupFrames loop
        for j in range(self.numGroupFrames):
            # if the index is the first few frames, go to the first frame?
            # if (j + padSize) <= 0: # previous: self.numGroupFrames//2:
            # print(j+padSize)
            if (j + padSize) <= self.numGroupFrames//2:
                idx = index - padSize
            # previous: elif j > (self.duration - 1 - padSize) + self.numGroupFrames//2:
            # elif (j + padSize) >= (self.duration - 1):
            elif j > (self.duration - 1 - padSize) + self.numGroupFrames//2:
                idx = index + (self.duration - 1 - padSize)
            else:
                idx += 1
            VRDAEPath_hori = self.VRDAEPaths_hori[idx]
            VRDAEPath_vert = self.VRDAEPaths_vert[idx]
            # shu:
            # print(self.VRDAEPaths_hori[idx])
            VRDAERealImag_hori = np.load(VRDAEPath_hori, allow_pickle=True)
            VRDAERealImag_vert = np.load(VRDAEPath_vert, allow_pickle=True)
            
            # # print('loaded shape', VRDAERealImag_hori.shape) # loaded shape (16, 64, 64, 8)

            idxSampleChirps = 0
            self.transformFunc = self.getTransformFunc(self.phase)

            for idxChirps in range(self.numChirps//2 - self.numFrames//2, self.numChirps//2 + self.numFrames//2):
                # 8 - 4, 8 + 4, only take center part chirps
                VRDAEmaps_hori[j, idxSampleChirps, 0, :, :, :] = self.transformFunc(VRDAERealImag_hori[idxChirps].real).permute(1, 2, 0)
                VRDAEmaps_hori[j, idxSampleChirps, 1, :, :, :] = self.transformFunc(VRDAERealImag_hori[idxChirps].imag).permute(1, 2, 0)
                VRDAEmaps_vert[j, idxSampleChirps, 0, :, :, :] = self.transformFunc(VRDAERealImag_vert[idxChirps].real).permute(1, 2, 0)
                VRDAEmaps_vert[j, idxSampleChirps, 1, :, :, :] = self.transformFunc(VRDAERealImag_vert[idxChirps].imag).permute(1, 2, 0)
                idxSampleChirps += 1
            
            # print('VRDAEmaps_hori shape', VRDAEmaps_hori.shape) # torch.Size([8, 8, 2, 64, 64, 8])

        # shu: maybe the transform has a nan?
        VRDAEmaps_hori = torch.nan_to_num(VRDAEmaps_hori)
        VRDAEmaps_vert = torch.nan_to_num(VRDAEmaps_vert)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        VRDAEmaps_hori = VRDAEmaps_hori.float().to(self.device)
        VRDAEmaps_vert = VRDAEmaps_vert.float().to(self.device)
        # print(torch.is_tensor(VRDAEmaps_hori))
        

        # print('VRDAEmaps_hori shape', VRDAEmaps_hori.shape) # torch.Size([8, 8, 2, 64, 64, 8])

        VRDAmaps_hori = VRDAEmaps_hori.mean(dim=5) # no batch dimension
        VRDAmaps_vert = VRDAEmaps_vert.mean(dim=5)
        # print('VRDAmaps_hori shape', VRDAmaps_hori.shape)

        path = '/home/jupyter/hupr/net_preprocessed/single_'+str(self.folder[index])+'/'
        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
       
        # hdf5_file = h5py.File(path+ ("%09d.hdf5" % index) , 'w')
        # # Create a group for VRDAEmaps_hori and VRDAEmaps_vert
        # group_hori = hdf5_file.create_group('VRDAEmaps_hori')
        # group_vert = hdf5_file.create_group('VRDAEmaps_vert')
        # # Save VRDAEmaps_hori
        # hdf5_file.create_dataset('hori', data=VRDAmaps_hori.cpu().numpy())
        # # Save VRDAEmaps_vert
        # hdf5_file.create_dataset('vert', data=VRDAmaps_vert.cpu().numpy())
        # hdf5_file.close()

        file = {'hori': VRDAmaps_hori.cpu().numpy().astype(np.float16), 'vert': VRDAmaps_vert.cpu().numpy().astype(np.float16)}
        np.save(path+ ("%09d.npy" % self.file_names[index]), file)

    #     joints = torch.LongTensor(self.annots[index]['joints'])
    #     bbox = torch.FloatTensor(self.annots[index]['bbox'])
    #     imageId = self.annots[index]['imageId']

    #     return {'VRDAEmap_hori': VRDAmaps_hori,
    #             'VRDAEmap_vert': VRDAmaps_vert,
    #             'imageId': imageId,
    #             'jointsGroup': joints,
    #             'bbox': bbox}

class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, radarData):
        # shu: scale it into [0,1]
        c = radarData.size(0)
        minValues = torch.min(radarData.view(c, -1), 1)[0].view(c, 1, 1)
        radarDataZero = radarData - minValues
        maxValues = torch.max(radarDataZero.view(c, -1), 1)[0].view(c, 1, 1)
        radarDataNorm = radarDataZero / maxValues

        # Shu: Then normalize it with mean and std
        std, mean = torch.std_mean(radarDataNorm.view(c, -1), 1)
        return (radarDataNorm - mean.view(c, 1, 1)) / std.view(c, 1, 1)

if __name__ == "__main__":
    DataPreprocess = DataPreprocess()
    for index in range(9000,13000): # for training 114000, val 12600, test 12600
        DataPreprocess.preprocess(index)
        print(index)