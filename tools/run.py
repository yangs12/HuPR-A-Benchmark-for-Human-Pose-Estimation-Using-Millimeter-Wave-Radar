import os
import torch
import numpy as np
import torch.optim as optim
from models import HuPRNet
from misc import plotHumanPose
from datasets import getDataset
import torch.utils.data as data
import torch.nn.functional as F
from tools.base import BaseRunner
import wandb
import logging
import time
logging.basicConfig(level=logging.DEBUG)
# logging.getLogger().setLevel(logging.INFO)
# logging.debug("test")
# logging.info("test")
# logging.warning("test")
# logging.error("test")
# logging.critical("test")


class Runner(BaseRunner):
    def __init__(self, args, cfg):
        super(Runner, self).__init__(args, cfg)    
        if not args.eval:
            self.trainSet = getDataset('train', cfg, args)
            self.trainLoader = data.DataLoader(self.trainSet,
                                  self.cfg.TRAINING.batchSize,
                                  shuffle=True,
                                  num_workers=cfg.SETUP.numWorkers,
                                  timeout=120,
                                  persistent_workers=True,)
                                #   pin_memory=True)
        else:
            self.trainLoader = [0] # an empty loader
        self.testSet = getDataset('test' if args.eval else 'val', cfg, args)
        self.testLoader = data.DataLoader(self.testSet, 
                              self.cfg.TEST.batchSize,
                              shuffle=False,
                              num_workers=cfg.SETUP.numWorkers,
                              timeout=120,)
                            #   persistent_workers=True,)
                            #   pin_memory=True) #cfg.SETUP.numWorkers)
        self.model = HuPRNet(self.cfg).to(self.device)
        self.stepSize = len(self.trainLoader) * self.cfg.TRAINING.warmupEpoch
        LR = self.cfg.TRAINING.lr if self.cfg.TRAINING.warmupEpoch == -1 else self.cfg.TRAINING.lr / (self.cfg.TRAINING.warmupGrowth ** self.stepSize)
        self.initialize(LR)
        self.beta = 0.0
    
    def eval(self, visualization=True, epoch=-1):
        logging.debug("Begin eval")
        self.model.eval()
        loss_list = []
        self.logger.clear(len(self.testLoader.dataset))
        logging.debug("clear logger")
        savePreds = []
        for idx, batch in enumerate(self.testLoader):
            # shu:
            if idx == 20:
                break

            logging.debug(f"Eval Batch idx: {idx}")
            keypoints = batch['jointsGroup']
            bbox = batch['bbox']
            imageId = batch['imageId']
            with torch.no_grad():
                VRDAEmaps_hori = batch['VRDAEmap_hori'].float().to(self.device)
                VRDAEmaps_vert = batch['VRDAEmap_vert'].float().to(self.device)

                # logging.debug("Predicting")
                preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert)
                # logging.debug("Calculating loss")
                loss, loss2, preds, gts = self.lossComputer.computeLoss(preds, keypoints)
                # logging.debug("Loss calculated")
                self.logger.display(loss, loss2, keypoints.size(0), epoch)

                if visualization:
                    plotHumanPose(preds*self.imgHeatmapRatio, self.cfg, 
                                  self.visDir, imageId, None)
                    # for drawing GT
                    # plotHumanPose(gts*self.imgHeatmapRatio, self.cfg, 
                    #               self.visDir, imageId, None)

            self.saveKeypoints(savePreds, preds*self.imgHeatmapRatio, bbox, imageId)
            loss_list.append(loss.item())
        # wandb.log({"Eval total loss": np.mean(loss_list)})
        # logging.debug("Write keypoints")
        self.writeKeypoints(savePreds)
        if self.args.keypoints:
            accAP = self.testSet.evaluateEach(self.dir)
        accAP = self.testSet.evaluate(self.dir)
        return accAP

    def train(self):
        # wandb.watch(self.model, log="all", log_freq=100)

        for epoch in range(self.start_epoch, self.cfg.TRAINING.epochs):
            logging.debug(f"Training Epoch: {epoch}")
            self.model.train()
            loss_list = []
            self.logger.clear(len(self.trainLoader.dataset))

            start_load = time.time()
            for idxBatch, batch in enumerate(self.trainLoader):
                after_load = time.time()
                print('\nloading time',  after_load - start_load)

                # logging.debug(f"Batch idxBatch: {idxBatch}")
                # # # shu: to quickly jump the training
                if idxBatch == 20:
                    break

                # if idxBatch < 2:
                self.optimizer.zero_grad()
                keypoints = batch['jointsGroup']
                bbox = batch['bbox']
                VRDAEmaps_hori = batch['VRDAEmap_hori'].float().to(self.device)
                VRDAEmaps_vert = batch['VRDAEmap_vert'].float().to(self.device)
                # print('torch in run', torch.is_tensor(VRDAEmaps_hori))
                
                # logging.debug("pred model")

                preds = self.model(VRDAEmaps_hori, VRDAEmaps_vert)
                # logging.debug("calculating loss")
                loss, loss2, _, _ = self.lossComputer.computeLoss(preds, keypoints)
                
                # wandb.log({"Batch train total loss": loss, "Batch train loss2": loss2})

                loss.backward()
                self.optimizer.step()                 
                self.logger.display(loss, loss2, keypoints.size(0), epoch)
                if idxBatch % self.cfg.TRAINING.lrDecayIter == 0: #200 == 0:
                    # logging.debug("Adjusting LR")
                    self.adjustLR(epoch)
                loss_list.append(loss.item())

                # after_train= time.time()
                # print('\ntraining time',  after_train - after_load)
            
            accAP = self.eval(visualization=False, epoch=epoch)
            self.saveModelWeight(epoch, accAP)
            self.saveLosslist(epoch, loss_list, 'train')
                
            # # wandb.log({"Epoch train total loss": loss, "Epoch train loss2": loss2})