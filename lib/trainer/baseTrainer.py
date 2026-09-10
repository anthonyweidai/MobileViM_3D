# path manager
import os
from pathlib import Path
# data processing
import math
from copy import deepcopy
from functools import partial
# torch module
import torch
from torch import nn
from torch.utils.data import DataLoader
# my module
from ..loss_fn import getLossFn
from ..metrics import Statistics
from ..data import  getDataset, buildSampler, buildCollateFn
from ..utils import (
    WeightMetrics, getImgPath, getSetPath, 
    workerManager, getLoaderBatch, saveModel, listPrinter,
)


class BaseTrainer(object):
    def __init__(self, opt) -> None:
        opt = workerManager(opt)
        self.opt = opt
        
        self.paramsInit()
        self.filePathInit()
        
        self.TrainState = Statistics(opt, IsTraining=True)
        self.ValState = Statistics(opt, IsTraining=False)
        self.MetricIndicator = [m for m in self.ValState.MetricListNames 
                                if self.opt.metric_indicator in m.lower()][0]
                
        LossFn: nn.Module = getLossFn(opt)
        self.LossFn = LossFn.to(opt.device)
    
    def paramsInit(self):
        self.Split = None
        self.Rnd = 0 # for best model saver
        
        self.TrainDL = None
        self.ValDL = None
        
        self.LrList = None
        self.CurrentLrRate = None
        
        self.TrainSet, self.TestSet = getImgPath(self.opt.dataset_path, self.opt.num_split, Mode=self.opt.get_path_mode)
        
        # drop last management, for batch normalisation
        NumOriTrainImgs = self.TrainSet.__len__()
        if NumOriTrainImgs % self.opt.batch_size == 1:
            self.opt.drop_last = True
        
        if "detection" in self.opt.task:
            if isinstance(self.TrainSet, list):
                NumOriTrainImgs = self.TrainSet[0].__len__() + self.TestSet[0].__len__()
            else:
                NumOriTrainImgs = self.TrainSet.__len__()
            NumTrainImgs = NumOriTrainImgs * self.opt.views
            NumIters = NumTrainImgs // self.opt.batch_size if \
                self.opt.drop_last else math.ceil(NumTrainImgs / self.opt.batch_size)
            self.opt.max_monitor_iter = round(NumIters * self.opt.milestones * 0.75)
            
            if self.opt.update_wt_freq is None:
                self.opt.update_wt_freq = round(NumIters * self.opt.milestones * 0.5)

    def splitInit(self):
        self.BestEpoch = 0
        self.BestLoss = 9e5
        self.BestComMetric = 0
        self.LastMinLossPath = ""
        self.LastMetricPath = ""
        self.BestStopIndicator = self.BestLoss if self.opt.metric_indicator == "loss" \
            else self.BestComMetric # Best metric indicator
    
    def filePathInit(self):
        self.RecordPath = self.opt.dest_path + "/metrics" # Save indicators during training
        self.ModelSavePath =  self.opt.dest_path + "/model"  # Save weight
        self.BestMetricsPath =  self.opt.dest_path + "/best_metrics.csv"  # Save metrics
        Path(self.RecordPath).mkdir(parents=True, exist_ok=True)
        Path(self.ModelSavePath).mkdir(parents=True, exist_ok=True)
    
    def dataLoaderSetting(self, opt):
        TrainSet, TestSet = getSetPath(self.TrainSet, self.TestSet, self.Split)
        
        opt.split = self.Split # for mean std init         
        self.TrainImgs = getDataset(opt, TrainSet, IsTraining=True)
        self.TestImgs = getDataset(opt, TestSet, IsTraining=False)
        
        CollateFn = buildCollateFn(opt)
        LoaderBatch = getLoaderBatch(opt)
        
        # You don’t need to shuffle the validation and test datasets, since no training is done, 
        # the model is used in model.eval() and thus the order of samples won’t change the results
        # https://discuss.pytorch.org/t/shuffle-true-or-shuffle-false-for-val-and-test-dataloaders/143720/5
        LoaderNames = ["Train"]
        self.TrainSampler = buildSampler(
            opt, NumSamples=self.TrainImgs.__len__(), LoaderBatch=LoaderBatch, IsTraining=True,
        )
        # self.TrainSampler = None
        InLoaderBatch = LoaderBatch if self.TrainSampler is None else 1
        InLoaderShuffle = opt.loader_shuffle if self.TrainSampler is None else False
        
        self.TrainDL = DataLoader(
            self.TrainImgs, InLoaderBatch, shuffle=InLoaderShuffle,
            batch_sampler=self.TrainSampler, num_workers=opt.num_workers, 
            collate_fn=partial(CollateFn, opt=opt), pin_memory=opt.pin_memory, drop_last=False,
            persistent_workers=False, prefetch_factor=opt.prefetch_factor,
        )
        
        self.ValSampler = None
        LoaderNames.append("Val")
        self.ValSampler = buildSampler(
            opt, NumSamples=self.TestImgs.__len__(), LoaderBatch=LoaderBatch, IsTraining=False,
        )
        # self.ValSampler = None
        InLoaderBatchVal = LoaderBatch if self.ValSampler is None else 1
        
        self.ValDL = DataLoader(
            self.TestImgs, InLoaderBatchVal, shuffle=False,
            batch_sampler=self.ValSampler, num_workers=opt.num_workers, 
            collate_fn=partial(CollateFn, opt=opt), pin_memory=opt.pin_memory, drop_last=False,
            persistent_workers=False, prefetch_factor=opt.prefetch_factor,
        )
        
        self.opt.max_train_iters = len(self.TrainDL.dataset) // LoaderBatch + (0 if opt.drop_last else 1)
        self.opt.max_train_iters *= opt.epochs
        
        # set tqdm iterations
        for n in LoaderNames:
            setattr(
                self, "%sIters" % n, 
                math.ceil(len(getattr(self, "%sDL" % n)) / LoaderBatch * InLoaderBatch)
            ) # drop last is unused
    
    @torch.no_grad()
    def modelSaver(self, opt, Epoch: int, Model):
        DataState = getattr(self, "%sState" % ("Val" if self.ValDL else "Train"))
        Loss = DataState.Loss
        ComMetric = DataState.ComMetric
        
        SavedMode = False
        SaveModelStr = ""
        SaveModelFlag = 0
        if Loss < self.BestLoss:
            SaveModelFlag = 2
            self.BestLoss = Loss
            SaveModelStr += "_minloss"
            if "loss" in opt.metric_indicator:
                SavedMode = True
                self.BestStopIndicator = self.BestLoss
                self.BestEpoch = Epoch
        if ComMetric > self.BestComMetric:
            SaveModelFlag += 3
            self.BestComMetric = ComMetric
            SaveStr = WeightMetrics.get(
                WeightMetrics.get("%s_%dd" % (opt.task, opt.spatial_dims), opt.task), ""
            )
            if SaveStr:
                SaveModelStr += ("_" + SaveStr)
            if "loss" not in opt.metric_indicator:
                SavedMode = True
                self.BestStopIndicator = self.BestComMetric
                self.BestEpoch = Epoch
        
        if "None" not in opt.save_point:
            # segmentation and detection models are super large
            # save only one model for segmentation and detection tasks
            if Epoch in opt.save_point:
                SaveModelFlag = 1
        
        if not SavedMode:
            # storage-friendly for tasks
            if SaveModelFlag == 1 or (opt.task in "classification" and 
                                      SaveModelFlag > 0 and Epoch >= opt.milestones):
                SavedMode = True
            
        if SavedMode:
            ModelName = opt.model_name
                
            SaveModelStr = "%s_epoch%d%s%s_f%s.pth" \
                % (ModelName, Epoch, SaveModelStr, 
                    ("_valrpt%d" % (self.SaveSign)) if "common" in opt.sup_method else "", SaveModelFlag)
            SavePath = "%s/%s" % (self.ModelSavePath, SaveModelStr)
            saveModel(SavePath, Model)
            
            if SaveModelFlag == 2:
                if os.path.isfile(self.LastMinLossPath):
                    if "_f2" in self.LastMinLossPath or \
                        ("loss" in opt.metric_indicator and "_f5" in self.LastMinLossPath):
                        os.remove(self.LastMinLossPath)
                        # print("The file has been deleted successfully")
                self.LastMinLossPath = SavePath
            elif SaveModelFlag == 3:
                if os.path.isfile(self.LastMetricPath):
                    if "_f3" in self.LastMetricPath or \
                        ("loss" not in opt.metric_indicator and "_f5" in self.LastMetricPath):
                        os.remove(self.LastMetricPath)
                        # print("The file has been deleted successfully")
                self.LastMetricPath = SavePath
            elif SaveModelFlag == 5:
                if os.path.isfile(self.LastMinLossPath):
                    os.remove(self.LastMinLossPath)
                if os.path.isfile(self.LastMetricPath):
                    os.remove(self.LastMetricPath)
                self.LastMinLossPath = SavePath
                self.LastMetricPath = SavePath

    @torch.no_grad()
    def bestManager(self, opt, Epoch: int, Model):
        Epoch = Epoch + 1 # for number convenience
        self.SaveSign = max(self.Split, self.Rnd) + 1
        self.modelSaver(opt, Epoch, Model)
        
        # printing the best and current metrics
        print("Epoch: [%d/%d], \tLearning rate: %.6f, %s" % (
            Epoch, opt.epochs, self.CurrentLrRate, 
            "\tCrossValid: [%d/%d]" % (self.Split + 1, opt.num_split) if self.ValDL else ""))
        
        LoopSet = ["Train", "Val"] if self.ValDL else ["Train"]
        
        ## best value, epoch, and train loss
        ListName = ["Best", "BEpoch", "Loss"]
        ListValue = [self.BestStopIndicator, self.BestEpoch, self.TrainState.Loss]
        
        ## current indicator metric
        if "loss" not in self.MetricIndicator.lower():
            IndiNames, IndiValues = [], []
            for s in LoopSet:
                if self.MetricIndicator in \
                    getattr(self, "%sState" % s).MetricListNames:
                    IndiNames.append(
                        "%s%s" % ("" if "Train" in s else "Val", self.MetricIndicator)
                    )
                    IndiValues.append(
                        getattr(getattr(self, "%sState" % s), "%s" % self.MetricIndicator)
                    )
            ListName = ListName[:2] + IndiNames + ListName[2:]
            ListValue = ListValue[:2] + IndiValues + ListValue[2:]
        
        # current train and validation metric
        for s in LoopSet:
            DataState = getattr(self, "%sState" % s)
            MetricListNames = deepcopy(DataState.MetricListNames)
            if s in "Train":
                MetricListNames.remove("Loss")
            MetricListNames = [n for n in MetricListNames if self.MetricIndicator not in n]
            
            NotPrintList = ["Loss", "Top5Acc", "mDice" if opt.spatial_dims == 2 else "mIoU"]
            ListName.extend([
                n if s in "Train" or n not in NotPrintList else "Val%s" % n 
                for n in MetricListNames
            ])
            ListValue.extend([getattr(DataState, n) for n in MetricListNames])
            
        listPrinter(ListName, ListValue, Mode=2, LineNum=6)