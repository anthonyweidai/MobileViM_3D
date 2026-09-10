# path manager, system
import time
from tqdm import tqdm
# torch module
import torch
from torch import nn
# my module
from .baseTrainer import BaseTrainer
from ..model import getModel
from ..loss_fn import mixCriteria
from ..optim import buildOptimiser
from ..optim.scheduler import buildScheduler
from ..utils import WinCols, colourText, moveToDevice


class Trainer(BaseTrainer):
    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.TqCols = WinCols - 9
        
    def runnerInit(self, Rnd, Split):
        # separate initialisation for inferFeature
        self.Rnd = Rnd
        self.Split = Split
        
        self.splitInit()
        self.dataLoaderSetting(self.opt)
        self.Model = getModel(opt=self.opt).to(self.opt.device)
        
        self.Optim = buildOptimiser(self.Model.parameters(), self.opt)
        self.Scheduler = buildScheduler(self.opt, Optimiser=self.Optim)
        
    def run(self, Rnd, Split):
        self.runnerInit(Rnd, Split)
        self.LrList = []
        
        # Start tranining
        print((" Training%s %s " % (" and Validating" if self.ValDL else "", 
                                    colourText(str(Split + 1)))).center(WinCols, "*"))
        for Epoch in range(self.opt.epochs):
            time.sleep(0.1)  # To prevent possible deadlock during epoch transition
            
            self.TrainSampler.epochInit(Epoch)
            self.TrainDL.dataset.callerInit()
            
            self.training(Epoch)
            if self.ValDL:
                if "5nn" in self.opt.clsval_mode:
                    self.generateFeatureBank()
                if Epoch >= self.opt.val_start_epoch:
                    self.validating(Epoch)
            
            if self.opt.lr_decay:
                self.Scheduler.step(Epoch=Epoch)
                self.CurrentLrRate = self.Scheduler.LastLR[0] 
            else:
                self.CurrentLrRate = self.opt.lr
            self.LrList.append(self.CurrentLrRate)

            # Use a barrier() to make sure that all processes have finished
            self.bestManager(self.opt, Epoch, self.Model)
            
    def training(self, Epoch):
        # Normalization is different in trainning and evaluation
        self.Model.train()
        if isinstance(self.LossFn, list):
            [l.eval() for l in self.LossFn]
        else:
            self.LossFn.train()
        self.TrainState.epochParamsInit()
        self.Optim.zero_grad()  # Initialize gradient
        
        for Batch in tqdm(self.TrainDL, total=self.TrainIters, ncols=self.TqCols, colour="magenta"):
            Batch = moveToDevice(Batch, self.opt.device)
            Img = Batch["image"]
            Label = Batch.get("label", None)
            
            PredLabel = self.Model(Img)  # prediction
            Loss, FinalPredLabel = mixCriteria(
                self.opt, self.LossFn, Img, PredLabel, Label
            )
            # use clip gradients when the error derivative is changed or clipped to a threshold
            # https://stackoverflow.com/a/66659607/15329637
            if self.opt.grad_clip is not None:
                # ensure that your model gets reasonably sized gradients
                nn.utils.clip_grad_norm_(
                    self.Model.parameters(), max_norm=self.opt.grad_clip,
                )
            
            # preventing accumulation
            self.Optim.zero_grad()
            
            self.TrainState.update(FinalPredLabel, Label, Loss, Epoch)
        
        self.TrainState.averageMetrics()

    @torch.no_grad()
    def validating(self, Epoch):
        # Normalization is different in trainning and evaluation
        self.Model.eval()
        if isinstance(self.LossFn, list):
            [l.eval() for l in self.LossFn]
        else:
            self.LossFn.eval()
        self.ValState.epochParamsInit()
        
        for Batch in tqdm(self.ValDL, total=self.ValIters, ncols=self.TqCols, colour="magenta"):
            Batch = moveToDevice(Batch, self.opt.device)
            Img = Batch["image"]
            Label = Batch["label"]
            
            PredLabel = self.Model(Img)
            Loss, FinalPredLabel = mixCriteria(
                self.opt, self.LossFn, Img, PredLabel, Label,
            )
            self.ValState.update(FinalPredLabel, Label, Loss, Epoch)
            
            self.ValState.update(FinalPredLabel, Label, None, Epoch)
        
        self.ValState.averageMetrics()
            
        torch.cuda.empty_cache()