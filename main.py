# torch module
import torch
# my module
from lib.trainer import Trainer
from lib.params_init import getArguments, overallParamsInit
from lib.utils import seedSetting, expFolderCreator


class Main(object):
    def __init__(self, opt) -> None:
        seedSetting(RPMode=False, Seed=opt.seed)
        
        if opt.target_supplement == 0:
            self.TargetExp = None
        else:
            self.TargetExp = opt.target_exp
            
        TaskType = opt.task
        
        # get target exp folder path and exp count
        DestPath, self.ExpCount = expFolderCreator(opt.exp_base, TaskType, opt.exp_level, self.TargetExp)
        opt.dest_path = DestPath
        
        self.ExpLogPath = "./%s/%s/%s/log_%s.csv" % (opt.exp_base, TaskType, opt.exp_level, opt.sup_method)
        
        self.opt = opt

    def running(self, **kwargs):
        if self.opt.num_split == 1 and self.opt.num_repeat > self.opt.num_split:
            SplitLoop = [0] * self.opt.num_repeat
        else:
            SplitLoop = range(self.opt.num_split)
        
        for i, Split in enumerate(SplitLoop):
            # end flag for general, split limit, supplement limit
            Round = i + self.opt.target_supplement
            Split += self.opt.target_supplement
            
            # run training and validating
            MyTrainer = Trainer(self.opt) # training class
            MyTrainer.run(Round, Split)
            
            torch.cuda.empty_cache()


if __name__ == "__main__":
    opt = getArguments()
    opt = overallParamsInit(opt)
    
    MyTrain = Main(opt)
    MyTrain.running()