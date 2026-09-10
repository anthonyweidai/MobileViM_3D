import torch


def saveModel(SavePath, Model):
    if isinstance(Model, torch.nn.DataParallel):
        StateDict = Model.module.state_dict()
    else:
        StateDict = Model.state_dict()
    torch.save(StateDict, SavePath)