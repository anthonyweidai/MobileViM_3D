import os
import importlib
import numpy as np

from torch import nn


def importModule(CurrentPath, PurePathMode: bool=False, SkipFolder: list=[]):
    # Automatically import the modules
    SubFolds = [""] # relative pure directory
    DefaultSkip = ["__pycache__", "config"] # default skip folders
    RegisterPath = os.path.dirname(CurrentPath.replace(os.getcwd(), ""))
    RegisterPath = RegisterPath.removeprefix("\\").removeprefix("/")
    if not PurePathMode:
        SubFolds.extend(next(os.walk(RegisterPath))[1]) # return folder only list
        SubFolds = [f for f in SubFolds if f not in SkipFolder + DefaultSkip]
    
    RegisterPath = RegisterPath.replace("\\", ".").replace("/", ".")
    for f1 in SubFolds:
        ModulesDir = "%s/%s" % (os.path.dirname(CurrentPath), f1)
        RelativePath = "%s.%s" % (RegisterPath, ("%s." % f1) if len(f1) > 0 else "")
        
        if os.path.isdir(ModulesDir):
            for f2 in [f for f in os.listdir(ModulesDir) if f not in SkipFolder + DefaultSkip]:
                JoinPath = os.path.join(ModulesDir, f2)
                if (
                        not f2.startswith("_") 
                        and not f2.startswith(".") 
                        and (f2.endswith(".py") or os.path.isdir(JoinPath))
                ):
                    if f2.endswith(".py") or f1 == "":
                        ModuleName = f2[: f2.find(".py")] if f2.endswith(".py") else f2
                        _ = importlib.import_module(RelativePath + ModuleName)
                    elif not PurePathMode:
                        importModule(JoinPath + "/__init__.py", SkipFolder=SkipFolder)


def setMethod(self, ElementName, ElementValue):
    """ Use self.add_module if it is defined within nn.Module class
    Any submodule that isn't declared as a direct attribute of the class
    will not be registered as a submodule of the model and its parameters won't show up.
    """
    if isinstance(self, nn.Module) and not isinstance(ElementValue, list):
        # register_buffer cannot register a list
        self.add_module(ElementName, ElementValue)
    else:
        setattr(self, ElementName, ElementValue)


def callMethod(self, ElementName):
    """ Akin to self.get_submodule(ElementName) within nn.Module class 
    Since it may not be an nn.Module, we don't use get_submodule function
    """
    return getattr(self, ElementName)


def unpair(Val):
    if isinstance(Val, (tuple, list)):
        OutVal = Val if len(Val) == 1 else Val[0]
    elif isinstance(Val, np.ndarray):
        OutVal = Val if Val.size == 1 else Val[0]
    else:
        OutVal = Val
    return OutVal


def pair(Val, RepNum=2):
    if RepNum == 1:
        return unpair(Val)
    
    if isinstance(Val, (tuple, list, np.ndarray)):
        Len = Val.size if isinstance(Val, np.ndarray) else len(Val)
        if Len == RepNum:
            OutVal = Val
        elif len(set(Val)) == 1:
            # ensure all values are the same
            OutVal = [Val[0] for _ in range(RepNum)]
        else:
            raise NotImplementedError
    else:
        OutVal = [Val for _ in range(RepNum)]
    return OutVal


def indicesSameEle(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def groupSort(ListGroup, KeyRule=lambda s: s.split("\\")[-1].split("_")[0]):
    # group elements in list with same substring 
    # images with same source -> similar features
    import itertools
    from itertools import groupby
    
    ListGroup = [list(i) for _, i in groupby(ListGroup, KeyRule)]
    np.random.shuffle(ListGroup)
    return list(itertools.chain(*ListGroup))
