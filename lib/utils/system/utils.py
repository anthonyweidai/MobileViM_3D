import sys

import random
import argparse
import numpy as np
from functools import partial
from typing import Dict, List, Iterable, Tuple, TypeVar, Union, Callable, Optional

import torch
import torchvision
from torch import Tensor

from ..utils import importModule
from ..variables import TextColors
from ..mathematics.utils import makeDivisible


def colourText(InputText: str, Mode=1, ColourName="") -> str:
    if ColourName:
        return TextColors[ColourName] + InputText + TextColors["end_colour"]
    else:
        if Mode == 1:
            # cyan colour, params init/input params
            return TextColors["light_cyan"] + InputText + TextColors["end_colour"]
        elif Mode == 2:
            # yellow colour, results output
            return TextColors["light_yellow"] + InputText + TextColors["end_colour"]
        elif Mode == 3:
            # green colour, others
            return TextColors["light_green"] + InputText + TextColors["end_colour"]


def colourTextList(InputText: list, Mode=1, ColourName="") -> list:
    return [colourText(str(t), Mode, ColourName) for t in InputText]
    
    
def getSysVersion(FullMode=True):
    if FullMode:
        print("Python version is:", colourText(sys.version))
    print("CUDA version is:", colourText(torch.version.cuda))
    print("PyTorch version is:", colourText(torch.__version__))
    if FullMode:
        print("Torchvision version is:", colourText(torchvision.__version__))


def seedSetting(RPMode, Seed=999):
    # Set random seed for reproducibility
    if RPMode:
        #manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random seed are set to: ", Seed)
        random.seed(Seed)
        np.random.seed(Seed)
        torch.manual_seed(Seed)
    return


def getLoaderBatch(opt):
    return opt.batch_size


def moveToDevice(
    x: Union[Dict, Tensor],
    Device: Optional[str] = "cpu",
    non_blocking: Optional[bool] = True,
    **kwargs
) -> Union[Dict, Tensor]:

    if isinstance(x, Dict):
        # return the tensor because if its already on Device
        if "on_gpu" in x and x["on_gpu"]:
            return x

        for k, v in x.items():
            if isinstance(v, Dict):
                x[k] = moveToDevice(x=v, Device=Device)
            elif isinstance(v, Tensor):
                x[k] = v.to(Device, non_blocking=non_blocking)

    elif isinstance(x, Tensor):
        x = x.to(Device, non_blocking=non_blocking)
    else:
        print(
            "Inputs of type Tensor or Dict of Tensors are only supported right now"
        )
    return x


class Registry(object):
    RegistryItem = TypeVar("RegistryItem", bound=Callable)
    def __init__(
        self, CurrentPath, BaseClass: type=None, Separator: str=":", **kwargs,
    ) -> None:
        """
        Args:
            BaseClass: If provided, will ensure that all items inside the registry
                are of type `base_class`.
            LoadDirs: If provided, will load all directories under these
                directories when inspecting for the modules of the registry.
        """
        self.LoadDir = CurrentPath
        self.ModulesLoaded = False
        
        self.Registry = {}
        self.BaseClass = BaseClass
        
        # For debugging purposes we want to throw a warning if someone accesses
        # arguments before registering all items.
        self.ArgumentsAccessed = False
        self.Separator = Separator
        
        self.kwargs = kwargs
        
    def loadAll(self) -> None:
        # Automatically import the load direcotry
        # cannot be run inside the LoadDir subfolder
        if not self.ModulesLoaded:
            self.ModulesLoaded = True
            importModule(self.LoadDir, **self.kwargs)
            
    def __iter__(self) -> Iterable[str]:
        self.loadAll()
        return iter(self.Registry)

    def __getitem__(self, Key: Union[Tuple[str, str], str]) -> RegistryItem:
        self.loadAll()

        type_ = None
        if isinstance(Key, Tuple) and len(Key) == 2:
            Key, type_ = Key

        assert isinstance(
            Key, str
        ), f"Key should be an instance of string. Got {type(Key)}"
        Name, Params = self.parseKey(Key)
        if type_:
            Name = f"{type_}{self.Separator}{Name}"

        if Name not in self.Registry:
            RegistryKeys = list(self.Registry.keys())
            TempStr = (
                f"\n{Name} is not yet supported."
                f"\nSupported values are:"
            )
            for i, s in enumerate(RegistryKeys):
                TempStr += "\n\t %d: %s" % (i, colourText(s))
            print(TempStr + "\n")

        RegItem = self.Registry[Name]

        if Params:
            RegItem = partial(RegItem, **Params)
        return RegItem
    
    def __contains__(self, Key: str) -> bool:
        self.loadAll()
        Name, _ = self.parseKey(Key)
        return Name in self.Registry
    
    def register(self, *arg):
        # register module (fuction, class) in a registry
        # so we can call the function by string name
        if isinstance(arg, (list, Tuple)):
            Name = self.Separator.join(arg)
        elif isinstance(arg, str):
            Name = arg
        
        def registerModule(Module):
            if Name in self.Registry:
                raise ValueError("Cannot register duplicate module %s" % (Name))
            if self.BaseClass is not None and not issubclass(Module, self.BaseClass):
                raise ValueError("Class %s must extend %s" % (Name, self.BaseClass))
            
            self.Registry[Name] = Module
            return Module
        
        return registerModule

    def items(self) -> List[Tuple[str, RegistryItem]]:
        self.loadAll()
        return self.Registry.items()

    def keys(self) -> List[str]:
        self.loadAll()
        return self.Registry.keys()
    
    def allArguments(self, Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Iterates through all items and fetches their arguments.

        Note: make sure that all items are already registered before calling this method.
        """
        self.loadAll()
        self.ArgumentsAccessed = True

        for _, v in self.items():
            Parser = v.add_arguments(Parser)

        return Parser
    
    def parseKey(self, Key: str) -> Tuple[str, Dict[str, str]]:
        """
        Parses `Key` which can contain arguments in the form of:
        <key_name>(arg1=value1, arg2=value2, ...)
        """
        Params = {}
        if "(" in Key:
            ParamsStr = Key.split("(")[1].split(")")[0]

            try:
                Params = dict(
                    [
                        [x.strip() for x in arg.split("=")]
                        for arg in ParamsStr.split(",")
                    ]
                )
            except Exception as e:
                print(
                    "Could not correctly parse Key parameters `{}` for registry {}."
                    " Please make sure to Key parameters have the format:"
                    " <key_name>(arg1=value1, arg2=value2, ...)".format(
                        Key, self.registry_name
                    )
                )
                raise e
        return Key.split("(")[0], Params
