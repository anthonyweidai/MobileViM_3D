import os
import re
from glob import glob
from pathlib import Path

import functools


def getImgPath(DatasetPath, NumSplit, Mode=1, Shuffle=True):
    # Put images into train set or test set
    from sklearn.model_selection import KFold
    
    TrainSet, TestSet = [], [] # init
    if Mode == 1:
        """ Cross-validation split without class folder
        root/split1/dog_1.png
        root/split1/dog_2.png
        root/split2/cat_1.png
        root/split2/cat_2.png
        """
        for i in range(1, NumSplit + 1):
            TestSet.append(glob("%s/split%d/*" % (DatasetPath, i)))
            
            TrainImgs = []
            for j in range(1, NumSplit + 1):
                if j != i:
                    TrainImgs.extend(glob("%s/split%d/*" % (DatasetPath, j)))
            TrainSet.append(TrainImgs)
        
    elif Mode == 2:
        """ No split with only class folder
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png
        """
        TrainSet, TestSet = [[] for _ in range(NumSplit)], [[] for _ in range(NumSplit)]
        ClassNames = os.listdir(DatasetPath)
        Kf = KFold(n_splits=NumSplit, shuffle=Shuffle)
        
        for ClassName in ClassNames:
            ImagePath = glob("%s/%s/*" % (DatasetPath, ClassName))
            IndexList = range(0, len(ImagePath))

            Kf.get_n_splits(IndexList)
            
            for idx, (TrainIndexes, TestIdexes) in enumerate(Kf.split(IndexList)):
                [TrainSet[idx].append(ImagePath[i]) for i in TrainIndexes]
                [TestSet[idx].append(ImagePath[j]) for j in TestIdexes]
                
    elif Mode == 3:
        """ Train and val/test split with class folder
        root/train/dog/xxx.png
        root/train/dog/xxy.png
        root/train/dog/[...]/xxz.png
        root/val/dog/xxx.png
        root/val/dog/xxy.png
        root/val/dog/[...]/xxz.png
        or,
        root/train/dog/xxx.png
        root/train/dog/xxy.png
        root/train/dog/[...]/xxz.png
        root/test/dog/xxx.png
        root/test/dog/xxy.png
        root/test/dog/[...]/xxz.png
        """
        TrainSet = glob("%s/train/*/*" % DatasetPath)
        
        Level = adaptValTest(DatasetPath, ValidStr=["test", "val"])
        TestSet = glob("%s/%s/*/*" % (DatasetPath, Level))
        
    elif Mode == 4:
        """  Train and val/test split without class folder
        root/train/xxx.png
        root/train/xxy.png
        root/train/[...]/xxz.png
        root/val/xxx.png
        root/val/xxy.png
        root/val/[...]/xxz.png
        or,
        root/train/xxx.png
        root/train/xxy.png
        root/train/[...]/xxz.png
        root/test/xxx.png
        root/test/xxy.png
        root/test/[...]/xxz.png
        """
        TrainSet = glob("%s/train/*" % DatasetPath)
        
        Level = adaptValTest(DatasetPath, ValidStr=["test", "val"])
        TestSet = glob("%s/%s/*" % (DatasetPath, Level))
        
    return TrainSet, TestSet


def getSetPath(TrainSet, TestSet, Split):
    # for source and target domain set path init
    SamplePath = TrainSet[0] if TrainSet else TestSet[0]
    if isinstance(SamplePath, str):
        TrainSet = TrainSet
        TestSet = TestSet
    else:
        TrainSet = TrainSet[Split]
        TestSet = TestSet[Split]
    
    return TrainSet, TestSet


def getSubdirectories(Dir):
    return [SubDir for SubDir in os.listdir(Dir)
            if os.path.isdir(os.path.join(Dir, SubDir))]


def expFolderCreator(BaseFolder, TaskType, ExpLevel="", TargetExp=None, Mode=0):
    # Count the number of exsited experiments
    FolderPath = "./%s/%s/%s" % (BaseFolder, TaskType, ExpLevel) 
    Path(FolderPath).mkdir(parents=True, exist_ok=True)
    
    ExpList = getSubdirectories(FolderPath)
    if TargetExp:
        ExpCount = TargetExp
    else:
        if len(ExpList) == 0:
            ExpCount = 1
        else:
            MaxNum = 0
            for idx in range(len(ExpList)):
                NumStr = re.findall("\d+", ExpList[idx])
                if NumStr: # should not be empty
                    temp = int(NumStr[0]) + 1
                    if MaxNum < temp:
                        MaxNum = temp
            ExpCount = MaxNum if Mode == 0 else MaxNum - 1
    
    DestPath = "%s/exp%s/" % (FolderPath, str(ExpCount))
    Path(DestPath).mkdir(parents=True, exist_ok=True)
    return DestPath, ExpCount


def getOnlyFileNames(PathStr):
    # get only file names among a path
    # # equivalent code:
    # [f for f in os.listdir(PathStr) if os.path.isfile(f)]
    return next(os.walk(PathStr))[2] if os.path.isdir(PathStr) else []


def getOnlyFolderNames(PathStr):
    # get only folder names among a path
    # # equivalent code:
    # FolderNames = [Name for Name in os.listdir(PathStr) \
    #     if os.path.isdir(os.path.join(PathStr, Name))]
    return next(os.walk(PathStr))[1] if os.path.isdir(PathStr) else []


def getOnlyFileDirs(PathStr):
    # get only file directories among a path
    FileNames = getOnlyFileNames(PathStr)
    return [os.path.join(PathStr, n) for n in FileNames]


def getOnlyFolderDirs(PathStr):
    # get only folder directories among a path
    FolderNames = getOnlyFolderNames(PathStr)
    return [os.path.join(PathStr, n) for n in FolderNames]


def getFileNameWithoutExt(PathStr):
    # get only file name without extension in a file path
    FileName = Path(PathStr).stem
    Idx = FileName.find('.')
    if 0 < Idx <= len(FileName) - 1:
        FileName = FileName[:Idx]
    return FileName


def getFileExtension(PathStr):
    # get file extension even when the file has dual extention
    # e.g., .nii.gz
    return "".join(Path(PathStr).suffixes)


def splitChecker(SetName: str, RSplit=False):
    # check if the dataset has multiple splits
    IsSplit = False
    
    if "split" in SetName:
        # for test mode
        NumSplit = 2
        IsSplit = True
    else:
        NumCand = re.findall(r"S(\d+)", SetName)
        if NumCand:
            NumSplit = int(NumCand[-1])
            if NumSplit <= 10: # no more than 10 split
                IsSplit = SetName.endswith("S%d" % NumSplit)
    
    if RSplit:
        # return the number of splits
        if not IsSplit: NumSplit = 1
        return IsSplit, NumSplit
    else:
        return IsSplit
    

def adaptValTest(DatasetPath, ValidStr=["test", "val"]):
    for Name in ValidStr:
        SetPath = "%s/%s/" % (DatasetPath, Name)
        if os.path.isdir(SetPath):
            break
    return Name


def replacedWithMask(ImgPaths):
    ReplaceStrs = {
        ".jpg": ".png", 
        **dict.fromkeys(
            ["/train", "/val", "/test", "\\train", "\\val", "\\test"], "/mask"
        ),
    }
    MaskPaths = [
        functools.reduce(lambda a, kv: a.replace(*kv), ReplaceStrs.items(), p)
        for p in ImgPaths
    ]
    return MaskPaths