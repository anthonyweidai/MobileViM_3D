import os


DEFAULT_DATA_ROOT = [
    "./dataset/", "../dataset/", "../../dataset/", # default, supplement,  
    "../autodl-tmp/", # auto-dl
    "../input/", # kaggle
]


def correctDatasetPath(DataPath, SetName: str=None):
    Count = 0
    while not os.path.isdir(DataPath):
        DataPath = DEFAULT_DATA_ROOT[Count]
        Count += 1
        
    if SetName is not None:
        if "../input/" in DataPath:
            FolderPath = DataPath + SetName.lower() + "/" + SetName # for kaggle
        else:
            FolderPath = DataPath + SetName
        return DataPath, FolderPath
    else:    
        return DataPath
