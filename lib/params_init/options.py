import argparse
from typing import List, Optional

from .arguments import (
    argumentsTask,
    argumentsDataset, argumentsAugmentation, argumentsSampler, argumentsCollateFn, 
    argumentsModeling, argumentsLoss, argumentsMetrics, argumentsOptimizer, argumentsScheduler,
    argumentsDDP, argumentsCommon,
)


def getArguments(
    ParseArgs: Optional[bool]=True, 
    args: Optional[List[str]]=None,
) -> argparse.Namespace:
    Parser = argparse.ArgumentParser(description="Training arguments", add_help=True)
    
    # task arguments
    Parser = argumentsTask(Parser=Parser)
    
    # dataset related arguments
    Parser = argumentsDataset(Parser=Parser)
    
    # sampler related arguments
    Parser = argumentsSampler(Parser=Parser)
    
    # collate fn related arguments
    Parser = argumentsCollateFn(Parser=Parser)

    # transform related arguments
    Parser = argumentsAugmentation(Parser=Parser)

    # cvnet arguments, including models
    Parser = argumentsModeling(Parser=Parser)
    
    # loss function arguments
    Parser = argumentsLoss(Parser=Parser)
    
    # metric arguments
    Parser = argumentsMetrics(Parser=Parser)

    # optimizer arguments
    Parser = argumentsOptimizer(Parser=Parser)
    Parser = argumentsScheduler(Parser=Parser)

    # DDP arguments
    Parser = argumentsDDP(Parser=Parser)
    
    # common
    Parser = argumentsCommon(Parser=Parser)
    
    if ParseArgs:
        return Parser.parse_args(args)
    else:
        return Parser