import copy
from typing import Union, Tuple

from torch import Tensor

from .baseSeg import BaseSegmentation
from .heads import buildSegmentationHead
from .. import MODEL_REGISTRY


@MODEL_REGISTRY.register("encoder_decoder", "segmentation")
class SegEncoderDecoder(BaseSegmentation):
    """ This class defines a encoder-decoder architecture for the task of semantic segmentation. 
    Different segmentation heads (e.g., PSPNet and DeepLabv3) can be used.
    Args:
        opt: command-line arguments
        Encoder (BaseEncoder): Backbone network (e.g., ResNext)
    """
    def __init__(self, opt, Encoder, **kwargs) -> None:
        super().__init__(opt, Encoder, **kwargs)
        # delete layers that are not required in segmentation network
        del self.Encoder.Classifier
        
        ModelConfigDict = copy.deepcopy(self.Encoder.ModelConfigDict) # no change in seghead
        if opt.seg_feature_guide and opt.fg_nostage5:
            # delete modules in the fifth stage
            MaxStage = 5
            for k, v in self.Encoder.ModelConfigDict.items():
                if v["stage"] >= MaxStage:
                    delattr(self.Encoder, k.capitalize())
                    del ModelConfigDict[k]
            self.Encoder.ModelConfigDict = ModelConfigDict
        
        self.SegHead = buildSegmentationHead(
            opt, HeadName=opt.seg_head_name, ModelConfigDict=ModelConfigDict,
        )
    
    def forward(self, x: Tensor, **kwargs) -> Union[Tuple[Tensor, Tensor], Tensor]:
        Input = self.preprocessInput(x)
        FeaturesTuple = self.Encoder.forwardTuple(Input)
        
        SegOutput = self.SegHead(FeaturesTuple, **kwargs)
        return SegOutput