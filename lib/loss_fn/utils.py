from typing import Union

import torch
from torch import Tensor


def getAuxOut(Prediction: Union[Tensor, tuple[Tensor, Tensor]]):
    AuxOut = None
    if isinstance(Prediction, (tuple, list)) and len(Prediction) == 2:
        PredLabel, AuxOut = Prediction
    elif isinstance(Prediction, dict):
        PredLabel = Prediction["prediction"]
        AuxOut = Prediction.get("auxout", None)
    elif isinstance(Prediction, Tensor):
        PredLabel = Prediction
    else:
        raise NotImplementedError(
            "For computing loss for segmentation task, we need prediction to be an instance of tuple or Tensor"
        )
    assert isinstance(PredLabel, Tensor)
    assert isinstance(AuxOut, (Tensor, type(None)))
    return PredLabel, AuxOut


def mixCriteria(opt, LossFn, Input, Prediction, Target):
    FinalPrediction = None
    if isinstance(Prediction, (Tensor, list, tuple, dict)) or not opt.loss_coeff:
        FinalLoss = LossFn(Input=Input, Prediction=Prediction, Target=Target)
        FinalPrediction = Prediction["prediction"] if isinstance(Prediction, dict) else Prediction
    elif isinstance(LossFn, (list, tuple)):
        LossList = [
            Factor * LossFn(
                Input=Input[i], Prediction=Prediction[i], Target=Target[i]
            ) for i, Factor in enumerate(opt.loss_coeff)
        ]
        FinalLoss = torch.stack(LossList, dim=0).sum(dim=0)
        FinalPrediction = Prediction[-1]
    else:
        raise NotImplementedError(
            "Not supporting prediction data type %s with loss type %s" 
            % (type(Prediction), type(LossFn))
        )
    
    if FinalPrediction is None: FinalPrediction = Prediction
    return FinalLoss, FinalPrediction