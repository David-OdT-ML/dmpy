from dgbpy.keras_classes import UserModel
from dgbpy.keras_classes import UserModel, DataPredType, OutputType, DimType
import monolithic.dGBUNet as unet
import dgbpy.mlmodel_keras_dGB as dGB


class dGB_UnetReg(UserModel):
    uiname = 'dGB UNet Regression'
    uidescription = 'dGBs Unet image regression'
    predtype = DataPredType.Continuous
    outtype = OutputType.Image
    dimtype = DimType.Any

    def _make_model(self, model_shape, nroutputs, learnrate):
        model = unet.build(model_shape, nroutputs, self.predtype)
        model = dGB.compile_model(model, nroutputs, True, True, learnrate)
        return model
