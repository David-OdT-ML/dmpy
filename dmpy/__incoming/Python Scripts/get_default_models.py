# -*- coding: utf-8 -*-
"""
This is a test script file.
"""

from os import path
from dgbpy import dgbkeras
from dgbpy import keras_classes as kc
from dgbpy import mlio as dgbmlio

dtect_data = 'E:\\surveys'
surveynm = 'F3_Demo_2020'

nladir = path.join( dtect_data, surveynm, 'NLAs' )


pickssetexfilenm = path.join( nladir, 'malenov_input_8x8x16_subsample_dBG.h5' )
pickssetinfos = dgbmlio.getInfo( pickssetexfilenm, quick=True )
pickssetmodel = dgbkeras.getDefaultModel( pickssetinfos )
pickssetmodel.summary()

wllexfilenm = path.join( nladir, 'Log_-_Lithology_supervised_prediction.h5' )
wllinfos = dgbmlio.getInfo( wllexfilenm, quick=True )
wllmodel = dgbkeras.getDefaultModel( wllinfos )
wllmodel.summary()

imgexfilenm = path.join( nladir, 'Layers_input_2D.h5' )
imginfos = dgbmlio.getInfo( imgexfilenm, quick=True )
imgmodel = dgbkeras.getDefaultModel( imginfos )
imgmodel.summary()

"""
Mode explicit/direct model creation, without any data
"""
lenetmodelclss = kc.UserModel.getModelsByType( kc.DataPredType.Continuous,
                                               kc.OutputType.Pixel,
                                               kc.DimType.D3 )[0]
lenetmodel = lenetmodelclss.model( (5,9,17,3), 1, learnrate=1e-4 )
lenetmodel.summary()

unetsegmodelclss = kc.UserModel.getModelsByType( kc.DataPredType.Classification,
                                                 kc.OutputType.Image,
                                                 kc.DimType.D2 )[0]
unetsegmodel = unetsegmodelclss.model( (1,128,128), 1, learnrate=1e-4 )
unetsegmodel.summary()
