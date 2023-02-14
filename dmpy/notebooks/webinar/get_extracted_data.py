# -*- coding: utf-8 -*-
"""
This is a test script file.
"""

from os import path
from dgbpy import mlio as dgbmlio

dtect_data = 'E:\\surveys'
surveynm = 'F3_Demo_2020'

nladir = path.join( dtect_data, surveynm, 'NLAs' )

pickssetexfilenm = path.join( nladir, 'malenov_input_8x8x16_subsample_dBG.h5' )
pickssetdp = dgbmlio.getTrainingData( pickssetexfilenm )


wllexfilenm = path.join( nladir, 'Log_-_Lithology_supervised_prediction.h5' )
wlldp = dgbmlio.getTrainingData( wllexfilenm )


imgexfilenm = path.join( nladir, 'Layers_input_2D.h5' )
imgdp = dgbmlio.getTrainingData( imgexfilenm )
