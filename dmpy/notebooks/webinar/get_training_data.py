# -*- coding: utf-8 -*-
"""
This is a test script file.
"""

from os import path
from dgbpy import mlapply as dgbmlapply

dtect_data = 'E:\\surveys'
surveynm = 'F3_Demo_2020'

nladir = path.join( dtect_data, surveynm, 'NLAs' )

pickssetexfilenm = path.join( nladir, 'malenov_input_8x8x16_subsample_dBG.h5' )
pickssetdp = dgbmlapply.getScaledTrainingData( pickssetexfilenm, split=0.2 )


wllexfilenm = path.join( nladir, 'Log_-_Lithology_supervised_prediction.h5' )
wlldp = dgbmlapply.getScaledTrainingData( wllexfilenm, split=0.2 )


imgexfilenm = path.join( nladir, 'Layers_input_2D.h5' )
imgdp = dgbmlapply.getScaledTrainingData( imgexfilenm, split=0.2 )
