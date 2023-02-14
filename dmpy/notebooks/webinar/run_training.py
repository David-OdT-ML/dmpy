# -*- coding: utf-8 -*-
"""
This is a test script file.
"""

from os import path
from dgbpy import mlapply as dgbmlapply

dtect_data = 'E:\\surveys'
surveynm = 'F3_Demo_2020'

nladir = path.join( dtect_data, surveynm, 'NLAs' )

exfilenm = path.join( nladir, 'malenov_input_8x8x16_subsample_dBG.h5' )
model = dgbmlapply.doTrain( exfilenm, outnm='new model test' )