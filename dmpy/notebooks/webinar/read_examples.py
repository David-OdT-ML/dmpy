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
pickssetsinfos = dgbmlio.getInfo( pickssetexfilenm, quick=True )

wllexfilenm = path.join( nladir, 'Log_-_Lithology_supervised_prediction.h5' )
wllinfos = dgbmlio.getInfo( wllexfilenm, quick=True )

imgexfilenm = path.join( nladir, 'Layers_input_2D.h5' )
imginfos = dgbmlio.getInfo( imgexfilenm, quick=True )

