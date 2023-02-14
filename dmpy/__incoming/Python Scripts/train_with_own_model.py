# -*- coding: utf-8 -*-
"""
This is a test script file.
"""

from os import path
from dgbpy import hdf5 as dgbhdf5
from dgbpy import dgbkeras
from dgbpy import mlio as dgbmlio
from dgbpy import mlapply as dgbmlapply
from dgbpy import keystr as dgbkeys

from myPythonRepo import unet3d

dtect_data = 'E:\\surveys'
surveynm = 'F3_Demo_2020'

nladir = path.join( dtect_data, surveynm, 'NLAs' )


imgexfilenm = path.join( nladir, 'Fault_Likelihood_input_single_surv_single_attr_dBG.h5' )
imgdp = dgbmlapply.getScaledTrainingData( imgexfilenm, split=0.2 )

infos = imgdp[dgbkeys.infodictstr]
inpshape = infos[dgbkeys.inpshapedictstr]
nrattribs = dgbhdf5.getNrAttribs( infos )

model = unet3d.unet_AH( (*inpshape, nrattribs) )

trainpars = dgbkeras.keras_dict
trainpars['nbchunk'] = 1
trainpars['epoch'] = 10
trainpars['batch'] = 4
model = dgbkeras.train( model, imgdp, trainpars )

modeltype = dgbmlio.getModelType( infos )
outfnm = dgbmlio.getSaveLoc( 'new model test2', modeltype, None )
dgbmlio.saveModel( model, imgexfilenm, dgbkeys.kerasplfnm, infos, outfnm )