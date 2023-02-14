from keras import backend as K
import tensorflow as tf

def waveloss(ValW, SpaW, ValInc=0.1, SpaInc=8, NumSteps=10, labelwise=False):
    def _waveloss(y_true, y_pred):  
        y_true = K.cast(y_true,"float32")

        #IntersSection = tf.math.minimum(y_pred, y_true)
        Union = tf.math.maximum(y_pred, y_true)

        CurrentWave = tf.math.minimum(y_pred, y_true)
        WaveLoss = 0

        for step in range(NumSteps):
            Growed = CurrentWave + ValInc
            Growed = tf.math.minimum(Growed, Union)
            ValueDiff = tf.reduce_sum(Growed - CurrentWave)
            Growed = tf.nn.max_pool3d(Growed, SpaInc, 1, padding='SAME', data_format='NCDHW')
            Growed = tf.math.minimum(Growed, Union)
            TopologyDiff = tf.reduce_sum(Growed - CurrentWave)
            CurrentWave = Growed
            WaveLoss = WaveLoss + ValW[step] * ValueDiff + SpaW[step] * TopologyDiff

        return WaveLoss

    if labelwise:
        def _labelwise_waveloss(y_true, y_pred):
            return _waveloss(y_true[:,0:1,:,:,:], y_pred[:,0:1,:,:,:]) + _waveloss(y_true[:,1:2,:,:,:], y_pred[:,1:2,:,:,:]) + _waveloss(y_true[:,2:,:,:,:], y_pred[:,2:,:,:,:])
        return _labelwise_waveloss

    return _waveloss