import keras.backend as K
import keras.losses
import keras.utils.generic_utils

import numpy as np

# ---------------------------------------------------------------------------------------------------
class RegularizedGaussNll(object):
    
    def __init__(self,reg_sigma=0.05):
        self.reg_sigma = reg_sigma
        self.n_params = 2
        self.__name__ = 'RegularizedGaussNll'
        
    def __call__(self,y_true,y_pred):
        mean   = y_pred[:,0]
        sigma2 = K.square(y_pred[:,1]) + self.reg_sigma**2
        z2 = K.square( y_true[:,0] - mean ) / sigma2
        return 0.5*K.mean( z2 + 0.5*K.log(sigma2), axis=-1 )

# ---------------------------------------------------------------------------------------------------
class HuberLoss(object):
    
    def __init__(self,delta=1.):
        self.delta = delta
        self.n_params = 1
        self.__name__ = 'HuberLoss'
        
    def __call__(self,y_true,y_pred):
        z = K.abs(y_true[:,0] - y_pred[:,0])
        mask = K.cast(K.less(z,self.delta),K.floatx())
        return K.mean( 0.5*mask*K.square(z) + (1.-mask)*(self.delta*z - 0.5*self.delta**2) )

# ---------------------------------------------------------------------------------------------------
class QuantileLoss(object):
    def __init__(self,taus=[0.5,0.25,0.75],weights=[1.,1.2,0.9]):
        self.taus = np.array(taus).reshape(1,-1)
        self.weights = np.array(weights).reshape(1,-1)
        self.n_params = len(taus)
        self.__name__ = 'QuantileLoss'
        
    def __call__(self,y_true,y_pred):
        e = y_true - y_pred
        print(e.shape)
        return K.mean( self.weights*( self.taus*e + K.clip( -e, K.epsilon(), np.inf ) ) )

# ---------------------------------------------------------------------------------------------------
class HybridLoss(object):

    def __init__(self,delta=1.,taus=[0.25,0.75],weights=[1.]):
        self.__name__ = 'HybridLoss'
        self.huber = HuberLoss(delta)
        self.quantiles = QuantileLoss(taus,weights)
        self.n_params = self.huber.n_params + self.quantiles.n_params
        
    def __call__(self,y_true,y_pred):
        return self.huber(y_true,y_pred) + self.quantiles(y_true,y_pred[:,1:])
    
    
# ---------------------------------------------------------------------------------------------------
def gauss_nll(y_true,y_pred):
    mean   = y_pred[:,0]
    sigma2 = K.square(y_pred[:,0]) + 0.01**2
    z2 = K.square( y_true[:,0] - mean) / sigma2
    return 0.5*K.mean( z2 + 0.5*K.log(sigma2), axis=-1 )
    ## return K.mean(K.square(y_true-y_pred))

# ---------------------------------------------------------------------------------------------------
def mse0(y_true,y_pred):
    return K.mean( K.square(y_true[:,0] - y_pred[:,0]) )

# ---------------------------------------------------------------------------------------------------
def mae0(y_true,y_pred):
    return K.mean( K.abs(y_true[:,0] - y_pred[:,0]) )

# ---------------------------------------------------------------------------------------------------
def r2_score0(y_true,y_pred):
    return 1. - K.sum( K.square(y_true[:,0] - y_pred[:,0]) ) / K.sum( K.square(y_true[:,0] - K.mean(y_true[:,0]) ) )


# ---------------------------------------------------------------------------------------------------
def masked_categorical_crossentropy(y_true,y_pred):
    
    return y_true[:,0] * keras.losses.categorical_crossentropy(y_true[:,1:],y_pred)
    

    
# ---------------------------------------------------------------------------------------------------
## keras.losses.gauss_nll = gauss_nll
## keras.utils.generic_utils.get_custom_objects().update(
##     dict(RegularizedGaussNll=RegularizedGaussNll,HuberLoss=HuberLoss)
## )
