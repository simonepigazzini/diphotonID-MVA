from .ffwd import FFWDRegression

import numpy as np

from keras.layers import Input, Dense, Add, Multiply
from keras.layers import Reshape, UpSampling1D, Flatten, concatenate, Cropping1D
from keras.layers import Activation, LeakyReLU, PReLU, Lambda
from keras.layers import BatchNormalization, Dropout
from keras.models import Model, Sequential
from keras.layers import Layer
from keras.constraints import non_neg

import keras.optimizers

from keras.regularizers import l1,l2

from keras import backend as K

from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint

from sklearn.base import BaseEstimator

from sklearn.utils import shuffle

from . import losses 


# --------------------------------------------------------------------------------------------------
class PivotClassifier(BaseEstimator):
    
    def __init__(self, name, clf, dsc, 
                 dsc_optimizer='Adam', dsc_optimizer_params=dict(lr=1.e-04), 
                 adv_optimizer='Adam', adv_optimizer_params=dict(lr=1.e-04), 
                 ext_dsc_inputs=False, lambd=50.):
        
        self.lambd = lambd
        self.clf = clf
        self.dsc = dsc
        self.ext_dsc_inputs = ext_dsc_inputs
        
        self.dsc_optimizer = dsc_optimizer
        self.dsc_optimizer_params = dsc_optimizer_params
        self.adv_optimizer = adv_optimizer
        self.adv_optimizer_params = adv_optimizer_params
                
        self.ext_dsc_inputs = ext_dsc_inputs

    def __call__(self,docompile=False):
        
        clf = self.clf(docompile=False)
        dsc = self.dsc(docompile=False)
        
        wrapped_inputs = clf.outputs
        ## wrapped_inputs = [clf.outputs]
        ## if self.ext_dsc_inputs:
        ##     wrapped_inputs.append(clf.inputs)
            
        wrapped_clf = clf(clf.inputs)
        wrapped_adv = dsc(wrapped_inputs)
        
        self.dsc_model = Model( inputs=clf.inputs, outputs=[wrapped_adv] )
        self.adv_model = Model( inputs=clf.inputs, outputs=[wrapped_clf,wrapped_adv] )
        
        adv_optimizer = getattr(keras.optimizers,self.adv_optimizer)(**self.adv_optimizer_params)
        dsc_optimizer = getattr(keras.optimizers,self.dsc_optimizer)(**self.dsc_optimizer_params)
                
        dsc.trainable = False
        clf.trainable = True
        self.adv_model.compile(optimizer=adv_optimizer,
                               loss=[self.clf.loss,
                                     losses.masked_categorical_crossentropy],
                               loss_weights=[1.,-self.lambd]
        )
        
        dsc.trainable = True
        clf.trainable = False
        self.dsc_model.compile(optimizer=dsc_optimizer,
                               loss=losses.masked_categorical_crossentropy
        )
        
        return self.adv_model,self.dsc_model

        
    def fit(self,X_train,y_train,w_train=None,
            validation_data=None,
            epochs=1,n_adv_steps=1,n_dsc_steps=100,
            batch_size=256,
            print_every=10,
            pretrain=False,pretrain_clf_args=dict(),
            pretrain_dsc_args=dict(),
            syst_shift={}
    ):
        
        if pretrain:
            self.clf.fit(X,y[:,0],w,validation_data=validation_data,**pretrain_clf_args)

            y_pred = self.clf.predict(X)
            dsc_inputs = [y_pred]
            if self.ext_adv_inputs:
                dsc_inputs.append(X)
            self.dsc.fit(dsc_inputs,y[:,1:],w,validation_data=validation_data,**pretrain_dsc_args)
            
            
        adv_model,dsc_model = self(docompile=True)
        
        gen = Generator(X_train,y_train,w_train,batch_size,syst_shift=syst_shift)
        nbatches = gen.nbatches()
        nsteps = nbatches // (n_dsc_steps+n_adv_steps)

        for ei in range(epochs):
            ep_gen = gen()
            dsc_loss = []
            adv_loss = []
            for si in range(nsteps):
                for di in range(n_dsc_steps):
                    Xb,yb,y2b,wb = next(ep_gen)
                    dsc_loss.append( dsc_model.train_on_batch(Xb,y2b,wb) )
                for ai in range(n_adv_steps):
                    Xb,yb,y2b,wb = next(ep_gen)
                    adv_loss.append( adv_model.train_on_batch(Xb,[yb,y2b],[wb,wb]) )
                if si % print_every == 1:
                    print("dsc_loss: %f  adv_loss: %f\r" %( np.array(dsc_loss).mean(), np.array(adv_loss).mean() ), )
            
            print("dsc_loss: %f  adv_loss: %f" %( np.array(dsc_loss).mean(), np.array(adv_loss).mean() ), )
            if validation_data is not None:
                Xvalid, yvalid, wvalid = validation_data
                valid_dsc = dsc_model.evaluate(Xvalid,yvalid,sample_weight=wvalid,batch_size=batch_size)
                valid_adv = adv_model.evaluate(Xvalid,yvalid,sample_weight=wvalid,batch_size=batch_size)
                print(" dsc_loss_valid: %f  adv_loss_valid: %f" %( valid_dsc, valid_adv ) )
            else:
                print()

# --------------------------------------------------------------------------------------------------
class Generator:
    
    def __init__(self,X,y,w,batch_size,syst_shift={}):

        self.X = X
        self.y = y
        self.w = w
        if self.w is None:
            self.w = np.ones(X.shape[0])
        self.batch_size = batch_size
        self.nb = self.X.shape[0] // self.batch_size
        self.last_batch = X.shape[0] % batch_size
        self.syst_shift = syst_shift
        self.y2 = [self.y]

        for feats, shifts in self.syst_shift.items():
            shifts = np.array(shifts).reshape(1, 1, -1)
            probs = np.ones_like(shifts)
            probs /= probs.sum()
            labels = np.random.multinomial(1, probs.ravel(), (self.X.shape[0], 1))
            labels = np.append(labels, labels, axis=1)
            feats = [feat for feat in feats] 
            #old_fits = self.X[feats]
            self.X[feats] += np.sum( (labels*shifts), axis=2)
            #print(np.diff(old_fits-self.X[feats]))
            print(self.y.shape, labels[:,0,:].shape)
            self.y2.append( labels[:,0,:] )

        self.y2 = np.hstack( self.y2 )
        print( X.shape, y.shape, self.y2.shape, w.shape )


    def nbatches(self):
        return self.nb + (self.last_batch!=0)
    
    def __call__(self):
        
        self.X,self.y,self.y2,self.w = shuffle(self.X,self.y,self.y2,self.w)    

        while True:
            for ib in range(self.nb):
                ## ret = []
                yield self.X.iloc[ib*self.batch_size:(ib+1)*self.batch_size],self.y[ib*self.batch_size:(ib+1)*self.batch_size],self.y2[ib*self.batch_size:(ib+1)*self.batch_size],self.w[ib*self.batch_size:(ib+1)*self.batch_size]
                ## ret.extend( [self.X[ib*self.batch_size:(ib+1)*self.batch_size],
                ##              self.y[ib*self.batch_size:(ib+1)*self.batch_size]] )
                ## if self.w is not None:
                ##     ret.append( self.w[ib*self.batch_size:(ib+1)*self.batch_size] )
                ## yield ret
            if self.last_batch > 0:
                yield self.X.iloc[-self.last_batch:],self.y[-self.last_batch:],self.y2[-self.last_batch:],self.w[-self.last_batch:]
                ## ret = []
                ## ret.extend( [self.X[-self.last_batch:],
                ##              self.y[-self.last_batch:]] )
                ## if self.w is not None:
                ##     ret.append( self.w[-self.last_batch:] )
                ## yield ret

            
        
        
        
        
