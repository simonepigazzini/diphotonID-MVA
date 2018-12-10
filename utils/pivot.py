from .ffwd import FFWDRegression

import numpy as np
import pandas as pd

from keras.layers import Input, Dense, Add, Multiply
from keras.layers import Reshape, UpSampling1D, Flatten, Concatenate, Cropping1D
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

from tqdm import tqdm
from collections import OrderedDict

from . import losses 


# --------------------------------------------------------------------------------------------------
class PivotClassifier(BaseEstimator):
    
    def __init__(self, name, clf, dsc, 
                 dsc_optimizer='Adam', dsc_optimizer_params=dict(lr=2.e-03), 
                 adv_optimizer='Adam', adv_optimizer_params=dict(lr=2.e-04), 
                 ext_dsc_inputs=False, lambd=1.,monitor_dir="./"):
        
        self.lambd = lambd
        self.clf = clf
        self.dsc = dsc
        self.ext_dsc_inputs = ext_dsc_inputs
        
        self.dsc_optimizer = dsc_optimizer
        self.dsc_optimizer_params = dsc_optimizer_params
        self.adv_optimizer = adv_optimizer
        self.adv_optimizer_params = adv_optimizer_params
                
        self.ext_dsc_inputs = ext_dsc_inputs
        self.monitor_dir = monitor_dir


    def __call__(self,docompile=False,save_best_only=True):
        
        clf = self.clf(docompile=False)
        dsc = self.dsc(docompile=False)
        
        if self.ext_dsc_inputs:
            wrapped_inputs = [clf.outputs[0], clf.input]
        else:
            wrapped_inputs = clf.outputs
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
                               loss_weights=[1.,-self.lambd],
                           )
        
        dsc.trainable = True
        clf.trainable = False
        self.dsc_model.compile(optimizer=dsc_optimizer,
                               loss=losses.masked_categorical_crossentropy,
        )
        
        return self.adv_model,self.dsc_model

    # ----------------------------------------------------------------------------------------------
    def get_callbacks(self,monitor='loss',save_best_only=True,label=""):
        monitor_dir = self.monitor_dir
        csv = CSVLogger("%s/%s_metrics.csv" % (monitor_dir,label))
        checkpoint = ModelCheckpoint("%s/%s-model-{epoch:02d}.hdf5" % (monitor_dir,label),
                                     monitor=monitor,save_best_only=save_best_only,
                                     save_weights_only=False)
        return [csv,checkpoint]
        

        
    # ----------------------------------------------------------------------------------------------
    def fit(self,X_train,y_train,w_train=None,
            validation_data=None,
            epochs=1,n_adv_steps=1,n_dsc_steps=100,
            batch_size=4096,
            print_every=10,
            pretrain=False,
            pretrain_clf_kwargs=dict(),
            pretrain_dsc_kwargs=dict(),
            syst_shift={}
    ):
        
        ###---pretrain both CLF and DSC using all the available data:
        ###   - the DCS input is taken from the CLF prediction + CLF inputs unshifted!
        ###   - The input to the DCS are formatted using Generator()
        if pretrain:
            #---CLF
            self.clf(docompile=True)
            self.clf.model.trainable = True
            self.clf.fit(X_train,y_train,w_train,validation_data=validation_data,**pretrain_clf_kwargs) 
            #---DSC
            self.clf.model.trainable = False
            self.dsc(docompile=True)
            self.dsc.model.trainable = True                        
            pgen = Generator(X_train,y_train,w_train,len(X_train.index),syst_shift=syst_shift)
            vgen = Generator(*validation_data,len(validation_data[0].index),syst_shift=syst_shift)
            
            pdata = []
            for igen in pgen,vgen:
                Xorig,X,y,w = next(igen())
                which=(y[0] == 1).ravel()
                Xorig = Xorig[which]
                X = X[which]
                y = [ yy[which] for yy in y ]
                w = [ ww[which] for ww in w ]
                y_pred = self.clf.predict(X).reshape(-1, 1)
                pdata.append( (Xorig,X,y,w,y_pred) )
            (Xorig,X,y,w,y_pred), (Xvorig,Xv,yv,wv,yv_pred) = pdata
            pretrain_dsc_kwargs["validation_data"] = ([yv_pred, Xvorig], yv[-1][:,np.newaxis,1:], wv[-1])
            self.dsc.fit(y_pred, y[-1][:,np.newaxis,1:], w[-1], extra_inputs=Xorig.values, **pretrain_dsc_kwargs)
            return
            
        adv_model,dsc_model = self(docompile=True)
        
        gen = Generator(X_train,y_train,w_train,batch_size,syst_shift=syst_shift)
        nbatches = gen.nbatches() 
        nsteps = nbatches // (n_dsc_steps+n_adv_steps)
        
        if validation_data is not None:
            vgen = Generator(*validation_data,batch_size,syst_shift=syst_shift)

        ## callbacks=self.get_callbacks(save_best_only=save_best_only,label="adv")
        ## callbacks=self.get_callbacks(save_best_only=save_best_only,label="dsc")
        
        store = dict(dsc_loss=[],
                     adv_loss=[],adv_clf_loss=[],adv_dsc_loss=[],
                     dsc_loss_valid=[],
                     adv_loss_valid=[],adv_clf_loss_valid=[],adv_dsc_loss_valid=[])
        
        eprog =  tqdm(range(epochs),"")
        for ei in eprog:
            ep_gen = gen()
            dsc_loss = []
            adv_loss = []
            sprog = tqdm(range(nsteps),"epoch %d" % ei,leave=True)
            for si in sprog:
                for di in range(n_dsc_steps):
                    Xb_orig,Xb,yb,wb = next(ep_gen)
                    dsc_loss.append( dsc_model.train_on_batch(Xb_orig,yb[-1],wb[-1]) )
                for ai in range(n_adv_steps):
                    Xb_orig,Xb,yb,wb = next(ep_gen)
                    adv_loss.append( adv_model.train_on_batch(Xb,yb,wb) )
                    if si % print_every == 1:
                        sprog.set_postfix( OrderedDict( [ ("dsc_loss",np.array(dsc_loss).mean()), 
                                                          ("adv_loss, adv_clf_loss, adv_dsc_loss",np.array(adv_loss).mean(axis=0)) ] ) )
                
            dsc_loss = np.array(dsc_loss).mean()
            adv_loss = np.array(adv_loss).mean(axis=0)
            epostfix = [ ("dsc_loss",dsc_loss), ("adv_loss",adv_loss) ]
            store["dsc_loss"].append(dsc_loss)
            store["adv_loss"].append(adv_loss[0])
            store["adv_clf_loss"].append(adv_loss[1])
            store["adv_dsc_loss"].append(adv_loss[2])
            if validation_data is not None:
                dsc_loss_valid = dsc_model.evaluate_generator(vgen(False),steps=vgen.nbatches())
                adv_loss_valid = adv_model.evaluate_generator(vgen(True),steps=vgen.nbatches())
                epostfix += [ ("dsc_loss_valid",dsc_loss_valid), ("adv_loss_valid",np.array(adv_loss_valid)) ]
                store["dsc_loss_valid"].append(dsc_loss_valid)
                store["adv_loss_valid"].append(adv_loss_valid[0])
                store["adv_clf_loss_valid"].append(adv_loss_valid[1])
                store["adv_dsc_loss_valid"].append(adv_loss_valid[2])
            eprog.set_postfix( epostfix )
            pd.DataFrame(store).to_csv(self.monitor_dir+"/pivot_metrics.csv")
            ## FIXME save models
            

# --------------------------------------------------------------------------------------------------
class Generator:
    
    def __init__(self,X,y,w,batch_size,syst_shift={}):

        self.origX = X
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
            self.X[feats] += np.sum( (labels*shifts), axis=2)
            print(self.y.shape, labels[:,0,:].shape)
            self.y2.append( labels[:,0,:] )
            
        self.y2 = np.hstack( self.y2 )
        print( X.shape, y.shape, self.y2.shape, w.shape )


    def nbatches(self):
        return self.nb + (self.last_batch!=0)
    
    def __call__(self,returny1=True):
        
        self.origX,self.X,self.y,self.y2,self.w = shuffle(self.origX,self.X,self.y,self.y2,self.w)    

        def mk_ret(p0,p1):
            ret = [self.origX.iloc[p0:p1], self.X.iloc[p0:p1]]
            if returny1:
                ret.extend( [ [self.y[p0:p1],self.y2[p0:p1]], [self.w[p0:p1],self.w[p0:p1]] ] )
            else:
                ret.extend( [ self.y2[p0:p1],self.w[p0:p1] ] )
            return ret 
            
        while True:
            for ib in range(self.nb):
                yield mk_ret(ib*self.batch_size,(ib+1)*self.batch_size)
            if self.last_batch > 0:
                yield mk_ret(-self.last_batch,None)
                
            
        
        
        
        
