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

from . import losses 

# --------------------------------------------------------------------------------------------------
class ConstOffsetLayer(Layer):

    def __init__(self, values, **kwargs):
        self.values = values
        super(ConstOffsetLayer, self).__init__(**kwargs)

    def get_config(self):
        return { 'values' : self.values }
        
    def call(self, x):
        return x+K.constant(self.values)


# --------------------------------------------------------------------------------------------------
class FFWDRegression(BaseEstimator):

    def __init__(self,name,input_shape,output_shape=None,
                 non_neg=False,
                 dropout=0.2, # 0.5 0.2
                 batch_norm=True,activations="lrelu",
                 layers=[1024]*5+[512,256,128], # 1024 / 2048
#                 layers=[128], # 1024 / 2048
                 do_bn0=True,
                 const_output_biases=None, 
                 optimizer="Adam", optimizer_params=dict(lr=1.e-04), # mse: 1e-3/5e-4
                 loss="RegularizedGaussNll",
                 loss_params=dict(),# dict(reg_sigma=3.e-2),
                 monitor_dir=".",
                 save_best_only=True,
                 valid_frac=None
    ):
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.const_output_biases = const_output_biases

        self.non_neg = non_neg
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.use_bias = not batch_norm
        self.layers = layers
        if type(activations) == str:
            self.activations = [activations]*len(layers)
        elif len(activations) < len(layers):
            rat = len(layers) // len(activations)
            remind = len(layers) % len(activations)
            self.activations = activations*rat
            if remind > 0:
                self.activations += activations[:remind]
        else:
            self.activations = activations
        self.do_bn0 = do_bn0
        
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.loss = loss
        self.loss_params = loss_params

        self.valid_frac = valid_frac
        self.save_best_only = save_best_only
        self.monitor_dir = monitor_dir
        
        self.model = None

        super(FFWDRegression,self).__init__()
        
    # ----------------------------------------------------------------------------------------------
    def __call__(self,docompile=False):
        
        if hasattr(losses,self.loss):
            loss = getattr(losses,self.loss)
            ## print(loss,isinstance(loss,object))
            if isinstance(loss,object):
                loss = loss(**self.loss_params)
            ## print(loss)
        else:
            loss = self.loss

        output_shape = self.output_shape
        if output_shape is None:
            output_shape = (getattr(loss,"n_params",1),)
            
        if self.model is None:
            inputs = Input(shape=self.input_shape,name="%s_inp" % self.name)
            
            if len(self.input_shape)>1:
                L = Flatten(name="%s_flt" % self.name)(inputs)
            else:
                L = inputs
            if self.do_bn0:
                L = BatchNormalization(name="%s_bn0" % self.name)(L)
                
            for ilayer,(isize,iact) in enumerate(zip(self.layers,self.activations)):
                il = ilayer + 1
                L = Dense(isize,use_bias=self.use_bias,name="%s_dense%d" % (self.name,il))(L)
                if self.batch_norm:
                    L = BatchNormalization(name="%s_bn%d" % (self.name,il))(L)
                if self.dropout is not None:
                    L = Dropout(self.dropout, name="%s_do%d" % (self.name,il))(L)
                if iact == "lrelu" or "lrelu_" in iact:
                    nslope = 0.2
                    if "_" in iact:
                        nslope = float(iact.rsplit("_",1))
                    L = LeakyReLU(nslope,name="%s_act%d_%s" % (self.name,il,iact))(L)
                elif iact == "prelu":
                    L = PReLU(name="%s_act%d_%s" % (self.name,il,iact))(L)
                else:
                    L = Activation(iact,name="%s_act%d_%s" % (self.name,il,iact))(L)
            
            reshape = False
            out_size = output_shape[0]
            if len(output_shape) > 1:
                reshape = True
                out_size *= output_shape[1]

            constr = None
            if out_size == 1:
                act = "sigmoid"
            else:
                act = "softmax"
            output = Dense(out_size,use_bias=True,activation=act,
                           name="%s_out" % self.name)(L)
            if reshape:
                output = Reshape(output_shape,name="%s_rshp" % self.name)(output)

            self.model = Model( inputs=inputs, outputs=output )
            
        if docompile:
            optimizer = self.get_optimizer()

            self.model.compile(optimizer=optimizer,loss=loss,metrics=[losses.mse0,losses.mae0,
                                                                      losses.r2_score0])
        return self.model

    # ----------------------------------------------------------------------------------------------
    def get_optimizer(self):
        return getattr(keras.optimizers,self.optimizer)(**self.optimizer_params)

    # ----------------------------------------------------------------------------------------------
    def get_callbacks(self,has_valid=False,monitor='loss',save_best_only=True,kfold=-1):
        if has_valid:
            monitor = 'val_'+monitor
        monitor_dir = self.monitor_dir
        csv = CSVLogger("%s/metrics_kfold%i.csv" % (monitor_dir,kfold))
        #save checkpoint only if it is a proper training, not cross-validation
        if kfold==-1 : 
            checkpoint = ModelCheckpoint("%s/model-{epoch:02d}.hdf5" % monitor_dir,
                                         monitor=monitor,save_best_only=save_best_only,
                                         save_weights_only=False)
            return [csv,checkpoint]
        else : return [csv]
    
    # ----------------------------------------------------------------------------------------------
    # def fit(self,X,y,kfold=-1,**kwargs):

    #     model = self(True)
        
    #     has_valid = kwargs.get('validation_data',None) is not None
    #     if not has_valid and self.valid_frac is not None:
    #         last_train = int( X.shape[0] * (1. - self.valid_frac) )
    #         X_train = X[:last_train]
    #         X_valid = X[last_train:]
    #         y_train = y[:last_train]
    #         y_valid = y[last_train:]
    #         kwargs['validation_data'] = (X_valid,y_valid)
    #         has_valid = True
    #     else:
    #         X_train, y_train = X, y
            
    #     if not 'callbacks' in kwargs:
    #         save_best_only=kwargs.pop('save_best_only',self.save_best_only)
    #         kwargs['callbacks'] = self.get_callbacks(has_valid=has_valid,
    #                                                  save_best_only=save_best_only,kfold=kfold)
            
    #     return model.fit(X_train,y_train,**kwargs)

    # ----------------------------------------------------------------------------------------------
    def fit(self,X,y,w=None,kfold=-1,**kwargs):

        model = self(True)
        
        has_valid = kwargs.get('validation_data',None) is not None
        if not has_valid and self.valid_frac is not None:

            last_train = int( X.shape[0] * (1. - self.valid_frac) )
            X_train = X[:last_train]
            X_valid = X[last_train:]
            y_train = y[:last_train]
            y_valid = y[last_train:]
            if w is not None:
                w_train = w[:last_train]
                w_valid = w[last_train:]
                kwargs['validation_data'] = (X_valid,y_valid,w_valid)
            else:
                w_train = None
                w_valid = None
                kwargs['validation_data'] = (X_valid,y_valid)
            has_valid = True
        else:
            X_train, y_train, w_train = X, y, w
            
        if not 'callbacks' in kwargs:
            save_best_only=kwargs.pop('save_best_only',self.save_best_only)
            kwargs['callbacks'] = self.get_callbacks(has_valid=has_valid,
                                                     save_best_only=save_best_only,kfold=kfold)
            
        return model.fit(X_train,y_train,sample_weight=w_train,**kwargs)


    # ----------------------------------------------------------------------------------------------
    def fit_generator(self,generator,kfold=-1,**kwargs):

        model = self(True)
        
        has_valid = kwargs.get('validation_data',None) is not None
            
        if not 'callbacks' in kwargs:
            save_best_only=kwargs.pop('save_best_only',self.save_best_only)
            kwargs['callbacks'] = self.get_callbacks(has_valid=has_valid,
                                                     save_best_only=save_best_only,kfold=kfold)
            
        return model.fit_generator(generator,**kwargs)
    
    # ----------------------------------------------------------------------------------------------
    def predict(self,X,p0=True,**kwargs):
        y_pred =  self.model.predict(X,**kwargs)
        if p0:
            return y_pred[:,0]
        else:
            return y_pred
    
    # ----------------------------------------------------------------------------------------------
    def score(self,X,y,**kwargs):
        return -self.model.evaluate(X,y,**kwargs)
    
# --------------------------------------------------------------------------------------------------
class Generator:
    
    def __init__(self,dfs,batch_size,features,target):
        self.dfs = dfs
        self.features = features
        self.target = target
        self.batch_size =  batch_size
        self.steps = sum([ (df[self.target].shape[0] // self.batch_size)+
                          (1 if df[self.target].shape[0] % self.batch_size > 0 else  0) for df in dfs] )
        
    def __call__(self):
      batch_size = self.batch_size
      while True:  #infinite loop for generator
        for df in self.dfs:
            X_df = df[self.features]
            y_df = df[self.target]

            nb = X_df.shape[0] // batch_size
            last_batch = X_df.shape[0] % batch_size

            for ib in range(nb):
                yield X_df[ib*batch_size:(ib+1)*batch_size], y_df[ib*batch_size:(ib+1)*batch_size]
            if last_batch > 0:
                yield X_df[-last_batch:], y_df[-last_batch:]

    def compute_mean_std(self):
        y_sum=0
        y_sum2=0
        N=0
        for df in self.dfs:
            y_df = df[self.target]
            y_sum += y_df.sum()
            y_sum2 += (np.square(y_df)).sum()
            N += df[self.target].shape[0]
        mean = y_sum/N	
        mean2 = y_sum2/N 
        std = np.sqrt(mean2-mean**2)
        return mean,std 
    
    def compute_stats(self):
        all_dfs_y = []
        for df in self.dfs:
            y_df = df[self.target]
            all_dfs_y.append(np.array(y_df))
        all_y = np.hstack((all_dfs_y))
        mean = np.median(all_y)
        std = np.std(all_y)
        return mean,std
        
        
