#!/usr/bin/env python

import numpy as np

import utils.ffwd as ffwd 
import utils.pivot as pivot 
import utils.io as io

import os
import json

from pprint import pprint

from sklearn.model_selection import train_test_split,KFold

from optparse import OptionParser, make_option

##
dipho_features = ['leadptom',
                  'subleadptom',
                  'leadmva',
                  'subleadmva',
                  'leadeta',
                  'subleadeta',
                  'sigmarv',
                  'sigmawv',
                  'CosPhi',
                  'vtxprob']

## command_line options
parser = OptionParser(option_list=[
    make_option("--inp-dir",type='string',dest="inp_dir",default=os.environ['SCRATCH']+'/diphotonID/samples/',help='input directory'),
    make_option("--out-dir",type='string',dest="out_dir",default=os.environ['SCRATCH']+'/diphotonID/AN_output/'),
    make_option("--inp-file",type='string',dest='inp_file',default='diphoton_id_s+b_train.hd5'),
    make_option("--features",type='string',dest='features',default=''),
    make_option("--normalize-target",action="store_true",dest="normalize_target",default=True),
    make_option("--no-normalize-target",action="store_false",dest="normalize_target"),
    make_option("--loss",type='string',dest="loss",default="binary_crossentropy"),
    make_option("--valid-frac",type='float',dest='valid_frac',default=0.10),
    make_option("--test-frac",type='float',dest='test_frac',default=0.10),
    make_option("--batch-size",type='int',dest='batch_size',default=1024),
    make_option("--epochs",type='int',dest='epochs',default=20),
    make_option("--activations",type='string',dest='activations',default='relu'),
    make_option("--hparams",type='string',dest='hparams',default=None),
    make_option("--seed",type='int',dest='seed',default=98543),
    make_option("--x-val",action="store_true",dest='x_val',default=False),
    make_option("--nkfolds",type='int',dest='nkfolds',default=5)
])

## parse options
(options, args) = parser.parse_args()

options.out_dir = options.out_dir
os.makedirs(options.out_dir, exist_ok=True)

if options.features == '':
    features = dipho_features
    options.features = ','.join(features)
else:
    features = options.features.split(',')

if os.path.exists(options.inp_file):
    inp_file = options.inp_file
else:
    inp_file = options.inp_dir+'/'+options.inp_file

hparams = {}
if options.hparams is not None:
    for fname in  options.hparams.split(','):
       with open(fname) as hf:
          pars = json.loads(hf.read())
          hparams.update(pars)    # if inside several files we change the same parameter, it will overwrite for the one in the last file    
    
## read data
data = io.read_data(inp_file)

data['isSignal'] = data['processIndex'] < 5
data['isBkg'] = data['processIndex'] >= 5
X = data[features]
y = data['isSignal'].values.reshape(-1,1)
w = np.abs(data['weight'].values.ravel())

# sort out model parameters
def get_kwargs(fn,**kwargs):
    params = set(fn.__code__.co_varnames[:fn.__code__.co_argcount]+tuple(kwargs.keys()))
    for par in params:
        if hasattr(options,par):
            kwargs[par] = getattr(options,par)
        if par in hparams:
            kwargs[par] = hparams.pop(par)
    return kwargs

loss_params=dict()
init_kwargs = get_kwargs(pivot.PivotClassifier.__init__, monitor_dir=options.out_dir, loss_params=loss_params)
fit_kwargs = get_kwargs(pivot.PivotClassifier.fit, batch_size=options.batch_size, epochs=options.epochs, syst_shift={('leadmva','subleadmva'):[0, -0.01, 0.01]})

print(init_kwargs)
print(fit_kwargs)

clf = ffwd.FFWDRegression('clf', X.shape[1:], y.shape[1:], loss='binary_crossentropy')
dsc = ffwd.FFWDRegression('dsc', y.shape[1:], (1, 3), loss='categorical_crossentropy')
net = pivot.PivotClassifier('pivot', clf, dsc)

pprint(net.get_params())

adv_model, dsc_model = net()

print(adv_model.summary())
print(dsc_model.summary())

adv_model_summary = str(adv_model.to_json()) 
dsc_model_summary = str(dsc_model.to_json()) 

if not os.path.exists(options.out_dir):
    os.mkdir(options.out_dir)

with open(options.out_dir+'/adv_model_summary.txt','w+') as fo:
    fo.write(adv_model_summary)
with open(options.out_dir+'/dsc_model_summary.txt','w+') as fo:
    fo.write(dsc_model_summary)

# store = dict( model_params = net.get_params(),
#               fit_kwargs = fit_kwargs,
#               options = options.__dict__
#               )
# with open(options.out_dir+'/config.json','w+') as fo:
#     fo.write( json.dumps( store, indent=4 ) )
#     fo.close()

if options.x_val==True:
    kf = KFold(n_splits=int(1./options.valid_frac),shuffle=True,random_state=options.seed) 
    kf_idx = iter(kf.split(X)) 
    for kfold in range(options.nkfolds):
        # split data
        train_idx,valid_idx = next(kf_idx)
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        net.fit(X_train,y_train, 
                validation_data=(X_valid,y_valid),
                **fit_kwargs)
else :
    X_train,X_valid,y_train,y_valid,w_train,w_valid = train_test_split(X,y,w,test_size=options.valid_frac,random_state=options.seed)
    
    print(w_train.shape, y_train.shape)

    ### here if you want to save hdf file after each better epoch, put kfold=-1
    net.fit(X_train,y_train,w_train,
            validation_data=(X_valid,y_valid,w_valid),
            **fit_kwargs)

