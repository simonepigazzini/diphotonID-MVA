#!~/usr/bin/fish

set LAMBDAS '2.' '2.5' '3.' '3.5' '4.'
set GPU 0

for lambda in $LAMBDAS
    set -xU CUDA_VISIBLE_DEVICES $GPU
    ./train_pivot_diphoId.py --out-dir $SCRATCH/diphotonID/AN_output_v7/ --ext-dsc-inputs --dsc-pretrain-weights $SCRATCH/diphotonID/AN_output_v4/pretrain/dsc-model-99.hdf5 --clf-pretrain-weights $SCRATCH/diphotonID/AN_output_v4/pretrain/clf-model-08.hdf5 --epochs=30 --lambda=$lambda &
    set GPU (math "$GPU + 1")
    sleep 30
end
