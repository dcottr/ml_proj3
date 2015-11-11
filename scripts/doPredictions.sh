#!/bin/bash

# two variables you need to set
pdnndir=$HOME/git/pdnn  # pointer to PDNN
device=cpu  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32

echo "Classifying with the CNN model ..."
python $pdnndir/cmds/run_Extract_Feats.py --data "digit_test_unrotated.pickle.gz" \
                                          --nnet-param cnn_on_5500.param --nnet-cfg cnn_on_5500.cfg \
                                          --output-file "cnn_unrotated.classify.pickle.gz" --layer-index -1 \
                                          --batch-size 100 >& cnn.testing.log

python report_predictions.py cnn_unrotated.classify.pickle.gz
