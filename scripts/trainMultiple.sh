#!/bin/bash

# two variables you need to set
pdnndir=$HOME/git/pdnn  # pointer to PDNN
device=gpu0  # the device to be used. set it to "cpu" if you don't have GPUs (gpu0 normally)

# export environment variables
export PYTHONPATH=$PYTHONPATH:$pdnndir
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32

for i in {1..30};

do

echo "on loop"
echo $i

# train CNN model
echo "Training the CNN model 1..."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_1_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_1of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

echo "Training the CNN model 2..."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_2_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_2of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

echo "Training the CNN model 3..."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_3_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_1of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

echo "Training the CNN model 4..."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_4_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_2of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

echo "Training the CNN model 5..."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_5_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_1of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

echo "Training the CNN model 6..."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_6_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_2of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

echo "Training the CNN model 7..."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_7_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_1of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

echo "Training the CNN model 8..."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_8_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_2of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

echo "Training the CNN model 9.."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_9_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_1of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

echo "Training the CNN model 10.."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_10_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_2of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

echo "Training the CNN model 11..."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_11_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_1of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

echo "Training the CNN model 12..."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_12_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_2of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

echo "Training the CNN model 13..."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_13_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_1of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

echo "Training the CNN model 14..."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_14_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_2of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

echo "Training the CNN model 15..."
python $pdnndir/cmds/run_CNN.py --train-data "digit_train_rotated_split_15_of_15.pickle.gz" \
                                --valid-data "digit_validation_unrotated_split_1of2.pickle.gz" \
                                --conv-nnet-spec "1x48x48:20,10x10,p2x2:50,10x10,p2x2,f" --nnet-spec "512:10" --wdir ./ \
                                --l2-reg 0.0001 --lrate "C:0.1:2" --model-save-step 1 \
                                --param-output-file cnn.param --cfg-output-file cnn.cfg  >>cnn.training.log 2>&1

done
