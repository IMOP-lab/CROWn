export CUDA_VISIBLE_DEVICES=0

### Initial hyperparameter
ROOT="./datasets/OIMHS" # Relative path to the dataset
OUTPUT="./output/OIMHS"
DATASET="OIMHS"  # dataloader
NETWORKS="CROWn" # You need to select the network you want to train here, please refer to ./model.py for details.
FINOUT=$OUTPUT/CROWn
NUM_CLASS='5'  # You need to replace num_class with the number of segmentation categories for the specified dataset, including background.

########################  train ##########################
torchrun --nproc_per_node=1 --master_port=29500 ./train.py \
    --root $ROOT \
    --output  $FINOUT/train \
    --dataset $DATASET \
    --network $NETWORKS \
    --mode train \
    --pretrain False \
    --batch_size 1 \
    --crop_sample 1 \
    --in_channel 1 \
    --out_classes $NUM_CLASS \
    --lr 0.0001 \
    --optim AdamW \
    --max_iter 320000 \
    --eval_step 2000 \
    --cache_rate 1 \
    --num_workers 4 \
    --world_size 1 \
    --testrank 0 \
    --lrschedule None \
    --distributed \
    # --gpu 0 \
    # --amp   # if you don't want to use mixed precision training, comment it out

########################  test ##########################
torchrun --nproc_per_node=1 --master_port=29500 ./test.py \
    --root $ROOT \
    --output $FINOUT/test \
    --dataset $DATASET \
    --network  $NETWORKS\
    --trained_weights $FINOUT/train/best_metric_model.pth \
    --mode test \
    --in_channel 1 \
    --out_classes $NUM_CLASS \
    --sw_batch_size 2 \
    --overlap 0.7 \
    --cache_rate 0 \
    --world_size 1 \
    --testrank 0 \
    --distributed \
    # --gpu 0 \

## You can adjust the metrics calculated by modifying 'metrics_list', which detailed in./utils/niigz2excel.py
python ./utils/niigz2excel.py \
    --root $ROOT \
    --output $FINOUT/test \
    --network $NETWORKS \
    --metrics_list iou dice assd hd hd95 adj_rand \
    --out_classes $NUM_CLASS

###  Paste the following command into the command line to quickly start the training and testing process, 
###  you need to replace 3dseg with your virtual environment.

: <<'END'

kill -9 $(lsof -t -i:29500)
lsof -i :29500

source activate 3dseg
bash run_gpu.sh

END