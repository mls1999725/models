# set -aux

DEVICE_NUM_PER_NODE=1
MASTER_ADDR=127.0.0.1
NUM_NODES=1
NODE_RANK=0

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED
export NCCL_LAUNCH_MODE=PARALLEL
echo NCCL_LAUNCH_MODE=$NCCL_LAUNCH_MODE
export NCCL_DEBUG=INFO

OFRECORD_PATH=/dataset/ImageNet/ofrecord/

OFRECORD_PART_NUM=256
LEARNING_RATE=0.768
MOM=0.875
EPOCH=50
TRAIN_BATCH_SIZE=96
VAL_BATCH_SIZE=50

# SRC_DIR=/path/to/models/resnet50
SRC_DIR=$(realpath $(dirname $0)/..)

python3 $SRC_DIR/train.py \
    --ofrecord-path $OFRECORD_PATH \
    --ofrecord-part-num $OFRECORD_PART_NUM \
    --num-devices-per-node $DEVICE_NUM_PER_NODE \
    --lr $LEARNING_RATE \
    --momentum $MOM \
    --num-epochs $EPOCH \
    --train-batch-size $TRAIN_BATCH_SIZE \
    --val-batch-size $VAL_BATCH_SIZE \
