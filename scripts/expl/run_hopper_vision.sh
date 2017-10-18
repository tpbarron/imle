LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/";
# LOG_DIR="/home/users/tbarron/Documents/dev/ml/data/outputs/imle/";
DEFAULT_ARGS="--bnn-n-updates-per-step 500 --num-processes 4 --num-steps 256 --num-stack 4 --batch-size 64 --ppo-epoch 10 --lr 3e-4 --entropy-coef 0 --gamma 0.99 --tau 0.95 --use-gae --num-frames 1000000"

# imle
for i in 1 2 3 4 5
do
  EXP_PATH="${LOG_DIR}/imle_baseline/hopper_vision_x/${i}/";
  mkdir -p $EXP_PATH
  python main.py $DEFAULT_ARGS --imle --seed $i --log-dir $EXP_PATH --env-name "HopperVisionBulletX-v0"
  # | tee  "${EXP_PATH}/out.log"
done

# no explr
# low dim
for i in 1 2 3 4 5
do
  EXP_PATH="${LOG_DIR}/baseline/hopper_vision_x/${i}/";
  mkdir -p $EXP_PATH
  python main.py $DEFAULT_ARGS --seed $i --log-dir $EXP_PATH --env-name "HopperVisionBulletX-v0"
  #  | tee  "${EXP_PATH}/out.log"
done
