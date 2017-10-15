LOG_DIR="test_dir/"; #"/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/";
# LOG_DIR="/home/users/tbarron/Documents/dev/ml/data/outputs/imle/";
DEFAULT_ARGS="--num-processes 4 --num-steps 256 --entropy-coef 0 --ppo-epoch 4 --lr 1e-3 --gamma 0.99 --tau 0.95 --batch-size 64 --num-frames 1000000 --num-stack 2 --use-gae"

# imle vision
for i in 1
do
  EXP_PATH="imle_baseline/mountain_car_continuous_vision_x/${i}/";
  mkdir -p "${LOG_DIR}/${EXP_PATH}";
  python main.py $DEFAULT_ARGS --imle --seed $i --log-dir "${LOG_DIR}/${EXP_PATH}" --env-name "MountainCarContinuousVisionX-v0" # | tee  "${LOG_DIR}/${EXP_PATH}/out.log"
done

# no explr
# high dim
# for i in 1
# do
#   EXP_PATH="baseline/mountain_car_continuous_vision_x/${i}/";
#   mkdir -p "${LOG_DIR}/${EXP_PATH}";
#   python main.py $DEFAULT_ARGS --seed $i --log-dir "${LOG_DIR}/${EXP_PATH}" --env-name "MountainCarContinuousVisionX-v0" #| tee  "${LOG_DIR}/${EXP_PATH}/out.log"
# done
