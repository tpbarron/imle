# LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/";
LOG_DIR="/home/users/tbarron/Documents/dev/ml/data/outputs/imle/";
DEFAULT_ARGS="--bnn-n-updates-per-step 500 --num-processes 1 --num-steps 2048 --entropy-coef 0 --ppo-epoch 10 --lr 3e-4 --gamma 0.99 --tau 0.95 --batch-size 64 --num-frames 250000 --num-stack 2 --use-gae"

# imle vision
for i in 1 2 3 4 5
do
  EXP_PATH="${LOG_DIR}/baseline/acrobot_continuous_vision_x/${i}/";
  mkdir -p $EXP_PATH;
  python main.py $DEFAULT_ARGS --seed $i --log-dir $EXP_PATH --env-name "AcrobotContinuousVisionX-v0"

  EXP_PATH="${LOG_DIR}/imle_baseline/acrobot_continuous_vision_x/${i}/";
  mkdir -p $EXP_PATH;
  python main.py $DEFAULT_ARGS --imle --seed $i --log-dir $EXP_PATH --env-name "AcrobotContinuousVisionX-v0"

  EXP_PATH="${LOG_DIR}/imle_baseline/acrobot_continuous_vision_x_eta_decay/${i}/";
  mkdir -p $EXP_PATH;
  python main.py $DEFAULT_ARGS --imle --seed $i --log-dir $EXP_PATH --env-name "AcrobotContinuousVisionX-v0" --eta-decay
done

# # no explr
# # high dim
# for i in 1 2 3 4 5
# do
#   EXP_PATH="${LOG_DIR}/baseline/acrobot_continuous_vision_x/${i}/";
#   mkdir -p $EXP_PATH;
#   python main.py $DEFAULT_ARGS --seed $i --log-dir $EXP_PATH --env-name "AcrobotContinuousVisionX-v0"
# done
