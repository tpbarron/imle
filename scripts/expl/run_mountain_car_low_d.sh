LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/";
# LOG_DIR="/home/users/tbarron/Documents/dev/ml/data/outputs/imle/";
DEFAULT_ARGS="--bnn-n-updates-per-step 500 --max-episode-steps 1000 --num-processes 1 --num-steps 2048 --entropy-coef 0 --ppo-epoch 10 --lr 3e-4 --gamma 0.99 --tau 0.95 --batch-size 64 --num-frames 250000 --use-gae"

#LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/nips_deeprl/";
# LOG_DIR="/home/users/tbarron/Documents/dev/ml/data/outputs/imle/10_18_17/";

# imle comparison
for i in 1 2 3
do
  ENV_NAME="${LOG_DIR}/baseline/mountaincar_continuous_x_t1000/${i}/";
  mkdir -p $ENV_NAME;
  python main.py $DEFAULT_ARGS --seed $i --log-dir $ENV_NAME --env-name "MountainCarContinuousX-v0" &

  ENV_NAME="${LOG_DIR}/imle_baseline/mountaincar_continuous_x_linbnn_t500/${i}/";
  mkdir -p $ENV_NAME;
  python main.py $DEFAULT_ARGS --imle --seed $i --log-dir $ENV_NAME --env-name "MountainCarContinuousX-v0" &

  ENV_NAME="${LOG_DIR}/vime_baseline/mountaincar_continuous_x_t1000/${i}/";
  mkdir -p $ENV_NAME;
  python main.py $DEFAULT_ARGS --vime --bnn-non-lin --seed $i --log-dir $ENV_NAME --env-name "MountainCarContinuousX-v0" &

  # wait;
done
