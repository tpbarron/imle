# LOG_DIR="test_dir/";
LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/";
# LOG_DIR="/home/users/tbarron/Documents/dev/ml/data/outputs/imle/";
DEFAULT_ARGS="--bnn-n-updates-per-step 5000 --num-processes 1 --num-steps 2048 --entropy-coef 0 --ppo-epoch 10 --lr 3e-4 --gamma 0.99 --tau 0.95 --batch-size 64 --num-frames 1000000 --num-stack 4 --use-gae"

# no explr
# high dim
for i in 2 3 4 5
do
  EXP_PATH="baseline/inverted_pendulum_swingup_vision_dense_rew/${i}/";
  mkdir -p "${LOG_DIR}/${EXP_PATH}";
  python main.py $DEFAULT_ARGS --seed $i --log-dir "${LOG_DIR}/${EXP_PATH}" --env-name "InvertedPendulumSwingupVisionBulletEnv-v0"
done

# imle vision
for i in 1 2 3 4 5
do
  EXP_PATH="imle_baseline/inverted_pendulum_swingup_vision_dense_rew/${i}/";
  mkdir -p "${LOG_DIR}/${EXP_PATH}";
  python main.py $DEFAULT_ARGS --imle --seed $i --log-dir "${LOG_DIR}/${EXP_PATH}" --env-name "InvertedPendulumSwingupVisionBulletEnv-v0"
done
