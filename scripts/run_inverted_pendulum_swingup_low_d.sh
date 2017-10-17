LOG_DIR="test_dir/"; #"/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/";
# LOG_DIR="/home/users/tbarron/Documents/dev/ml/data/outputs/imle/";
DEFAULT_ARGS="--bnn-n-updates-per-step 500 --num-processes 4 --num-steps 256 --entropy-coef 0 --ppo-epoch 10 --lr 3e-4 --gamma 0.99 --tau 0.95 --batch-size 64 --num-frames 250000 --use-gae"

# imle vision
for i in 1 2 3 4 5
do
  EXP_PATH="imle_baseline/inverted_pendulum_swingup/${i}/";
  mkdir -p "${LOG_DIR}/${EXP_PATH}";
  python main.py $DEFAULT_ARGS --imle --seed $i --log-dir "${LOG_DIR}/${EXP_PATH}" --env-name "InvertedPendulumSwingupBulletX-v0"
done

# no explr
# high dim
# for i in 1
# do
#   EXP_PATH="baseline/inverted_pendulum_swingup/${i}/";
#   mkdir -p "${LOG_DIR}/${EXP_PATH}";
#   python main.py $DEFAULT_ARGS --seed $i --log-dir "${LOG_DIR}/${EXP_PATH}" --env-name "InvertedPendulumSwingupBulletX-v0"
# done
