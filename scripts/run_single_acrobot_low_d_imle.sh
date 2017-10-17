
LOG_DIR="test_dir/";
# LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/";
# LOG_DIR="/home/trevor/Documents/dev/ml/data/outputs/imle/";
DEFAULT_ARGS="--num-processes 1 --num-steps 1024 --entropy-coef 0 --ppo-epoch 10 --lr 3e-4 --gamma 0.99 --tau 0.95 --batch-size 64 --num-frames 50000 --use-gae"

# imle comparison
# for i in 1 2 3 4 5
# do
#   mkdir -p "${LOG_DIR}/imle_baseline/acrobot_continuous_x_real_eta_decay/${i}/"
#   python main.py $DEFAULT_ARGS --imle --seed $i --log-dir "${LOG_DIR}/imle_baseline/acrobot_continuous_x_real_eta_decay/${i}/" --env-name "AcrobotContinuousX-v0" --eta 0.001 --eta-decay
# done

# no explr
# low dim
for i in 2
do
  mkdir -p "${LOG_DIR}/baseline/acrobot_continuous_x/${i}/";
  python main.py $DEFAULT_ARGS --seed $i --log-dir "${LOG_DIR}/baseline/acrobot_continuous_x/${i}/" --env-name "AcrobotContinuousX-v0"
done
