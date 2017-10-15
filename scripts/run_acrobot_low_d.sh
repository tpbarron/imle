
LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/";
# LOG_DIR="/home/trevor/Documents/dev/ml/data/outputs/imle/";
DEFAULT_ARGS="--num-processes 1 --num-steps 128 --batch-size 64 --num-frames 100000 --use-gae"

# imle comparison
for i in 6 7 8 9 10 #1 2 3 4 5
do
  mkdir -p "${LOG_DIR}/imle_baseline/acrobot_continuous_x/${i}/"
  python main.py $DEFAULT_ARGS --imle --seed $i --log-dir "${LOG_DIR}/imle_baseline/acrobot_continuous_x/${i}/" --env-name "AcrobotContinuousX-v0" | tee  "${LOG_DIR}/imle_baseline/acrobot_continuous_x/${i}/out.log"
done

# vime baselines
for i in 1 2 3 4 5
do
  mkdir -p "${LOG_DIR}/vime_baseline/acrobot_continuous_x/${i}/"
  python main.py $DEFAULT_ARGS --vime --seed $i --log-dir "${LOG_DIR}/vime_baseline/acrobot_continuous_x/${i}/" --env-name "AcrobotContinuousX-v0" | tee  "${LOG_DIR}/vime_baseline/acrobot_continuous_x/${i}/out.log"
done

# no explr
# low dim
for i in 1 2 3 4 5
do
  mkdir -p "${LOG_DIR}/baseline/acrobot_continuous_x/${i}/"
  python main.py $DEFAULT_ARGS --seed $i --log-dir "${LOG_DIR}/baseline/acrobot_continuous_x/${i}/" --env-name "AcrobotContinuousX-v0" | tee  "${LOG_DIR}/baseline/acrobot_continuous_x/${i}/out.log"
done
