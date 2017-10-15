LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/";
# LOG_DIR="/home/users/tbarron/Documents/dev/ml/data/outputs/imle/";
DEFAULT_ARGS="--num-processes 1 --num-steps 128 --batch-size 64 --max-num-updates 100 --num-stack 4"

# imle vision
for i in 1 2 3 4 5
do
  mkdir -p "${LOG_DIR}/imle_baseline/acrobot_continuous_vision_x/${i}/"
  python main.py $DEFAULT_ARGS --imle --seed $i --log-dir "${LOG_DIR}/imle_baseline/acrobot_continuous_vision_x/${i}/" --env-name "AcrobotContinuousVisionX-v0" | tee  "${LOG_DIR}/imle_baseline/acrobot_continuous_vision_x/${i}/out.log"
done

# no explr
# high dim
for i in 1 2 3 4 5
do
  mkdir -p "${LOG_DIR}/baseline/acrobot_continuous_vision_x/${i}/"
  python main.py $DEFAULT_ARGS --seed $i --log-dir "${LOG_DIR}/baseline/acrobot_continuous_vision_x/${i}/" --env-name "AcrobotContinuousVisionX-v0" | tee  "${LOG_DIR}/baseline/acrobot_continuous_vision_x/${i}/out.log"
done
