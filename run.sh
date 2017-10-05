
LOG_DIR="/home/users/tbarron/Documents/dev/ml/data/outputs/imle/";
DEFAULT_ARGS="--num-processes 8 --num-steps 20 --batch-size 160"

mkdir -p "${LOG_DIR}vime_baseline/acrobot_continuous_x/1/"
mkdir -p "${LOG_DIR}vime_baseline/acrobot_continuous_x/2/"
mkdir -p "${LOG_DIR}vime_baseline/acrobot_continuous_x/3/"
mkdir -p "${LOG_DIR}vime_baseline/acrobot_continuous_x/4/"
mkdir -p "${LOG_DIR}vime_baseline/acrobot_continuous_x/5/"
mkdir -p "${LOG_DIR}vime_baseline/mountaincar_continuous_x/1/"
mkdir -p "${LOG_DIR}vime_baseline/mountaincar_continuous_x/2/"
mkdir -p "${LOG_DIR}vime_baseline/mountaincar_continuous_x/3/"
mkdir -p "${LOG_DIR}vime_baseline/mountaincar_continuous_x/4/"
mkdir -p "${LOG_DIR}vime_baseline/mountaincar_continuous_x/5/"
mkdir -p "${LOG_DIR}imle_baseline/acrobot_continuous_x/1/"
mkdir -p "${LOG_DIR}imle_baseline/acrobot_continuous_x/2/"
mkdir -p "${LOG_DIR}imle_baseline/acrobot_continuous_x/3/"
mkdir -p "${LOG_DIR}imle_baseline/acrobot_continuous_x/4/"
mkdir -p "${LOG_DIR}imle_baseline/acrobot_continuous_x/5/"
mkdir -p "${LOG_DIR}imle_baseline/mountaincar_continuous_x/1/"
mkdir -p "${LOG_DIR}imle_baseline/mountaincar_continuous_x/2/"
mkdir -p "${LOG_DIR}imle_baseline/mountaincar_continuous_x/3/"
mkdir -p "${LOG_DIR}imle_baseline/mountaincar_continuous_x/4/"
mkdir -p "${LOG_DIR}imle_baseline/mountaincar_continuous_x/5/"
mkdir -p "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/1/"
mkdir -p "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/2/"
mkdir -p "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/3/"
mkdir -p "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/4/"
mkdir -p "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/5/"
mkdir -p "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/1/"
mkdir -p "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/2/"
mkdir -p "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/3/"
mkdir -p "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/4/"
mkdir -p "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/5/"

# vime baselines
python main.py $DEFAULT_ARGS --vime --seed 1 --log-dir "${LOG_DIR}vime_baseline/acrobot_continuous_x/1/" --env-name "AcrobotContinuousX-v0" > "${LOG_DIR}vime_baseline/acrobot_continuous_x/1/out.log" 2>&1
python main.py $DEFAULT_ARGS --vime --seed 2 --log-dir "${LOG_DIR}vime_baseline/acrobot_continuous_x/2/" --env-name "AcrobotContinuousX-v0" > "${LOG_DIR}vime_baseline/acrobot_continuous_x/2/out.log" 2>&1
python main.py $DEFAULT_ARGS --vime --seed 3 --log-dir "${LOG_DIR}vime_baseline/acrobot_continuous_x/3/" --env-name "AcrobotContinuousX-v0" > "${LOG_DIR}vime_baseline/acrobot_continuous_x/3/out.log" 2>&1
python main.py $DEFAULT_ARGS --vime --seed 4 --log-dir "${LOG_DIR}vime_baseline/acrobot_continuous_x/4/" --env-name "AcrobotContinuousX-v0" > "${LOG_DIR}vime_baseline/acrobot_continuous_x/4/out.log" 2>&1
python main.py $DEFAULT_ARGS --vime --seed 5 --log-dir "${LOG_DIR}vime_baseline/acrobot_continuous_x/5/" --env-name "AcrobotContinuousX-v0" > "${LOG_DIR}vime_baseline/acrobot_continuous_x/5/out.log" 2>&1

python main.py $DEFAULT_ARGS --vime --seed 1 --log-dir "${LOG_DIR}vime_baseline/mountaincar_continuous_x/1/" --env-name "MountainCarContinuousX-v0" > "${LOG_DIR}vime_baseline/mountaincar_continuous_x/1/out.log" 2>&1
python main.py $DEFAULT_ARGS --vime --seed 2 --log-dir "${LOG_DIR}vime_baseline/mountaincar_continuous_x/2/" --env-name "MountainCarContinuousX-v0" > "${LOG_DIR}vime_baseline/mountaincar_continuous_x/2/out.log" 2>&1
python main.py $DEFAULT_ARGS --vime --seed 3 --log-dir "${LOG_DIR}vime_baseline/mountaincar_continuous_x/3/" --env-name "MountainCarContinuousX-v0" > "${LOG_DIR}vime_baseline/mountaincar_continuous_x/3/out.log" 2>&1
python main.py $DEFAULT_ARGS --vime --seed 4 --log-dir "${LOG_DIR}vime_baseline/mountaincar_continuous_x/4/" --env-name "MountainCarContinuousX-v0" > "${LOG_DIR}vime_baseline/mountaincar_continuous_x/4/out.log" 2>&1
python main.py $DEFAULT_ARGS --vime --seed 5 --log-dir "${LOG_DIR}vime_baseline/mountaincar_continuous_x/5/" --env-name "MountainCarContinuousX-v0" > "${LOG_DIR}vime_baseline/mountaincar_continuous_x/5/out.log" 2>&1

# imle comparison
python main.py $DEFAULT_ARGS --imle --seed 1 --log-dir "${LOG_DIR}imle_baseline/acrobot_continuous_x/1/" --env-name "AcrobotContinuousX-v0" > "${LOG_DIR}imle_baseline/acrobot_continuous_x/1/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 2 --log-dir "${LOG_DIR}imle_baseline/acrobot_continuous_x/2/" --env-name "AcrobotContinuousX-v0" > "${LOG_DIR}imle_baseline/acrobot_continuous_x/2/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 3 --log-dir "${LOG_DIR}imle_baseline/acrobot_continuous_x/3/" --env-name "AcrobotContinuousX-v0" > "${LOG_DIR}imle_baseline/acrobot_continuous_x/3/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 4 --log-dir "${LOG_DIR}imle_baseline/acrobot_continuous_x/4/" --env-name "AcrobotContinuousX-v0" > "${LOG_DIR}imle_baseline/acrobot_continuous_x/4/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 5 --log-dir "${LOG_DIR}imle_baseline/acrobot_continuous_x/5/" --env-name "AcrobotContinuousX-v0" > "${LOG_DIR}imle_baseline/acrobot_continuous_x/5/out.log" 2>&1

python main.py $DEFAULT_ARGS --imle --seed 1 --log-dir "${LOG_DIR}imle_baseline/mountaincar_continuous_x/1/" --env-name "MountainCarContinuousX-v0" > "${LOG_DIR}imle_baseline/mountaincar_continuous_x/1/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 2 --log-dir "${LOG_DIR}imle_baseline/mountaincar_continuous_x/2/" --env-name "MountainCarContinuousX-v0" > "${LOG_DIR}imle_baseline/mountaincar_continuous_x/2/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 3 --log-dir "${LOG_DIR}imle_baseline/mountaincar_continuous_x/3/" --env-name "MountainCarContinuousX-v0" > "${LOG_DIR}imle_baseline/mountaincar_continuous_x/3/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 4 --log-dir "${LOG_DIR}imle_baseline/mountaincar_continuous_x/4/" --env-name "MountainCarContinuousX-v0" > "${LOG_DIR}imle_baseline/mountaincar_continuous_x/4/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 5 --log-dir "${LOG_DIR}imle_baseline/mountaincar_continuous_x/5/" --env-name "MountainCarContinuousX-v0" > "${LOG_DIR}imle_baseline/mountaincar_continuous_x/5/out.log" 2>&1

# imle vision
python main.py $DEFAULT_ARGS --imle --seed 1 --log-dir "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/1/" --env-name "AcrobotContinuousVisionX-v0" --num-stack 4 > "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/1/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 2 --log-dir "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/2/" --env-name "AcrobotContinuousVisionX-v0" --num-stack 4 > "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/2/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 3 --log-dir "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/3/" --env-name "AcrobotContinuousVisionX-v0" --num-stack 4 > "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/3/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 4 --log-dir "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/4/" --env-name "AcrobotContinuousVisionX-v0" --num-stack 4 > "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/4/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 5 --log-dir "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/5/" --env-name "AcrobotContinuousVisionX-v0" --num-stack 4 > "${LOG_DIR}imle_baseline/acrobot_continuous_vision_x/5/out.log" 2>&1

python main.py $DEFAULT_ARGS --imle --seed 1 --log-dir "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/1/" --env-name "MountainCarContinuousVisionX-v0" --num-stack 4 > "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/1/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 2 --log-dir "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/2/" --env-name "MountainCarContinuousVisionX-v0" --num-stack 4 > "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/2/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 3 --log-dir "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/3/" --env-name "MountainCarContinuousVisionX-v0" --num-stack 4 > "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/3/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 4 --log-dir "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/4/" --env-name "MountainCarContinuousVisionX-v0" --num-stack 4 > "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/4/out.log" 2>&1
python main.py $DEFAULT_ARGS --imle --seed 5 --log-dir "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/5/" --env-name "MountainCarContinuousVisionX-v0" --num-stack 4 > "${LOG_DIR}imle_baseline/mountaincar_continuous_vision_x/5/out.log" 2>&1
