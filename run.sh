
LOG_DIR="/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/"
# vime baselines
python main.py --vime --seed 1 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/vime_baseline/acrobot_continuous_x/1/' --no-cuda --env-name 'AcrobotContinuousX-v0' > $LOG_DIR'vime_baseline/acrobot_continuous_x/1/out.log' 2>&1
python main.py --vime --seed 2 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/vime_baseline/acrobot_continuous_x/2/' --no-cuda --env-name 'AcrobotContinuousX-v0' > $LOG_DIR'vime_baseline/acrobot_continuous_x/2/out.log' 2>&1
python main.py --vime --seed 3 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/vime_baseline/acrobot_continuous_x/3/' --no-cuda --env-name 'AcrobotContinuousX-v0' > $LOG_DIR'vime_baseline/acrobot_continuous_x/3/out.log' 2>&1
python main.py --vime --seed 4 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/vime_baseline/acrobot_continuous_x/4/' --no-cuda --env-name 'AcrobotContinuousX-v0' > $LOG_DIR'vime_baseline/acrobot_continuous_x/4/out.log' 2>&1
python main.py --vime --seed 5 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/vime_baseline/acrobot_continuous_x/5/' --no-cuda --env-name 'AcrobotContinuousX-v0' > $LOG_DIR'vime_baseline/acrobot_continuous_x/5/out.log' 2>&1

# imle comparison
python main.py --imle --seed 1 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/imle_baseline/acrobot_continuous_x/1/' --no-cuda --env-name 'AcrobotContinuousX-v0' > $LOG_DIR'imle_baseline/acrobot_continuous_x/1/out.log' 2>&1
python main.py --imle --seed 2 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/imle_baseline/acrobot_continuous_x/2/' --no-cuda --env-name 'AcrobotContinuousX-v0' > $LOG_DIR'imle_baseline/acrobot_continuous_x/2/out.log' 2>&1
python main.py --imle --seed 3 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/imle_baseline/acrobot_continuous_x/3/' --no-cuda --env-name 'AcrobotContinuousX-v0' > $LOG_DIR'imle_baseline/acrobot_continuous_x/3/out.log' 2>&1
python main.py --imle --seed 4 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/imle_baseline/acrobot_continuous_x/4/' --no-cuda --env-name 'AcrobotContinuousX-v0' > $LOG_DIR'imle_baseline/acrobot_continuous_x/4/out.log' 2>&1
python main.py --imle --seed 5 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/imle_baseline/acrobot_continuous_x/4/' --no-cuda --env-name 'AcrobotContinuousX-v0' > $LOG_DIR'imle_baseline/acrobot_continuous_x/5/out.log' 2>&1

# imle vision
python main.py --imle --seed 1 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/imle_baseline/acrobot_vision_continuous_x/1/' --no-cuda --env-name 'AcrobotVisionContinuousX-v0' --num-stack 4 > $LOG_DIR'imle_baseline/acrobot_vision_continuous_x/1/out.log' 2>&1
python main.py --imle --seed 2 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/imle_baseline/acrobot_vision_continuous_x/2/' --no-cuda --env-name 'AcrobotVisionContinuousX-v0' --num-stack 4 > $LOG_DIR'imle_baseline/acrobot_vision_continuous_x/2/out.log' 2>&1
python main.py --imle --seed 3 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/imle_baseline/acrobot_vision_continuous_x/3/' --no-cuda --env-name 'AcrobotVisionContinuousX-v0' --num-stack 4 > $LOG_DIR'imle_baseline/acrobot_vision_continuous_x/3/out.log' 2>&1
python main.py --imle --seed 4 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/imle_baseline/acrobot_vision_continuous_x/4/' --no-cuda --env-name 'AcrobotVisionContinuousX-v0' --num-stack 4 > $LOG_DIR'imle_baseline/acrobot_vision_continuous_x/4/out.log' 2>&1
python main.py --imle --seed 5 --log-dir '/media/trevor/22c63957-b0cc-45b6-9d8f-173d9619fb73/outputs/imle/imle_baseline/acrobot_vision_continuous_x/5/' --no-cuda --env-name 'AcrobotVisionContinuousX-v0' --num-stack 4 > $LOG_DIR'imle_baseline/acrobot_vision_continuous_x/5/out.log' 2>&1
