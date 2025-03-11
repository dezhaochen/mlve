(nohup python -u train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --lambdaR 1e-10 1e-6 1e-4 \
    --lambdaD 1e-6 \
    --lambdaE 1e-8 1e-4 \
    --qt 1000 600 1 \
    --KLl 2 22 \
    --save_model \
    --work_dir ./logsave/ \
    --seed 1 > ./trainlog/train.log 2>&1 ) &

