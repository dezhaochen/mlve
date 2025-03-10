(nohup python -u train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --work_dir ./logsave/pixel \
    --save_model \
    --seed 1 > pixel.log 2>&1 ) &