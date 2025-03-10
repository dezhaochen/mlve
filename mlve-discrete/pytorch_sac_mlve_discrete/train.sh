(nohup python -u train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --testpsnr \
    --q \
    --save_model \
    --work_dir ./logsave/ \
    --seed 1 > ./trainlog/total/train.log 2>&1 ) &

