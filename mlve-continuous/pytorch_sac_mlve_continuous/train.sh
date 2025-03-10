(nohup python -u train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --testpsnr \
    --work_dir ./logsave/sigm-rhvae-ssa/-8_t4 \
    --seed 652 > ./trainlog/sigm-rhvae-ssa/-8_t4.log 2>&1 ) &