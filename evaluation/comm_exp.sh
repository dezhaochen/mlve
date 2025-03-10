(nohup python -u comm_multiagent_periodic_fog.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 4 \
    --seed 1 > ./test.log 2>&1 ) &