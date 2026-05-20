device=0
#gpu 지정
LOG=${save_dir}"res.log"
echo ${LOG}
#log파일경로

data_root=${DATA_ROOT:-./dataset} #<Set_YOUR_DATASET_DIR> 
vl_reduction=4
n_ctx=12
#learnable context token개수
pq_mid_dim=128
#핵심 파라미터

# train on MVTec dataset
base_dir=${n_ctx}_${vl_reduction}_${pq_mid_dim}_train_on_mvtec_3adapters_batch8
save_dir=./adaptclip_checkpoints-debug/${base_dir}/
CUDA_VISIBLE_DEVICES=${device} python train.py --dataset mvtec --train_data_path ${data_root}/MVTec/ \
--save_path ${save_dir} \
--features_list 6 12 18 24 --image_size 518  --batch_size 8  --print_freq 1 \
--epoch 15 --save_freq 1  \
--n_ctx ${n_ctx}  --vl_reduction ${vl_reduction} --pq_mid_dim ${pq_mid_dim} \
--visual_learner --textual_learner --pq_learner --pq_context   

# train on VisA dataset
base_dir=${n_ctx}_${vl_reduction}_${pq_mid_dim}_train_on_visa_3adapters_batch8
save_dir=./adaptclip_checkpoints/${base_dir}/
CUDA_VISIBLE_DEVICES=${device} python train.py --dataset visa --train_data_path ${data_root}/Visa \
--save_path ${save_dir} \
--features_list 6 12 18 24  --image_size 518  --batch_size 8  --print_freq 1 \
--epoch 15 --save_freq 1  \
--n_ctx ${n_ctx}  --vl_reduction ${vl_reduction} --pq_mid_dim ${pq_mid_dim} \
--visual_learner --textual_learner --pq_learner --pq_context 
