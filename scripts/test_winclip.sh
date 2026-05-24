device=${CUDA_DEVICE:-0}
data_root=${DATA_ROOT:-./dataset}

test_dataset=${TEST_DATASET:-mvtec}
image_size=${IMAGE_SIZE:-240}
batch_size=${BATCH_SIZE:-8}

for shot in 0 1 2 4
do
    if [ ${shot} -eq 0 ]; then
        seeds="10"
    else
        seeds="10 20 30"
    fi

    for seed in ${seeds}
    do
        save_dir=./results/winclip

        if [ "${test_dataset}" = "visa" ]; then
            test_data_path=${data_root}/Visa
        else
            test_data_path=${data_root}/MVTec
        fi

        CUDA_VISIBLE_DEVICES=${device} python test_winclip.py \
        --dataset ${test_dataset} \
        --test_data_path ${test_data_path} \
        --seed ${seed} \
        --k_shots ${shot} \
        --save_path ${save_dir} \
        --image_size ${image_size} \
        --batch_size ${batch_size}
    wait
    done
done
