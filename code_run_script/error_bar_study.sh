# Measure variance of results in Table 1
# SVHN; eps=4/255; PGD (nb_iter=9); Simple and Simple-BN

# mkdir table1_error_bar

# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 4  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_simple_0 --logs table1_error_bar/svhn_simple_0.txt --seed 0
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 4  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_simple_1 --logs table1_error_bar/svhn_simple_1.txt --seed 1
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 4  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_simple_2 --logs table1_error_bar/svhn_simple_2.txt --seed 2
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 4  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_simple_3 --logs table1_error_bar/svhn_simple_3.txt --seed 3
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 4  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_simple_4 --logs table1_error_bar/svhn_simple_4.txt --seed 4
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 4  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_simple_5 --logs table1_error_bar/svhn_simple_5.txt --seed 5
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 4  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_simple_6 --logs table1_error_bar/svhn_simple_6.txt --seed 6
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 4  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_simple_7 --logs table1_error_bar/svhn_simple_7.txt --seed 7
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 4  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_simple_8 --logs table1_error_bar/svhn_simple_8.txt --seed 8
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 4  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_simple_9 --logs table1_error_bar/svhn_simple_9.txt --seed 9

# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 2  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_wrn_0 --logs table1_error_bar/svhn_wrn_0.txt --seed 0
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 2  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_wrn_1 --logs table1_error_bar/svhn_wrn_1.txt --seed 1
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 2  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_wrn_2 --logs table1_error_bar/svhn_wrn_2.txt --seed 2
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 2  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_wrn_3 --logs table1_error_bar/svhn_wrn_3.txt --seed 3
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 2  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_wrn_4 --logs table1_error_bar/svhn_wrn_4.txt --seed 4
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 2  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_wrn_5 --logs table1_error_bar/svhn_wrn_5.txt --seed 5
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 2  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_wrn_6 --logs table1_error_bar/svhn_wrn_6.txt --seed 6
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 2  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_wrn_7 --logs table1_error_bar/svhn_wrn_7.txt --seed 7
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 2  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_wrn_8 --logs table1_error_bar/svhn_wrn_8.txt --seed 8
# python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 2  --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model table1_error_bar/svhn_wrn_9 --logs table1_error_bar/svhn_wrn_9.txt --seed 9

# touch simple_eps_iter_table1_error_bar.txt

# printf "\nSimple Table 1 Error Bar\n" >> simple_eps_iter_table1_error_bar.txt
# printf "PGD Baseline \n" >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_0_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_1_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_2_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_3_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_4_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_5_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_6_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_7_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_8_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_9_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> simple_eps_iter_table1_error_bar.txt

# printf "\nPGD Zero + Nondiff \n" >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_0_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_1_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_2_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_3_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_4_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_5_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_6_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_7_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_8_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> simple_eps_iter_table1_error_bar.txt
# python cifar_evaluate_all.py --load_model table1_error_bar/simple_9_full_train.pt --model_type simple --no_batch_norm --scale_factor 4  --attack_type pgd --eps 0.0157 --eps_iter 0.0118 --nb_iter 20 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> simple_eps_iter_table1_error_bar.txt

## CIFAR-10: Simple, Simple-BN, WRN Evaluate for FGSM, R-FGSM, PGD (nb_iter=9, eps_iter=2/255) for eps=4/255 in Linf
MODEL_PATH="table1_error_bar/"
MODEL_INFO="--model_type simple --scale_factor 4 --no_batch_norm"
ATTACK_MODES=("--attack_type fgsm --eps 0.0157" "--attack_type rfgsm --eps 0.0157" "--attack_type pgd --eps 0.0157 --eps_iter 0.0078 --nb_iter 9")
COMP_MODES=("--bpda_compensation --relu_sub relu --maxpool_sub_p 0" "--bpda_compensation --relu_sub relu --maxpool_sub_p 0 --zero_compensation targeted --targeted_to second" "--bpda_compensation --relu_sub softplus --maxpool_sub_p 5" "--bpda_compensation --relu_sub softplus --maxpool_sub_p 5 --zero_compensation targeted --targeted_to second")
LOGFILE="cifar10_evals.txt"
N=10

touch $LOGFILE
# printf "Simple 4\n\n" >> $LOGFILE

# for ATTACK in "${ATTACK_MODES[@]}"
# do
#     for COMP in "${COMP_MODES[@]}"
#     do
#         printf "Attack and compensation modes: ${ATTACK}, ${COMP}\n\n" >> $LOGFILE
#         for(( i=0; i<$N; i++ ))
#         do
#             MODEL="${MODEL_PATH}simple_${i}_full_train.pt"
#             PARAMS="--load_model ${MODEL} ${MODEL_INFO} ${ATTACK} ${COMP}"
#             python cifar_evaluate_all.py ${PARAMS} >> $LOGFILE
#         done
#     done
# done

printf "\n\n Simple-BN 4 \n\n" >> $LOGFILE
MODEL_INFO="--model_type simple --scale_factor 4"
for ATTACK in "${ATTACK_MODES[@]}"
do
    for COMP in "${COMP_MODES[@]}"
    do
        printf "Attack and compensation modes: ${ATTACK}, ${COMP}\n\n" >> $LOGFILE
        for(( i=0; i<$N; i++ ))
        do
            MODEL="${MODEL_PATH}simple_bn${i}_full_train.pt"
            PARAMS="--load_model ${MODEL} ${MODEL_INFO} ${ATTACK} ${COMP}"
            python cifar_evaluate_all.py $PARAMS >> $LOGFILE
        done
    done
done

printf "\n\n WRN2 \n\n" >> $LOGFILE
MODEL_INFO="--model_type wrn --scale_factor 2"
for ATTACK in "${ATTACK_MODES[@]}"
do
    for COMP in "${COMP_MODES[@]}"
    do
        printf "Attack and compensation modes: ${ATTACK}, ${COMP}\n\n" >> $LOGFILE
        for(( i=0; i<$N; i++ ))
        do
            MODEL="${MODEL_PATH}wrn_${i}_full_train.pt"
            PARAMS="--load_model ${MODEL} ${MODEL_INFO} ${ATTACK} ${COMP}"
            python cifar_evaluate_all.py $PARAMS >> $LOGFILE
        done
    done
done

## SVHN: Simple-BN, WRN Evaluate for FGSM, R-FGSM, PGD same as above
# touch svhn_evals.txt