mkdir cifar_pruning_check

# As a test for code functionality,
# prune a WRN 28 trained w/ weight decay for 2 pruning iterations, and keep only 50% of weights each iteration
# fientuning for 40 epochs w/ weight decay for each iteration
# to inflate the influence of weight decay as we are only using 2 pruning iterations,
# set weight decay to 5e-3
python cifar_pruning.py --disable_adv --no_adv_eval --model_type wrn --scale_factor 2 --lr 1e-3 --weight_decay 5e-3 --prune --epochs 2 --retrain_epochs 40 --prune_fixed_percent --prune_ratio_per_step 0.5 --load_model cifar_train_architecture_check_models/cifar_wrn_scale2_wd5e-4_full_train.pt --save_model cifar_pruning_check/wrn2_wd5e-3 --prune_txt cifar_pruning_check/wrn2_wd5e-3.txt

# Then evaluate the first dense and the resulting sparse models
# w/ L2 PGD (eps=0.3, eps_iter=0.1, nb_iter=9) only w/ zero+nondiff compensation
echo "Dense: L2 PGD Evaluation with total 5 random starts, ZERO + NONDIFF COMPENSATION (eps=0.3)"
python cifar_evaluate_all.py --load_model cifar_train_architecture_check_models/cifar_wrn_scale2_wd5e-4_full_train.pt --model_type wrn --scale_factor 2 --attack_type l2pgd --eps 0.3 --eps_iter 0.1 --nb_iter 9 --bpda_compensation --relu_sub softplus --zero_compensation targeted --targeted_to second

echo "Sparse: L2 PGD Evaluation with total 5 random starts,  ZERO + NONDIFF COMPENSATION (eps=0.3)"
python cifar_evaluate_all.py --load_model cifar_pruning_check/wrn2_wd5e-3_2.pt --model_type wrn --scale_factor 2 --attack_type l2pgd --eps 0.3 --eps_iter 0.1 --nb_iter 9 --bpda_compensation --relu_sub softplus --zero_compensation targeted --targeted_to second

# Then, evaluate using PGD + Eigen
echo "Dense: L2 PGD Evaluation with total 5 random starts, ZERO + NONDIFF + SECOND COMPENSATION (eps=0.3)"
python cifar_evaluate_all.py --load_model cifar_train_architecture_check_models/cifar_wrn_scale2_wd5e-4_full_train.pt --model_type wrn --scale_factor 2 --attack_type l2pgd --eps 0.3 --eps_iter 0.1 --nb_iter 7 --bpda_compensation --relu_sub softplus --zero_compensation targeted --targeted_to second --second_order_init_method miyato

echo "Sparse: L2 PGD Evaluation with total 5 random starts,  ZERO + NONDIFF + SECOND COMPENSATION (eps=0.3)"
python cifar_evaluate_all.py --load_model cifar_pruning_check/wrn2_wd5e-3_2.pt --model_type wrn --scale_factor 2 --attack_type l2pgd --eps 0.3 --eps_iter 0.1 --nb_iter 7 --bpda_compensation --relu_sub softplus --zero_compensation targeted --targeted_to second --second_order_init_method miyato