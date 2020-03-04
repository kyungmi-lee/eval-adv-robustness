mkdir cifar_pruning_check

# Train a WRN 28-10 on CIFAR-10 with weight decay of 5e-4
python cifar_training.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 10 --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 5e-4 --save_model cifar_pruning_check/wrn10_wd5e-4 --logs cifar_pruning_check/wrn10_wd5e-4.txt

python cifar_pruning.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 10 --lr 1e-3 --weight_decay 5e-4 --prune --epochs 9 --retrain_epochs 10 --prune_fixed_percent --prune_ratio_per_step 0.75 --load_model cifar_pruning_check/wrn10_wd5e-4_early_stop.pt --save_model cifar_pruning_check/wrn10_wd5e-4 --prune_txt cifar_pruning_check/wrn10_wd5e-4_prune.txt

# Then evaluate the first dense and the resulting sparse models
# w/ L2 PGD (eps=0.3, eps_iter=0.1, nb_iter=9) only w/ zero+nondiff compensation
echo "Dense: L2 PGD Evaluation with total 5 random starts, ZERO + NONDIFF COMPENSATION (eps=0.3)"
python cifar_evaluate_all.py --load_model cifar_pruning_check/wrn10_wd5e-4_early_stop.pt --model_type wrn --scale_factor 10 --attack_type l2pgd --eps 0.3 --eps_iter 0.1 --nb_iter 9 --bpda_compensation --relu_sub softplus --zero_compensation targeted --targeted_to second

echo "Sparse: L2 PGD Evaluation with total 5 random starts,  ZERO + NONDIFF COMPENSATION (eps=0.3)"
python cifar_evaluate_all.py --load_model cifar_pruning_check/wrn10_wd5e-4_9.pt --model_type wrn --scale_factor 10 --attack_type l2pgd --eps 0.3 --eps_iter 0.1 --nb_iter 9 --bpda_compensation --relu_sub softplus --zero_compensation targeted --targeted_to second

# Then, evaluate using PGD + Eigen
echo "Dense: L2 PGD Evaluation with total 5 random starts, ZERO + NONDIFF + SECOND COMPENSATION (eps=0.3)"
python cifar_evaluate_all.py --load_model cifar_pruning_check/wrn10_wd5e-4_early_stop.pt --model_type wrn --scale_factor 10 --attack_type l2pgd --eps 0.3 --eps_iter 0.1 --nb_iter 7 --bpda_compensation --relu_sub softplus --zero_compensation targeted --targeted_to second --second_order_init_method miyato

echo "Sparse: L2 PGD Evaluation with total 5 random starts,  ZERO + NONDIFF + SECOND COMPENSATION (eps=0.3)"
python cifar_evaluate_all.py --load_model cifar_pruning_check/wrn10_wd5e-4_9.pt --model_type wrn --scale_factor 10 --attack_type l2pgd --eps 0.3 --eps_iter 0.1 --nb_iter 7 --bpda_compensation --relu_sub softplus --zero_compensation targeted --targeted_to second --second_order_init_method miyato