# How to use compensation methods

# 1) Plain FGSM Evaluation (No multiple random_start required; no stochasticity in the attack)
# Model: Simple 4 (No BN) saved at cifar_train_architecture_check_models/cifar_simple_scale4_noreg_full_train.pt
# Eps for attack: 4/255 (0.0157)
echo "Plain FGSM Evaluation (eps=4/255)"
python cifar_evaluate_all.py --load_model cifar_train_architecture_check_models/cifar_simple_scale4_noreg_full_train.pt --model_type simple --scale_factor 4 --no_batch_norm --attack_type fgsm --eps 0.0157

# 2) Baseline R-FGSM Evaluation with total 4 random starts (same total number of evaluation as compensation methods)
#    To set the same total number of evaluation
#    turn bpda_compensation on but set relu_sub to be relu and maxpool_sub_p to be 0 (maxpooling in this case)
echo "Plain R-FGSM Evaluation with total 4 random starts (eps=4/255)"
python cifar_evaluate_all.py --load_model cifar_train_architecture_check_models/cifar_simple_scale4_noreg_full_train.pt --model_type simple --scale_factor 4 --no_batch_norm --attack_type rfgsm --eps 0.0157 --bpda_compensation --relu_sub relu --maxpool_sub_p 0

# 3) and similarly for PGD (eps=4/255, eps_iter=2/255, nb_iter=9), with total 5 random starts
echo "Plain PGD Evaluation with total 5 random starts (eps=4/255)"
python cifar_evaluate_all.py --load_model cifar_train_architecture_check_models/cifar_simple_scale4_noreg_full_train.pt --model_type simple --scale_factor 4 --no_batch_norm --attack_type pgd --eps 0.0157 --eps_iter 0.0078 --nb_iter 9 --bpda_compensation --relu_sub relu --maxpool_sub_p 0

# 4) Let's apply zero compensation by targeting to the second most likely class on FGSM
echo "FGSM Evaluation with ZERO COMPENSATION (eps=4/255)"
python cifar_evaluate_all.py --load_model cifar_train_architecture_check_models/cifar_simple_scale4_noreg_full_train.pt --model_type simple --scale_factor 4 --no_batch_norm --attack_type fgsm --eps 0.0157 --zero_compensation targeted --targeted_to second

# 5) Let's apply zero + non-diff compensation
echo "FGSM Evaluation with ZERO + NONDIFF COMPENSATION (eps=4/255)"
python cifar_evaluate_all.py --load_model cifar_train_architecture_check_models/cifar_simple_scale4_noreg_full_train.pt --model_type simple --scale_factor 4 --no_batch_norm --attack_type fgsm --eps 0.0157 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5

# 6) Let's compare the second-order initialization method (Eigen & BFGS)
#    First, measure plain PGD accuracy of WRN 28 trained with excessive weight decay
#    model saved at: cifar_train_architecture_check_models/cifar_wrn_scale2_wdexcess_full_train.pt
#    Attack: L2 PGD, eps=0.5, eps_iter=0.2, nb_iter=7
echo "Plain L2 PGD Evaluation with total 5 random starts (eps=0.5)"
python cifar_evaluate_all.py --load_model cifar_train_architecture_check_models/cifar_wrn_scale2_wdexcess_full_train.pt --model_type wrn --scale_factor 2 --attack_type l2pgd --eps 0.5 --eps_iter 0.2 --nb_iter 7 --bpda_compensation --relu_sub relu --maxpool_sub_p 0

# 7) Applying zero + non-diff compensation results in...
echo "L2 PGD Evaluation with ZERO + NONDIFF COMPENSATION (eps=0.5)"
python cifar_evaluate_all.py --load_model cifar_train_architecture_check_models/cifar_wrn_scale2_wdexcess_full_train.pt --model_type wrn --scale_factor 2 --attack_type l2pgd --eps 0.5 --eps_iter 0.2 --nb_iter 7 --bpda_compensation --relu_sub softplus --zero_compensation targeted --targeted_to second

# 8) Now apply second-order init method ('miyato' refers to Eigen in the paper)
#    since this init method uses TWO back-props
#    set number of iteration to be 5, so that the total number of back-props can be the same as (6) & (7)
echo "L2 PGD Evaluation with ZERO + NONDIFF + SECOND ORDER INIT COMPENSATION (eps=0.5)"
python cifar_evaluate_all.py --load_model cifar_train_architecture_check_models/cifar_wrn_scale2_wdexcess_full_train.pt --model_type wrn --scale_factor 2 --attack_type l2pgd --eps 0.5 --eps_iter 0.2 --nb_iter 5 --bpda_compensation --relu_sub softplus --zero_compensation targeted --targeted_to second --second_order_init_method miyato
