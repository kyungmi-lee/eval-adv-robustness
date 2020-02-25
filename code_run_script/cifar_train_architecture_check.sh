mkdir cifar_train_architecture_check_models

# Train different architectures and model widths

# 1) Train a Simple (w/o batch norm) with width factor of 4
#    using SGD w/ momentum=0.9, no explicit regularization
#    Train/val split (90%/10%)
#    Save model and log file in cifar_train_functionality_check_models
python cifar_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 4 --no_batch_norm --lr 1e-2 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model cifar_train_architecture_check_models/cifar_simple_scale4_noreg --logs cifar_train_architecture_check_models/cifar_simple_scale4_noreg.txt

# 2) Train a Simple-BN with width factor of 1
#    other training condition same as above
python cifar_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 1 --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model cifar_train_architecture_check_models/cifar_simplebn_scale1_noreg --logs cifar_train_architecture_check_models/cifar_simplebn_scale1_noreg.txt

# 3) Train a WRN 28 with width factor of 2
#    with weight decay of 5e-4
python cifar_training.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 2 --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 5e-4 --save_model cifar_train_architecture_check_models/cifar_wrn_scale2_wd5e-4 --logs cifar_train_architecture_check_models/cifar_wrn_scale2_wd5e-4.txt

# 4) Train a WRN 28 with width factor of 2
#    with excessive weight decay
python cifar_training.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 2 --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 1e-4 --weight_decay_schedule --save_model cifar_train_architecture_check_models/cifar_wrn_scale2_wdexcess --logs cifar_train_architecture_check_models/cifar_wrn_scale2_wdexcess.txt