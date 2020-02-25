mkdir cifar_train_functionality_check_models

# Train a model on CIFAR-10 dataset

# 1) Train a Simple model (w/o batch norm) with scale factor=1
#    using SGD w/ momentum=0.9, no explicit regularization
#    Train/val split (90%/10%)
#    Save model and log file in cifar_train_functionality_check_models
python cifar_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 1 --no_batch_norm --lr 1e-2 --early_stop --early_stop_criteria cln --weight_decay 0 --save_model cifar_train_functionality_check_models/cifar_simple_scale1_noreg --logs cifar_train_functionality_check_models/cifar_simple_scale1_noreg.txt

# 2) Train the above model now with weight decay (fixed to 5e-4)
python cifar_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 1 --no_batch_norm --lr 1e-2 --early_stop --early_stop_criteria cln --weight_decay 5e-4 --save_model cifar_train_functionality_check_models/cifar_simple_scale1_wd5e-4 --logs cifar_train_functionality_check_models/cifar_simple_scale1_wd5e-4.txt

# 3) Train the above model with "excessive" wegith decay
#    Weight decay hyperparameter (strength) initialize to be 1e-4
#    Then per every 40 epochs, the hyperparameter gets multiplied with factor of 10
#    Resulting in final 1e-2 at the end of the training
python cifar_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 1 --no_batch_norm --lr 1e-2 --early_stop --early_stop_criteria cln --weight_decay 1e-4 --weight_decay_schedule --save_model cifar_train_functionality_check_models/cifar_simple_scale1_wdexcess --logs cifar_train_functionality_check_models/cifar_simple_scale1_wdexcess.txt

# 4) Train the above model with Spectral normalization
python cifar_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 1 --no_batch_norm --lr 1e-2 --early_stop --early_stop_criteria cln --weight_decay 0 --spectral_norm --save_model cifar_train_functionality_check_models/cifar_simple_scale1_spectral --logs cifar_train_functionality_check_models/cifar_simple_scale1_spectral.txt

# 5) Train the above model with Orthonormal regularization of factor 1e-3
python cifar_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 1 --no_batch_norm --lr 1e-2 --early_stop --early_stop_criteria cln --weight_decay 0 --orthonormal 1e-3 --save_model cifar_train_functionality_check_models/cifar_simple_scale1_orth --logs cifar_train_functionality_check_models/cifar_simple_scale1_orth.txt

# 6) Train the above model with Jacobian regularization of factor 1
python cifar_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 1 --no_batch_norm --lr 1e-2 --early_stop --early_stop_criteria cln --weight_decay 0 --jacobian 1 --save_model cifar_train_functionality_check_models/cifar_simple_scale1_jaco --logs cifar_train_functionality_check_models/cifar_simple_scale1_jaco.txt

# 7) Train the above model using adversarial training (PGD eps=2/255, eps_iter=1/255, nb_iter=3)
#    with weight decay of 5e-4
python cifar_training.py --train_val_split --model_type simple --scale_factor 1 --no_batch_norm --lr 1e-2 --early_stop --early_stop_criteria adv --weight_decay 5e-4 --attack_type pgd --eps 0.0078 --eps_iter 0.004 --nb_iter 3 --save_model cifar_train_functionality_check_models/cifar_simple_scale1_adv --logs cifar_train_functionality_check_models/cifar_simple_scale1_adv.txt