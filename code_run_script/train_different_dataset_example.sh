mkdir train_different_dataset_example

# This bash file displays example commands on training with different dataset (SVHN & TinyImageNet)

# SVHN; Simple-BN (w=4)
# train w/o explicit regularization, with SGD
python svhn_training.py --train_val_split --disable_adv --no_adv_eval --model_type simple --scale_factor 4 --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --optimizer sgd --save_model train_different_dataset_example/svhn_simple4_noreg --logs train_different_dataset_example/svhn_simple4_noreg.txt

# TinyImageNet; WRN 50 (w=1)
# train w/o explicit regularization, with Adam (beta_1=0.9, beta_2=0.99)
python tinyimagenet_training.py --train_val_split --disable_adv --no_adv_eval --model_type wrn --scale_factor 1 --lr 1e-1 --early_stop --early_stop_criteria cln --weight_decay 0 --optimizer adam --save_model train_different_dataset_example/tinyimagenet_wrn1_noreg --logs train_different_dataset_example/tinyimagenet_wrn1_noreg.txt