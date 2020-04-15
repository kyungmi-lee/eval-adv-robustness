# touch places365_resnet18.txt

# python places365_evaluate_all.py --attack_type fgsm --eps 0.002 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus >> places365_resnet18.txt
# python places365_evaluate_all.py --attack_type rfgsm --eps 0.002 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus >> places365_resnet18.txt
# python places365_evaluate_all.py --attack_type pgd --eps 0.002 --eps_iter 0.0005 --nb_iter 9 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus >> places365_resnet18.txt
# python places365_evaluate_all.py --attack_type pgd --eps 0.002 --eps_iter 0.0005 --nb_iter 7 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --second_order_init_method miyato >> places365_resnet18.txt

# python places365_evaluate_all.py --attack_type fgsm --eps 0.004  >> places365_resnet18.txt
# python places365_evaluate_all.py --attack_type rfgsm --eps 0.004 --bpda_compensation --relu_sub relu >> places365_resnet18.txt
# python places365_evaluate_all.py --attack_type pgd --eps 0.004 --eps_iter 0.001 --nb_iter 9 --bpda_compensation --relu_sub relu >> places365_resnet18.txt

# python places365_evaluate_all.py --attack_type fgsm --eps 0.004 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus >> places365_resnet18.txt
# python places365_evaluate_all.py --attack_type rfgsm --eps 0.004 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus >> places365_resnet18.txt
# python places365_evaluate_all.py --attack_type pgd --eps 0.004 --eps_iter 0.001 --nb_iter 9 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus >> places365_resnet18.txt
# python places365_evaluate_all.py --attack_type pgd --eps 0.004 --eps_iter 0.001 --nb_iter 7 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --second_order_init_method miyato >> places365_resnet18.txt

touch places365_resnet50.txt
python places365_evaluate_all.py --model_type resnet50 --batch_size 32 --attack_type fgsm --eps 0.002 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> places365_resnet50.txt
python places365_evaluate_all.py --model_type resnet50 --batch_size 32 --attack_type rfgsm --eps 0.002 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> places365_resnet50.txt
python places365_evaluate_all.py --model_type resnet50 --batch_size 32 --attack_type pgd --eps 0.002 --eps_iter 0.0005 --nb_iter 9 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> places365_resnet50.txt

python places365_evaluate_all.py --model_type resnet50 --batch_size 32 --attack_type fgsm --eps 0.002 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> places365_resnet50.txt
python places365_evaluate_all.py --model_type resnet50 --batch_size 32 --attack_type rfgsm --eps 0.002 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> places365_resnet50.txt
python places365_evaluate_all.py --model_type resnet50 --batch_size 32 --attack_type pgd --eps 0.002 --eps_iter 0.0005 --nb_iter 9 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> places365_resnet50.txt
python places365_evaluate_all.py --model_type resnet50 --batch_size 32 --attack_type pgd --eps 0.002 --eps_iter 0.0005 --nb_iter 7 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 --second_order_init_method miyato >> places365_resnet50.txt

python places365_evaluate_all.py --model_type resnet50 --batch_size 32 --attack_type fgsm --eps 0.004  >> places365_resnet50.txt
python places365_evaluate_all.py --model_type resnet50 --batch_size 32 --attack_type rfgsm --eps 0.004 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> places365_resnet50.txt
python places365_evaluate_all.py --model_type resnet50 --batch_size 32 --attack_type pgd --eps 0.004 --eps_iter 0.001 --nb_iter 9 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> places365_resnet50.txt

python places365_evaluate_all.py --model_type resnet50 --batch_size 32 --attack_type fgsm --eps 0.004 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> places365_resnet50.txt
python places365_evaluate_all.py --model_type resnet50 --batch_size 32 --attack_type rfgsm --eps 0.004 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> places365_resnet50.txt
python places365_evaluate_all.py --model_type resnet50 --batch_size 32 --attack_type pgd --eps 0.004 --eps_iter 0.001 --nb_iter 9 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> places365_resnet50.txt
python places365_evaluate_all.py --model_type resnet50 --batch_size 32 --attack_type pgd --eps 0.004 --eps_iter 0.001 --nb_iter 7 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --second_order_init_method miyato --maxpool_sub_p 5 >> places365_resnet50.txt

touch places365_alexnet.txt
python places365_evaluate_all.py --model_type alexnet --batch_size 128 --attack_type fgsm --eps 0.002 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> places365_alexnet.txt
python places365_evaluate_all.py --model_type alexnet --batch_size 128 --attack_type rfgsm --eps 0.002 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> places365_alexnet.txt
python places365_evaluate_all.py --model_type alexnet --batch_size 128 --attack_type pgd --eps 0.002 --eps_iter 0.0005 --nb_iter 9 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> places365_alexnet.txt

python places365_evaluate_all.py --model_type alexnet --batch_size 128 --attack_type fgsm --eps 0.002 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> places365_alexnet.txt
python places365_evaluate_all.py --model_type alexnet --batch_size 128 --attack_type rfgsm --eps 0.002 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> places365_alexnet.txt
python places365_evaluate_all.py --model_type alexnet --batch_size 128 --attack_type pgd --eps 0.002 --eps_iter 0.0005 --nb_iter 9 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> places365_alexnet.txt
python places365_evaluate_all.py --model_type alexnet --batch_size 128 --attack_type pgd --eps 0.002 --eps_iter 0.0005 --nb_iter 7 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 --second_order_init_method miyato >> places365_alexnet.txt

python places365_evaluate_all.py --model_type alexnet --batch_size 128 --attack_type fgsm --eps 0.004  >> places365_alexnet.txt
python places365_evaluate_all.py --model_type alexnet --batch_size 128 --attack_type rfgsm --eps 0.004 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> places365_alexnet.txt
python places365_evaluate_all.py --model_type alexnet --batch_size 128 --attack_type pgd --eps 0.004 --eps_iter 0.001 --nb_iter 9 --bpda_compensation --relu_sub relu --maxpool_sub_p 0 >> places365_alexnet.txt

python places365_evaluate_all.py --model_type alexnet --batch_size 128 --attack_type fgsm --eps 0.004 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> places365_alexnet.txt
python places365_evaluate_all.py --model_type alexnet --batch_size 128 --attack_type rfgsm --eps 0.004 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> places365_alexnet.txt
python places365_evaluate_all.py --model_type alexnet --batch_size 128 --attack_type pgd --eps 0.004 --eps_iter 0.001 --nb_iter 9 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 >> places365_alexnet.txt
python places365_evaluate_all.py --model_type alexnet --batch_size 128 --attack_type pgd --eps 0.004 --eps_iter 0.001 --nb_iter 7 --zero_compensation targeted --targeted_to second --bpda_compensation --relu_sub softplus --maxpool_sub_p 5 --second_order_init_method miyato >> places365_alexnet.txt
