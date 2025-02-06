# Cora
python gold.py --backbone gcn --dataset cora --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -3 --device 0
python gold.py --backbone gcn --dataset cora --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -5 --m_out -3 --device 0
python gold.py --backbone gcn --dataset cora --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -4 --m_out -2 --device 0

# Amazon-Photo
python gold.py --backbone gcn --dataset amazon-photo --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -9 --m_out -3 --device 0
python gold.py --backbone gcn --dataset amazon-photo --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -9 --m_out -3 --device 0
python gold.py --backbone gcn --dataset amazon-photo --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -9 --m_out -5 --device 0

# Coauthor-CS
python gold.py --backbone gcn --dataset coauthor-cs --ood_type structure --mode detect --use_bn --use_prop --use_reg --m_in -9 --m_out -5 --device 0
python gold.py --backbone gcn --dataset coauthor-cs --ood_type feature --mode detect --use_bn --use_prop --use_reg --m_in -9 --m_out -5 --device 0
python gold.py --backbone gcn --dataset coauthor-cs --ood_type label --mode detect --use_bn --use_prop --use_reg --m_in -7 --m_out -5 --device 0

# Twitch
python gold.py --backbone gcn --dataset twitch --mode detect --use_bn --use_prop --use_reg --m_in -6 --m_out -4 --device 0

# Arxiv
python gold.py --backbone gcn --dataset arxiv --mode detect --use_bn --use_prop --use_reg --m_in -9 --m_out -3 --device 0

