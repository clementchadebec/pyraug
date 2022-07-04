#$ -S /bin/bash
#$ -l tmem=11G
#$ -l gpu=true
#$ -l h_rt=100:0:0
#$ -j y
#$ -N CS_cv0_nc16
#$ -cwd
hostname
date
python3 -u train_cbct.py \
--project GenSynthSeg \
--exp_name GenSynthSeg_test \
--data_path ~/thesis/Data/others/deepRegData/fullResCropIntensityClip_resampled \
--batch_size 8 \
--cv 0 \
--input_shape 64 101 91 \
--lr 3e-5 \
--affine_scale 0.15 \
--save_frequency 500 \
--num_epochs 50000 \
--w_dce 1.0 \
--using_HPC 0 \
--nc_initial 16