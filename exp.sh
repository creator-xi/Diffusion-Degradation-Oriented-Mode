## SR 4x
# train
CUDA_VISIBLE_DEVICES=0 python train.py --A_type adapter --epochs 100 --nums_rb 20 --add_temb True --res_adap True \
--lr 1e-5 --ni --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --batch_size 2 --init_x Apy \
--deg "sr_averagepooling" --deg_scale 4 --sigma_y 0. -i main_sr4 --train_size 1 \
--save save/main_sr4

# test sr4 adapter
CUDA_VISIBLE_DEVICES=0 python main.py --A_type adapter --epochs 100 --nums_rb 20 --add_temb True --res_adap True \
--lr 1e-5 --ni --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --batch_size 2 --init_x Apy \
--deg "sr_averagepooling" --deg_scale 4 --sigma_y 0. -i main_sr4 --train_size 1 \
--adap_pth save/main_sr4_xxxxxxxx_checkpoint.pth  # please modify the path to use right checkpoint


## inpainting facemask
# train
CUDA_VISIBLE_DEVICES=0 python train.py --A_type adapter --epochs 100 --nums_rb 20 --add_temb True --res_adap True \
--lr 1e-5 --ni --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --batch_size 2 --init_x Apy \
--deg "inpainting" --mask facemask --sigma_y 0. -i img_inp_facemask --train_size 1 \
--save save/main_inp_face

# test inpainting adapter
CUDA_VISIBLE_DEVICES=0 python main.py --A_type adapter --epochs 100 --nums_rb 20 --add_temb True --res_adap True \
--lr 1e-5 --ni --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --batch_size 2 --init_x Apy \
--deg "inpainting" --mask facemask --sigma_y 0. -i img_inp_facemask --train_size 1 \
--adap_pth save/main_inp_face_xxxxxxxx_checkpoint.pth

## CS 10%
# train
CUDA_VISIBLE_DEVICES=0 python train.py --A_type adapter --epochs 100 --nums_rb 20 --add_temb True --res_adap True \
--lr 1e-5 --ni --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --batch_size 2 --init_x Apy \
--deg "cs_blockbased" --deg_scale 0.10 --sigma_y 0. -i main_cs10 --train_size 1 \
--save save/main_cs10

# test csbb10 adapter
CUDA_VISIBLE_DEVICES=0 python main.py --A_type adapter --epochs 100 --nums_rb 20 --add_temb True --res_adap True \
--lr 1e-5 --ni --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --batch_size 2 --init_x Apy \
--deg "cs_blockbased" --deg_scale 0.10 --sigma_y 0. -i main_cs10 --train_size 1 \
--adap_pth save/main_cs10_xxxxxxxx_checkpoint.pth


## deblur
CUDA_VISIBLE_DEVICES=0 python train.py --A_type adapter --epochs 100 --nums_rb 20 --add_temb True --res_adap True \
--lr 1e-5 --ni --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --batch_size 2 --init_x Apy \
--deg "deblur_gauss" --sigma_y 0. -i main_deblur_g --train_size 1 \
--save save/main_deblur_g

# test deblur adapter
CUDA_VISIBLE_DEVICES=2 python main.py --A_type adapter --epochs 100 --nums_rb 20 --add_temb True --res_adap True \
--lr 1e-5 --ni --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --batch_size 2 --init_x Apy \
--deg "deblur_gauss" --sigma_y 0. -i main_deblur_g --train_size 1 \
--adap_pth save/main_deblur_g_xxxxxxxx_checkpoint.pth





