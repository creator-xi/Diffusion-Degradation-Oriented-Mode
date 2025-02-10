# Diffusion Degradation Oriented Mode

## Environment
<p>
Please clone this repo and run following command locally for install the environment:
<pre>
conda create --name ddom
pip install -r requirements.txt
</pre>
</p>

## Datasets
Please put the training dataset into <code>datasets/</code> folder and test dataset into <code>exp/datasets/</code>. The restoration results will be saved in <code>exp/image_samples/</code>. Datasets can be downloaded at [link1](https://aistudio.baidu.com/datasetdetail/49050#:~:text=(2610.14M)-,%E4%B8%8B%E8%BD%BD,-File%20Name) and [link2](https://aistudio.baidu.com/datasetdetail/178435#:~:text=(6361.69M)-,%E4%B8%8B%E8%BD%BD,-File%20Name).

## Quick Start
<p>
We have prepared training and testing scripts for various image restoration tasks in the <code>exp.sh</code> file. For example, if we want to train DDOM for SR, we can run below command:
<pre>
CUDA_VISIBLE_DEVICES=0 python train.py --A_type adapter --epochs 100 --nums_rb 20 --add_temb True --res_adap True \
--lr 1e-5 --ni --config celeba_hq.yml --path_y celeba_hq --eta 0.85 --batch_size 2 --init_x Apy \
--deg "sr_averagepooling" --deg_scale 4 --sigma_y 0. -i main_sr4 --train_size 2 \
--save save/main_sr4
</pre>
</p>
where <code>nums_rb</code> is the number of layers in DO-Adapter, <code>deg</code> is degradation task, <code>config</code> is the name of the config file (see <code>configs/</code> folder), <code>path_y</code> is the path of test images (please save in <code>exp/datasets/</code>).

##
This implementation is extended based on [DDNM](https://github.com/wyhuai/DDNM/tree/main), thanks!