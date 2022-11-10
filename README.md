# lat_var_bev_pred_model
Latent variable BEV prediction model

Python 3.9.6

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install pytorch-lightning
pip install pytorch-lightning[extra]
pip install matplotlib
// pip install onnx onnxruntime-gpu




# How to use

## 0. Generate preprocessed samples

```shell
python datamodules/bev_datamodule.py
```

`pc-accumulation-lib` output directory (ex: `bev_kitti360_256px_aug_gt_3_rev`) will be converted to directory of preprocessed torch.Tensor samples (ex: `bev_kitti360_256px_aug_gt_3_rev_preproc`).

TODO: Dynamic input arguments

## 1. Latent variable predictive model (structure)

```shell
python lat_var_pred_model.py \
    --accelerator gpu \
    --devices 4 \
    --precision 32 \
    --profiler simple \
    --num_workers 16 \
    --in_ch 1 \
    --in_size 256 \
    --batch_size 32 \
    --enc_dim 128 \
    --lat_dim 32 \
    --z_hidden_dim 256 \
    --y_dim 10 \
    --enc_str 2x16,2x32,2x64,2x128,2x256,2x256,2x512,2x128 \
    --dec_str 2x512,2x256,2x256,2x128,2x64,2x32,2x16 \
    --lr 2e-4 \
    --beta_oracle 0 \
    --beta_mix 1e-4 \
    --check_val_every_n_epoch 10 \
    --train_data_dir bev_kitti360_256px_aug_gt_3_rev_preproc_train \
    --val_data_dir bev_kitti360_256px_aug_gt_3_rev_preproc_val \
    --sample_type road
```

Trains model on preprocessed samples and outputs a checkpoint file `lat_var_pred_version6_ep999.ckpt`.

```shell
python /home/r_karlsson/workspace6/lat_var_bev_pred_model/lat_var_pred_model_prod.py \
    --checkpoint_path lightning_logs/version_6/checkpoints/epoch=999-step=701000.ckpt \
    --use_oracle_sample \
    --batch_size 1 \
    --in_ch 1 \
    --in_size 256 \
    --enc_dim 128 \
    --lat_dim 32 \
    --z_hidden_dim 256 \
    --y_dim 10 \
    --enc_str 2x16,2x32,2x64,2x128,2x256,2x256,2x512,2x128 \
    --dec_str 2x512,2x256,2x256,2x128,2x64,2x32,2x16 \
    --data_dir bev_kitti360_256px_aug_gt_3_rev_preproc_train \
    --sample_type road
```

Generates a model configuration dict file `lat_var_pred_model_config.pkl` for initializing model.

## 2. Road adversarial generative model (texture)

```shell
python road_adv_pred.py \
    --accelerator gpu \
    --devices 4 \
    --precision 32 \
    --num_workers 4 \
    --profiler simple \
    --max_epochs -1 \
    --in_ch 2 \
    --in_size 256 \
    --batch_size 16 \
    --enc_dim 512 \
    --gen_enc_str 2x64,2x64,2x64,2x64,2x128,2x128,2x256,2x256 \
    --gen_dec_str 2x256,2x256,2x128,2x128,2x64,2x64,2x64,2x64 \
    --adv_enc_str 2x32,2x32,2x64,2x64,2x128,2x128,2x256,2x256 \
    --adv_dec_str 2x256,2x256,2x128,2x128,2x64,2x64,2x32,2x32 \
    --lr 1e-4 \
    --check_val_every_n_epoch 1 \
    --train_data_dir bev_kitti360_256px_aug_gt_3_rev_preproc_train \
    --val_data_dir bev_kitti360_256px_aug_gt_3_rev_preproc_val \
    --sample_type all \
    --objective_type mse
```

### 3. 