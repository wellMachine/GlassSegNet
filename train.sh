CUDA_VISIBLE_DEVICES="1" \
python train.py \
--dinov3_path "" \
--input_size 352 \
--train_image_path "" \
--train_mask_path "" \
--save_path "" \
--epoch 200 \
--lr 0.0005 \
--batch_size 16