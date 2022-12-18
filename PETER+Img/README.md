# Image of Amazon Dataset link:
https://drive.google.com/file/d/12iGtpJdWbiIT-j_HVow-qf9TlH-KEger/view?usp=sharing

# Amazon Dataset link:
https://lifehkbueduhk-my.sharepoint.com/personal/16484134_life_hkbu_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F16484134%5Flife%5Fhkbu%5Fedu%5Fhk%2FDocuments%2FCIKM20%2DNETE%2DDatasets&ga=1

# The results of directly concatenate image embeddings on AM-CSJ are updated in following link:
https://docs.google.com/spreadsheets/d/1p5PjSx6aRJbZpFPIkfCmCX3wtjNYlgf5s7u-y2toNn4/edit#gid=1645809363


# Download img embeddings and datasets from google drive, then you can run the cmd: 
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--data_path ./Datasets/Amazon/ClothingShoesAndJewelry/reviews.pickle \
--index_dir ./Datasets/Amazon/ClothingShoesAndJewelry/1/ \
--cuda \
--checkpoint ./model_saver/ \
--image_fea_path ./Embeddings/CSJ/Res_embeddings/ \
--epochs 200 \
--image_fea \
--peter_mask \
--img_reg 0.01 \
--batch_size 128 >> ./CSJ.log



