with feature:
CUDA_VISIBLE_DEVICES=0 python -u main.py \
--data_path ./Amazon/Clothing_Shoes_and_Jewelry/reviews_profile_index.pickle \
--index_dir ./Amazon/Clothing_Shoes_and_Jewelry/1/ \
--cuda \
--checkpoint ./CSJ_output/rank_key_profile_transformer_256/AZ_CSJ_1/ \
--image_fea_path ./Amazon/Clothing_Shoes_and_Jewelry/Res_embeddings/ \
--key_len 4 \
--epochs 100 \
--peter_mask \
--use_feature \
--profile_len 5 \
--feature_extract_path ./Amazon/Clothing_Shoes_and_Jewelry/CSJ_feature_keybert.pickle \
--user_profile_path ./Amazon/Clothing_Shoes_and_Jewelry/user_profile.json \
--item_profile_path ./Amazon/Clothing_Shoes_and_Jewelry/item_profile.json \
--batch_size 128 \
--profile_words 15 \
--item_keyword_path ./Amazon/Clothing_Shoes_and_Jewelry/CSJ_keyword_one_four.pickle \
--words 15 >> ./CSJ_output/rank_key_profile_transformer_256/AZ_CSJ_1.log


without feature:

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--data_path ./Amazon/Clothing_Shoes_and_Jewelry/reviews_profile_index.pickle \
--index_dir ./Amazon/Clothing_Shoes_and_Jewelry/1/ \
--cuda \
--checkpoint ./CSJ_output/rank_key_profile_transformer_256/AZ_CSJ_1/ \
--image_fea_path ./Amazon/Clothing_Shoes_and_Jewelry/Res_embeddings/ \
--key_len 4 \
--epochs 100 \
--peter_mask \
--profile_len 5 \
--feature_extract_path ./Amazon/Clothing_Shoes_and_Jewelry/CSJ_feature_keybert.pickle \
--user_profile_path ./Amazon/Clothing_Shoes_and_Jewelry/user_profile.json \
--item_profile_path ./Amazon/Clothing_Shoes_and_Jewelry/item_profile.json \
--batch_size 128 \
--profile_words 15 \
--item_keyword_path ./Amazon/Clothing_Shoes_and_Jewelry/CSJ_keyword_one_four.pickle \
--words 15 >> ./CSJ_output/rank_key_profile_transformer_256/AZ_CSJ_1.log