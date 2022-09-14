dataset is same as you downloaded in NSP file.\
Then you can run the cmd: 

CUDA_VISIBLE_DEVICES=0 python -u main.py \
--data_path ./TripAdvisor/reviews_profile_index.pickle \
--index_dir ./TripAdvisor/1/ \
--cuda \
--checkpoint ./TA_output/TA_1_df_gamma_0.25_linb_steps_100/ \
--image_fea_path None \
--key_len 4 \
--epochs_teacher 200 \
--epochs_student 200 \
--peter_mask \
--profile_len 5 \
--feature_extract_path None \
--user_profile_path ./TripAdvisor/user_profile.json \
--item_profile_path ./TripAdvisor/item_profile.json \
--batch_size 128 \
--item_keyword_path None \
--mae \
--endure_times 5 \
--words 15 \
--test_step 1 \
--gamma 0.25 \
--alpha_beta_schedule linb \
--n_steps 100 \
--log_interval 500  >> ./TA_output/TA_1_df_gamma_0.25_linb_steps_100.log


Notice: 
The DF version code has some differences with NSP code so you need make a new folder for it. and pls customize the dataset directoty to your file folder and change the --checkpoint and log name when you run different hyper-parameters.

You need to try:\
--gamma from （0， 1）\
--alpha_beta_schedule in 'cosb,lina,linb,palpha'\
--n_steps from 100-1000, or bigger (but it will take more time to train)
