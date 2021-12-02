#INPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/03_FastText_LM/01_preprocessed/00_2018
#ID_2_VEC_PATH=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/03_FastText_LM/01_preprocessed/00_2018/word_to_vec.pkl
#OUTPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/03_runs/02_embeddings/00_binary

INPUT_DIR=/home/sibanez/Projects/00_MyInvestor/00_data/02_preprocessed/03_FastText_LM/01_preprocessed/2018
ID_2_VEC_PATH=/home/sibanez/Projects/00_MyInvestor/00_data/02_preprocessed/03_FastText_LM/01_preprocessed/2018/word_to_vec.pkl 
OUTPUT_DIR=/home/sibanez/Projects/00_MyInvestor/00_data/03_runs/02_LSTM/05_100ep_non-empty_entries_lr_2e-3

python train_test.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --id_2_vec_path=$ID_2_VEC_PATH \
    --task=Train \
    \
    --model_name=ProsusAI/finbert \
    --seq_len=256 \
    --num_classes=1 \
    --emb_dim=300 \
    --hidden_dim=100 \
    --pad_idx=1 \
    --seed=1234 \
    --use_cuda=True \
    --remove_empty_entries=True \
    \
    --n_epochs=100 \
    --batch_size_train=24000 \
    --shuffle_train=True \
    --drop_last_train=False \
    --dev_train_ratio=2 \
    --train_toy_data=False \
    --len_train_toy_data=30 \
    --lr=2e-3 \
    --wd=1e-6 \
    --dropout=0.4 \
    --momentum=0.9 \
    --freeze_emb=False \
    --save_final_model=True \
    --save_model_steps=True \
    --save_step_cliff=90 \
    --gpu_ids_train=0,1,2,3 \
    \
    --test_file=model_test.pkl \
    --model_file=model.pt.90 \
    --batch_size_test=200 \
    --gpu_id_test=0 \

#read -p 'EOF'

#PREVIOUS
#--lr=2e-5

# OPTIONS
#--task=Train / Test
#--gpu_ids_train=0,1,2,3
