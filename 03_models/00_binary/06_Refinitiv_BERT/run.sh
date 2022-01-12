#INPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/04_full_final_2019
#OUTPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/03_runs/01_ProsusAI_finbert/00_binary/05_run_full_2019_final
#MODEL_PATH=

INPUT_DIR=/home/sibanez/Projects/00_MyInvestor/00_data/02_preprocessed/05_RefinitivBert/00_binary/00_2018_sorted
OUTPUT_DIR=/home/sibanez/Projects/00_MyInvestor/00_data/03_runs/04_Refinitiv_Bert/00_2018_TEST
MODEL_PATH=/data/news/LanguageModels/model_checkpoints/refinitiv-bert-large-uncased/checkpoint-1013628/

python train_test.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --task=Train \
    \
    --model_path=$MODEL_PATH \
    --seq_len=256 \
    --num_labels=1 \
    --n_heads=8 \
    --hidden_dim=1024 \
    --pad_idx=0 \
    --seed=1234 \
    --use_cuda=True \
    \
    --n_epochs=10 \
    --batch_size_train=70 \
    --shuffle_train=True \
    --drop_last_train=False \
    --dev_train_ratio=2 \
    --train_toy_data=False \
    --len_train_toy_data=30 \
    --lr=2e-5 \
    --wd=1e-6 \
    --dropout=0.4 \
    --momentum=0.9 \
    --save_final_model=True \
    --save_model_steps=True \
    --save_step_cliff=0 \
    --gpu_ids_train=0,1,2,3 \
    \
    --test_file=model_test.pkl \
    --model_file=model.pt.8 \
    --batch_size_test=200 \
    --gpu_id_test=0 \

#read -p 'EOF'

#--model_name=nlpaueb/legal-bert-small-uncased \
#--hidden_dim=512 \

#--task=Train / Test
#--batch_size=280 / 0,1,2,3
