#INPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/01_full_dataset
#OUTPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/03_runs/01_ProsusAI_finbert/01_run_binary_full

INPUT_DIR=/home/sibanez/Projects/MyInvestor/NLP/01_spyproject/00_data/02_preprocessed/02_ProsusAI_finbert/00_binary/01_FULL
OUTPUT_DIR=/home/sibanez/Projects/MyInvestor/NLP/01_spyproject/02_runs/01_TEST_1

python train_test.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --task=Train \
    \
    --model_name=ProsusAI/finbert \
    --seq_len=64 \
    --num_labels=1 \
    --n_heads=8 \
    --hidden_dim=768 \
    --pad_idx=0 \
    --seed=1234 \
    --use_cuda=True \
    \
    --n_epochs=10 \
    --batch_size_train=1600 \
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
    --model_file=model.pt.3 \
    --batch_size_test=200 \
    --gpu_id_test=0 \

read -p 'EOF'

#--model_name=nlpaueb/legal-bert-small-uncased \
#--hidden_dim=512 \

#--task=Train / Test
#--batch_size=40
#--batch_size=25 / 0,1,2,3
#--n_epochs=100
#--max_n_pars=200
