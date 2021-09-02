#INPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/01_preprocessed
#OUTPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/03_runs

INPUT_DIR=/data/rsg/nlp/sibanez/00_temp/01_fin_pred/00_data/01_preprocessed
OUTPUT_DIR=/data/rsg/nlp/sibanez/00_temp/01_fin_pred/00_data/02_runs/00_run_test

python -m ipdb train_test.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --task=Test \
    \
    --model_name=nlpaueb/legal-bert-small-uncased
    --seq_len=256 \
    --num_labels=3 \
    --n_heads=8 \
    --hidden_dim=512 \
    --pad_idx=0 \
    --seed=1234 \
    --use_cuda=True \
    \
    --n_epochs=10 \
    --batch_size_train=600 \
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
    --gpu_ids_train=1,2 \
    \
    --test_file=model_test.pkl \
    --model_file=model.pt.9 \
    --batch_size_test=200 \
    --gpu_id_test=1 \

#read -p 'EOF'

#--task=Train / Test
#--model_name=nlpaueb/legal-bert-small-uncased
#--batch_size=40
#--batch_size=25 / 0,1,2,3
#--n_epochs=100
#--max_n_pars=200
