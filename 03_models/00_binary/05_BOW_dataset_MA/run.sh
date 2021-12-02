INPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/02_preprocessed/04_BOW/00_binary/00_2018
OUTPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/03_spy_project/00_data/03_runs/03_BOW/00_TEST

#INPUT_DIR=/home/sibanez/Projects/00_MyInvestor/00_data/02_preprocessed/03_FastText_LM/01_preprocessed/2018
#OUTPUT_DIR=/home/sibanez/Projects/00_MyInvestor/00_data/03_runs/02_LSTM/00_TEST

python -m ipdb train_test.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --task=Train \
    \
    --bow_len=2000 \
    --num_classes=1 \
    --seed=1234 \
    --use_cuda=True \
    --remove_empty_entries=True \
    \
    --n_epochs=10 \
    --batch_size_train=6000 \
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
    --model_file=model.pt.2 \
    --batch_size_test=200 \
    --gpu_id_test=0 \

read -p 'EOF'

# OPTIONS
#--task=Train / Test
#--gpu_ids_train=0,1,2,3
