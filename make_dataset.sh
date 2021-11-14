DATASET_DIR='/home/pkusz/home/PKU_team/gcw'
DATASET_SPLIT_DIR='/home/pkusz/home/PKU_team/new_data'

python extract_label.py --names=$DATASET_DIR

python data_split.py --names=$DATASET_DIR --data_split_root=$DATASET_SPLIT_DIR

python test_split.py --data_path=$DATASET_SPLIT_DIR

