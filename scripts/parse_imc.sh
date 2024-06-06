# !/bin/zsh -l
DATASET_NAME=phototourism

for SCENE_NAME in british_museum florence_cathedral_side lincoln_memorial_statue milan_cathedral mount_rushmore piazza_san_marco sagrada_familia st_pauls_cathedral london_bridge
    do
        echo "Process scene: $SCENE_NAME"
        python -u tools/parse_data/parse_IMC_dataset.py --IMC_base_dir="SfM_datasets/IMC_2021" --scene_name=$SCENE_NAME --split=test --output_base_dir="SfM_datasets/IMC_dataset"
    done