# !/bin/zsh -l
DATASET_NAME=phototourism

for SCENE_NAME in british_museum florence_cathedral_side lincoln_memorial_statue milan_cathedral mount_rushmore piazza_san_marco sagrada_familia st_pauls_cathedral london_bridge
    do
        echo "Process scene: $SCENE_NAME"
        python -u $PROJECT_DIR/tools/parse_data/parse_IMC_dataset.py \
                --scene_name=$SCENE_NAME \
                --split=test \
    done