# !/bin/zsh -l
DEFAULT_EXP_NAME=''
EXP_NAME=${1:-${DEFAULT_EXP_NAME}}

DATASET_BASE_DIR='SfM_dataset'
OUTPUT_BASE_DIR='SfM_metric_output'
for MATCHERNAME in loftr_official aspanformer matchformer
    do
        for DATASET_NAME in phototourism_florence_cathedral_side phototourism_lincoln_memorial_statue phototourism_milan_cathedral phototourism_mount_rushmore phototourism_piazza_san_marco phototourism_sagrada_familia phototourism_st_pauls_cathedral phototourism_london_bridge phototourism_british_museum
            do
                echo "Run ${DATASET_NAME}"
                python eval_dataset.py +IMC=dfsfm dataset_name=IMC_dataset/${DATASET_NAME} exp_name=${EXP_NAME} dataset_base_dir=${DATASET_BASE_DIR} output_base_dir=${OUTPUT_BASE_DIR} neuralsfm.NEUSFM_coarse_matcher=${MATCHERNAME}
            done
    done