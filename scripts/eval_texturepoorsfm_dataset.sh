# !/bin/zsh -l
DEFAULT_EXP_NAME=''
DATASET_BASE_DIR='SfM_dataset'
OUTPUT_BASE_DIR='SfM_metric_output'

for MATCHERNAME in loftr_official aspanformer matchformer
    do
        # '1012' '1026' '1029' are validation scenes
        for DATASET_NAME in '1000' '1002' '1003' '1005' '1006' '1007' '1008' '1010' '1014' '1021' '1024' '1025' '1027' '1030'
            do
                python eval_dataset.py +texturepoor_sfm=dfsfm dataset_name=TexturePoorSfM_dataset/${DATASET_NAME} exp_name=${EXP_NAME} dataset_base_dir=${DATASET_BASE_DIR} output_base_dir=${OUTPUT_BASE_DIR} neuralsfm.NEUSFM_coarse_matcher=${MATCHERNAME}
            done
    done