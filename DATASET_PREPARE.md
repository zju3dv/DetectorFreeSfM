### IMC 2021 Dataset:
```
# Download:
cd /your/repo/path/DetectorFreeSfM/SfM_dataset
mkdir IMC2021 && cd IMC2021
wget https://www.cs.ubc.ca/research/kmyi_data/imc2021-public/imc-2021-test-gt-phototourism.tar.gz

tar -xzvf imc-2021-test-gt-phototourism.tar.gz

# Convert data format:
cd /your/repo/path/DetectorFreeSfM/SfM_dataset
sh scripts/parse_imc.sh
```

### ETH3D Dataset:
```
# Download all undistorted images of training and test datasets, and GT scans of training dataset:

cd /your/repo/path/DetectorFreeSfM/SfM_dataset
mkdir ETH3D_source_data && cd ETH3D_source_data

wget https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z
wget https://www.eth3d.net/data/multi_view_training_dslr_scan_eval.7z

wget https://www.eth3d.net/data/multi_view_test_dslr_undistorted.7z

7z x multi_view_training_dslr_undistorted.7z && rm multi_view_training_dslr_undistorted.7z
7z x multi_view_training_dslr_scan_eval.7z && rm multi_view_training_dslr_scan_eval.7z
7z x multi_view_test_dslr_undistorted.7z && rm multi_view_test_dslr_undistorted.7z

# Now we get all 25 scenes. Then, convert data format:
cd /your/repo/path/DetectorFreeSfM/SfM_dataset

# For SfM dataset:
python tools/parse_eth3d_dataset.py --triangulation_mode False --output_base_dir SfM_dataset/eth3d_dataset

# For Triangulation dataset:
python tools/parse_eth3d_dataset.py --triangulation_mode True --output_base_dir SfM_dataset/eth3d_triangulation_dataset
```

### TexturePoorSfM Dataset:
Download dataset from [here](https://drive.google.com/file/d/1UDo8K0uYCi-YtpwLviE6hT-6Mxm5jHYH/view?usp=sharing) and place it under `SfM_dataset` folder.
```
tar -xvf TexturePoorSfM_dataset.tar
```

### Training Dataset:
Our multi-view matching refinement model is trained on MegaDepth dataset. If you don't need training, please skip this part.

Firstly, download MegaDepth depth maps and undistorted images following [LoFTR's instruction](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md), and place them as following structure:
```
repo_path/megadepth
    - Undistorted_SfM
    - phoenix
        - S6
            - zl548
                - MegaDepth_v1
```

Then, download the multiview matching training scene indices from [here](https://drive.google.com/file/d/1WCrlUEf4xU_7nnjhbuu-_Td0oabVgISM/view?usp=sharing) and unzip it:
```
tar -xvf multiview_matching_indices.tar
```
Finally, place the indices under `megadepth` folder: