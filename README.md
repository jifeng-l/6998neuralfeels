## What is in this repo
A course project based on NeuralFeels. Check the original repo at https://github.com/facebookresearch/neuralfeels.

### What we have done
Build a retrain framework for and apply mamba backbone with the perception module - tactile transformer, which is a vision transformer trained with tactile sensor dataset.

## Setup

### 1. Clone repository

```bash
git clone git@github.com:jifeng-l/6998neuralfeels.git
```

### 2. Install the `neuralfeels` environment

Our preferred choice is via `micromamba` ([link](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)). Run the bash script that sets everything up:

```bash
./install.sh -e neuralfeels
micromamba activate neuralfeels  
```

### 3. Download the FeelSight dataset

Clone the 🤗 dataset and unzip all files. Make sure you have `git-lfs` installed, it may take a while: 

```bash
cd data && git clone https://huggingface.co/datasets/suddhu/Feelsight
mv Feelsight/* . && rm -r Feelsight
find . -name "*.tar.gz" -exec tar -xzf {} \; -exec rm {} \; && cd ..
```

The artifacts should be in the `data/` directory:
```bash 
data/
 ├──  feelsight/ # simulation dataset, 25G
 ├──  feelsight_real/ # real-world dataset, 15G
 ├──  feelsight_occlusion/ # simulated occlusion dataset, 12G
 └──  assets/ # ground-truth 3D models
```

### 4. Download models

Get the `tactile_transformer` 🤗 model: 
```bash
cd data && git clone https://huggingface.co/suddhu/tactile_transformer && cd ..
```

Get the [Segment-anything](https://segment-anything.com/) weights:
```bash
mkdir -p data/segment-anything && cd data/segment-anything
for model in sam_vit_h_4b8939.pth sam_vit_l_0b3195.pth sam_vit_b_01ec64.pth; do
  gdown https://dl.fbaipublicfiles.com/segment_anything/$model
done
cd ../..
```

## Run NeuralFeels

Run interactive perception experiments with our FeelSight data from both the simulated and real-world in-hand experiments. Try one of our `--preset` commands or use the `--help` flag to see all options:

```bash
$ ./scripts/run --help
```
```bash
Usage: ./scripts/run DATASET SLAM_MODE MODALITY OBJECT LOG FPS RECORD OPEN3D
Arguments:
  DATASET: string    # The dataset to be used, options are 'feelsight', 'feelsight_real'
  SLAM_MODE: string  # The mode to be used, options are 'slam', 'pose', 'map'
  MODALITY: string   # The modality to be used, options are 'vitac', 'vi', 'tac'
  OBJECT: string     # The object to be used, e.g., '077_rubiks_cube'
  LOG: string        # The log identifier, e.g., '00', '01', '02'
  FPS: integer       # The frames per second, e.g., '1', '5'
  RECORD: integer    # Whether to record the session, options are '1' (yes) or '0' (no)
  OPEN3D: integer    # Whether to use Open3D, options are '1' (yes) or '0' (no)
Presets:
  --slam-sim         # Run neural SLAM in simulation with rubber duck
  --pose-sim         # Run neural tracking in simulation with Rubik's cube
  --slam-real        # Run neural SLAM in real-world with bell pepper
  --pose-real        # Run neural tracking in real-world with large dice
  --three-cam        # Three camera pose tracking in real-world with large dice
  --occlusion-sim    # Run neural tracking in simulation with occlusion logs
```

This will launch the GUI and train the neural field model live. You must have a performant GPU (tested on RTX 3090/4090) for best results. In our work, we've experimented with an FPS of 1-5Hz, optimizing the performance is future work. See below for the interactive visualization of sensor measurements, mesh, SDF, and neural field.

https://github.com/user-attachments/assets/63fc2992-d86e-4f69-8fc9-77ede86942c7

## Other scripts 

Here are some additional scripts to test different modules of NeuralFeels:
| Task                                | Command                                                       |
|-------------------------------------|---------------------------------------------------------------|
| Test the tactile-transformer model  | ```python neuralfeels/contrib/tactile_transformer/touch_vit.py ``` |
| Test prompt-based visual segmentation      | ```python neuralfeels/contrib/sam/test_sam.py ```         |
| Allegro URDF visualization in Open3D| ```python /neuralfeels/contrib/urdf/viz.py ```            |
| Show FeelSight object meshes in [`viser`](https://github.com/nerfstudio-project/viser)   | ```python neuralfeels/viz/show_object_dataset.py ```      |

## Folder structure
```bash
neuralfeels
├── data              # downloaded datasets and models
├── neuralfeels       # source code 
│   ├── contrib       # based on third-party code
│   ├── datasets      # dataloader and dataset classes
│   ├── eval          # metrics and plot scripts
│   ├── geometry      # 3D and 2D geometry functions
│   ├── modules       # frontend and backend modules
│   └── viz           # rendering and visualization
├── outputs           # artifacts from training runs
└── scripts           # main run script and hydra confids
```
