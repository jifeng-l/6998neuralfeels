<h1 align="center"><img src=".github/logo.svg"
    width=30px>
      NeuralFeels with neural fields <br/>
      <small>Visuo-tactile perception for in-hand manipulation</small></h1>
</h1>

<div align="center">
  <a href="http://www.cs.cmu.edu/~sudhars1/" target="_blank">Sudharshan Suresh</a> &nbsp;•&nbsp;
  <a href="https://haozhi.io/" target="_blank">Haozhi Qi</a> &nbsp;•&nbsp;
  <a href="https://scholar.google.com/citations?user=9bt2Z5QAAAAJ&hl=en" target="_blank">Tingfan Wu</a> &nbsp;•&nbsp;
  <a href="https://scholar.google.com/citations?user=3PJeg1wAAAAJ&hl=en" target="_blank">Taosha Fan</a> &nbsp;•&nbsp;
  <a href="https://scholar.google.com/citations?user=rebEn8oAAAAJ&hl=en" target="_blank">Luis Pineda</a> &nbsp;•&nbsp;
  <a href="https://scholar.google.com/citations?user=p6DCMrQAAAAJ&hl=en" target="_blank">Mike Lambeta</a> &nbsp;•&nbsp;
  <a href="https://people.eecs.berkeley.edu/~malik/" target="_blank">Jitendra Malik</a> <br/>
  <a href="https://scholar.google.com/citations?user=DMTuJzAAAAAJ&hl=en" target="_blank">Mrinal Kalakrishnan</a> &nbsp;•&nbsp;
  <a href="https://scholar.google.ch/citations?user=fA0rYxMAAAAJ&hl=en" target="_blank">Roberto Calandra</a> &nbsp;•&nbsp;
  <a href="https://www.cs.cmu.edu/~kaess/" target="_blank">Michael Kaess</a> &nbsp;•&nbsp;
  <a href="https://joeaortiz.github.io/" target="_blank">Joseph Ortiz</a> &nbsp;•&nbsp;
  <a href="https://www.mustafamukadam.com/" target="_blank">Mustafa Mukadam</a>
  <br/> <br/>
  <a href="http://www.science.org/doi/10.1126/scirobotics.adl0628">Science Robotics</a>, Nov 2024</a>
</div>

<h4 align="center">
  <a href="https://suddhu.github.io/neural-feels/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/c0/Web.svg" alt="Website" width="10px"/> <b>Website</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="http://www.science.org/doi/10.1126/scirobotics.adl0628"><img src=".github/s.svg" alt="Science Journal" width="8px"/> <b>Paper</b></a> &nbsp;&nbsp;&nbsp; &nbsp;
  <a href="https://youtu.be/KOHh0awhSEg?si=sjSEdC54lKEY3hFy"><img src="https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png" alt="YouTube" width="15px"/> <b>Presentation</b></a> &nbsp;&nbsp;&nbsp; &nbsp;
  🤗 <a href="https://huggingface.co/datasets/suddhu/Feelsight"> <b>Dataset</b></a> + <a href="https://huggingface.co/suddhu/tactile_transformer">Models</a>
</h4>

<div align="center">
<b>TL;DR</b>:  Neural perception with vision and touch yields robust tracking <br/>
and reconstruction of novel objects for in-hand manipulation.
<br> <br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) &nbsp; [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) &nbsp; <a href="https://huggingface.co/collections/suddhu/neuralfeels-673184a97ddcac2df69ff489"><img src="https://img.shields.io/badge/Models%20and%20datasets-Link-yellow?logo=huggingface"></img></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img height="20" src=".github/fair.png" alt="Meta-AI" />  &nbsp;&nbsp; <img height="22" src=".github/cmu.svg" alt="CMU" /> &nbsp;&nbsp; <img height="22" src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Seal_of_University_of_California%2C_Berkeley.svg/600px-Seal_of_University_of_California%2C_Berkeley.svg.png" alt="Berkeley" />  &nbsp;&nbsp; <img height="22" src="https://suddhu.github.io/neural-feels/img/tu_dresden.svg" alt="Dresden" /> &nbsp;&nbsp; <img height="22" src="https://suddhu.github.io/neural-feels/img/ceti.png" alt="ceti" />  
</div>

NeuralFeels combines vision, touch, and robot proprioception into a neural field optimization. Here, we apply it towards in-hand rotation of novel objects.  For details and further results, refer to our <a href="https://suddhu.github.io/neural-feels/">website</a> and <a href="http://www.science.org/doi/10.1126/scirobotics.adl0628"> journal paper</a>. Also see: [FeelSight dataset](./data/README.md) and [Tactile transformer](./neuralfeels/contrib/tactile_transformer/README.md) READMEs. 


<div align="center">
  <img src=".github/preview.gif"
  width="90%">
</div>


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

## Citing NeuralFeels

If you find NeuralFeels useful in your research, please consider citing our paper:

```bibtex
@article{suresh2024neuralfeels,
  title={{N}eural feels with neural fields: {V}isuo-tactile perception for in-hand manipulation},
  author={Suresh, Sudharshan and Qi, Haozhi and Wu, Tingfan and Fan, Taosha and Pineda, Luis and Lambeta, Mike and Malik, Jitendra and Kalakrishnan, Mrinal and Calandra, Roberto and Kaess, Michael and Ortiz, Joseph and Mukadam, Mustafa},
  journal={Science Robotics},
  pages={adl0628},
  year={2024},
  publisher={American Association for the Advancement of Science}
}
```

## License

This project is licensed under [LICENSE](./LICENSE).

## Contributing

We actively welcome your pull requests! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for more info.

## Acknowledgements

- Our neural rendering and Open3D visualizer are based on [iSDF](https://github.com/facebookresearch/iSDF) by [Joe Ortiz](https://joeaortiz.github.io/) and others;
- For in-hand rotation, we train a [HORA](https://github.com/HaozhiQi/hora) policy by [Haozhi Qi](https://haozhi.io/) and others;
- We thank the [DIGIT](https://digit.ml/) team for the vision-based touch sensors, [TACTO](https://github.com/facebookresearch/tacto) for tactile sensor simulation, [Theseus](https://github.com/facebookresearch/theseus) for the PyTorch-friendly optimizer, [DPT](https://github.com/isl-org/DPT) and [FocusOnDepth](https://github.com/antocad/FocusOnDepth) for the transformer architecture, [Segment-anything](https://segment-anything.com/) for prompt-based segmentation, [Helper3D](https://github.com/Jianghanxiao/Helper3D) for URDF-visualization in Open3D; 
- Some of our mesh models are sourced from [YCB](https://www.ycbbenchmarks.com/) and [ContactDB](https://contactdb.cc.gatech.edu/).

We thank Dhruv Batra, Theophile Gervet, Akshara Rai for feedback on the writing, and Wei Dong, Tess Hellebrekers, Carolina Higuera, Patrick Lancaster, Franziska Meier, Alberto Rodriguez, Akash Sharma, Jessica Yin for helpful discussions on the research.
