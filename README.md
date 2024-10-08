# Polar Bear Analysis Pipeline

In this repository we provide the code for our automated stereotypy detection framework.
The corresponding publication is published in the peer-reviewed Journal _Ecological Informatics_: https://doi.org/10.1016/j.ecoinf.2024.102840. Please refer to this publication when using the code.
All necessary models, as well as two demo videos can be downloaded here: https://drive.google.com/drive/folders/1O1wHMxC2E6LGo2rUevRpvsvhi-K36FzF?usp=drive_link.
A demo video explaining our pipeline can be found here: https://youtu.be/HMebXKtyFKI
You can visit our website for further information: https://team-vera.github.io/
Ultimately, we are part of the Machine Learning and Data Analytics Lab: https://www.mad.tf.fau.de

Abstract:

_The welfare of animals under human care is often assessed by observing behaviours indicative of stress or discomfort, such as stereotypical behaviour (SB), which often shows as repetitive, invariant pacing.
Traditional behaviour monitoring methods, however, are labour-intensive and subject to observer bias.
Our study presents an innovative automated approach utilising computer vision and machine learning to non-invasively detect and analyse SB in managed populations, exemplified by a longitudinal study of two polar bears.
We designed an animal tracking framework to localise and identify individual animals in the enclosure.
After determining their position on the enclosure map via homographic transformation, we refined the resulting trajectories using a particle filter.
Finally, we classified the trajectory patterns as SB or normal behaviour using a lightweight random forest approach with an accuracy of 94.9~\%.
The system not only allows for continuous, objective monitoring of animal behaviours but also provides insights into seasonal variations in SB, illustrating its potential for improving animal welfare in zoological settings.
Ultimately, we analysed 607 days for the occurrence of SB, allowing us to discuss seasonal patterns of SB in both the male and female polar bear.
This work advances the field of animal welfare research by introducing a scalable, efficient method for the long-term, automated detection and monitoring of stereotypical behaviour, paving the way for its application across various settings and species that can be continuously monitored with cameras._

![](/Users/eq64opiv/Documents/01_repos/stereotypy-detector/images/Demo_Stereo.gif)

## Setup

1. Pull `yolov5` Repo with `git submodule update --init`
2. Set up a virtual python environment with `python3 -m venv <path_to_env>`
   - Optional: Setup a Python3.9 miniconda env with `conda env create --file conda_env.yaml` and activate it with `conda activate stereotypy-detector` before setting up the virtual environment to ensure that all modules can be installed in their proper version
3. Source environment with `. <path_to_env>/bin/activate`:
   - Optional: If you created a miniconda env deactivate it with `conda deactivate` beforehand
4. Install requirements from [`requirements.txt`](requirements.txt) with `pip install -r requirements.txt`
5. Apply the workaround for [this issue in yolov5](https://github.com/ultralytics/yolov5/issues/6948) by patching `<path_to_env>/lib/python3.9/site-packages/torch/nn/modules/upsampling.py` with `patch -b <path_to_env>/lib/python3.9/site-packages/torch/nn/modules/upsampling.py patches/torch_upsampling.patch` (revert by rerunning this command with `-R`)

## Steps and Components

![pipeline.svg](images/pipeline.svg)

### 1/2 Detection, Identification and Mapping

- What is needed:
  - Trained detector (`models/yolov5m_oc.pt`)
  - Trained identification network (`models/ident.pt`)
  - Download both models here: https://drive.google.com/drive/folders/1O1wHMxC2E6LGo2rUevRpvsvhi-K36FzF?usp=drive_link
  - Homography mapping for enclosure (`tools/mapping/`)
- Script:
  - `analyze.py`
- Output:
  - `traj_0[1,2,3].csv`: Exclusive trajectories, where double occurances of the same identity are ignored
  - `bbox_0[1,2,3].csv`: Bounding boxes
  - `traj_raw_0[1,2,3].json`: Raw trajectories with multiple occurances of the same identity possible. Input for next steps.

Example:

```bash
python3 analyze.py --hm tools/mapping/ --det_weights models/yolov5m_oc.pt --ident_weights models/ident.pt \
  --video ~/Downloads/ch01_20200427082531.mp4 --out /tmp/test --batch 64 --qsize 8 --ident_batch 128 --max_det_t 8 --max_vid_t 4
```

### 3 Filtering

- Script:
  - `trajectories/process_trajectory.py`
- Output:
  - Filtered trajectory with only one position per identity at every time step

Example:

```bash
cd trajectories
python3 process_trajectory.py --traj /tmp/test/traj_raw_*.json --format raw --ex_out /tmp/test/traj_filtered.csv --overlay ../images/PB_Maps_Overlay.png
```

### 4 Behaviour Recognition

- What is needed:
  - Trained behaviour classifier (`models/behaviour_classifier.pkl`)
- Scripts:
  - `trajectories/annotate_behaviour.py`
- Output:
  - Trajectory with added predicted behaviour in the .csv file

Example:

```bash
cd trajectories
python3 annotate_behaviour.py --traj /tmp/test/traj_filtered.csv --out /tmp/test/ --clf ../models/behaviour_classifier.pkl --stereo-only
```


### 5 Visualisation (optional)

- Scripts:
  - `trajectories/plot_trajectory.py`
- Output:
  - Trajectory plots (Enclosure Plot, Observational Plot, Particle-Filtered Plot, ...)

Example:

```bash
cd trajectories
python3 plot_trajectory.py --traj /tmp/test/traj_raw_*.json --overlay ../images/PB_Maps_Overlay.png --out /tmp/test/
```

## Output Format

### 2D Enclosure Positions

- time: Timestamp in seconds since epoch start (January 1, 1970, 00:00:00 (UTC))
- x1, y1: Position of first bear (Nanuq) (-1, if not present)
- x2, y2: Position of second bear (Vera) (-1, if not present)

```csv
time,x1,y1,x2,y2
1629800864.60,0.3,0.4,0.1,0.1
1629800865.71,0.3,0.4,0.1,0.1
1629800865.71,0.3,0.4,-1,-1
...
```

### Bounding Boxes

- time: Timestamp in seconds since epoch start (January 1, 1970, 00:00:00 (UTC))
- bear: ID of bear (0: Nanuq, 1: Vera)
- camera: ID of camera (here: [0,1,2])
- x_c, y_c: Center coordinates of bounding box
- width, height: Dimensions of bounding box
- prob_i, prob_b: Probability for identifiation and bounding box (-1, if not given)

```csv
time,bear,camera,x_c,y_c,width,height,prob_i,prob_b
1629800864.60,0,0,0.3,0.2,0.1,0.15,0.7,0.6
1629800864.60,1,1,0.5,0.5,0.1,0.15,0.99,0.8
1629800865.71,0,2,0.1,0.2,0.1,0.3,0.75,0.9
...
```

### Time Conversion

How to properly handle time of an enclosure position entry:

```python
import time, datetime

# get time t as dummy
>>> t = time.time()
>>> print(t)
1629800864.66491

# convert time t to datetime object dt
>>> dt = datetime.datetime.fromtimestamp(t)
>>> print(dt)
2021-08-24 12:27:44.664910
>>> dt
datetime.datetime(2021, 8, 24, 12, 27, 44, 664910)
```

## Position Mapping

In [utilities/homography.py](utilities/homography.py) one can find the position dependent homography transformation necessary for the conversion of bounding boxes to 2D-enclosure positions.
It is required to provide a directory with the expected format

```txt
transformations
├── mapping_0.png
├── mapping_1.png
├── transformations_0.pt
└── transformations_1.pt
```

`mapping_<n>.png` should contain the mapping of the n-th part of the enclosure as a segmentation map and `transformations_<n>.pt` the corresponding homography matrices as a list of `numpy` arrays.
