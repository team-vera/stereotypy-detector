# Polar Bear Analysis Pipeline

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
  - Trajectory with predicted behaviour as a .csv file

Example:

```bash
cd trajectories
python3 annotate_behaviour.py --traj /tmp/test/traj_filtered.csv --out /tmp/test/ --clf ../models/behaviour_classifier.pkl --stereo-only
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
