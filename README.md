## Prepare Data
```bash
# Step 1: Download data from: https://graphics.tu-bs.de/people-snapshot
# Step 2: Preprocess using our script
python scripts/peoplesnapshot/preprocess_PeopleSnapshot.py --root <PATH_TO_PEOPLESNAPSHOT> --subject male-3-casual

# Step 3: Download SMPL from: https://smpl.is.tue.mpg.de/ and place the model in ./data/SMPLX/smpl/
# └── SMPLX/smpl/
#         ├── SMPL_FEMALE.pkl
#         ├── SMPL_MALE.pkl
#         └── SMPL_NEUTRAL.pkl
```

## Quick Start
Quickly learn and animate an avatar with `bash ./bash/run-demo.sh`


## Play with Your Own Video
Here we use the in the wild video provided by [Neuman](https://github.com/apple/ml-neuman) as an example:

1. create a yaml file specifying the details about the sequence in `./confs/dataset/`. In this example it's provided in `./confs/dataset/neuman/seattle.yaml`.
2. download the data from [Neuman's Repo](https://github.com/apple/ml-neuman), and run `cp <path-to-neuman-dataset>/seattle/images ./data/custom/seattle/`
3. run the bash script `bash scripts/custom/process-sequence.sh ./data/custom/seattle neutral` to preprocess the images, which
    - uses [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to estimate the 2D keypoints,
    - uses [Segment-Anything](https://github.com/facebookresearch/segment-anything) to segment the scene
    - uses [ROMP](https://github.com/Arthur151/ROMP) to estimate camera and smpl parameters
4. run the bash script `bash ./bash/run-neuman-demo.sh` to learn an avatar

```
