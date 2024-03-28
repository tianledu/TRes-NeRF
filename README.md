## install the dependencies
```
python -m venv env
source activate env/bin/activate
bash install.sh
```
## Prepare Data
```
# Step 1: Download data from: https://graphics.tu-bs.de/people-snapshot
# Step 2: Preprocess using our script
python scripts/peoplesnapshot/preprocess_PeopleSnapshot.py --root <PATH_TO_PEOPLESNAPSHOT> --subject male-3-casual

# Step 3: Download SMPL from: https://smpl.is.tue.mpg.de/ and place the model in ./data/SMPLX/smpl/
# └── SMPLX/smpl/
#         ├── SMPL_FEMALE.pkl
#         ├── SMPL_MALE.pkl
#         └── SMPL_NEUTRAL.pkl
```

## Run
```
# run the demo
bash ./bash/run-demo.sh
```
