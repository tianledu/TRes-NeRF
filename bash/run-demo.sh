experiment="demo"
SEQUENCES=("male-2-casual") # 3
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="peoplesnapshot/$SEQUENCE"
    python train.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment
    python animate.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment
done
