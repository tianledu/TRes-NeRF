# experiment="baseline"
# # SEQUENCES=("male-4-casual" "male-3-casual" "male-2-casual" "male-2-sport" "male-3-sport" "female-4-casual" "female-3-casual" "female-4-sport")
# SEQUENCES=("female-3-casual" "male-4-casual" "male-3-casual" "female-4-casual" "female-3-casual")
# for SEQUENCE in ${SEQUENCES[@]}; do
#     dataset="peoplesnapshot/$SEQUENCE"
#     python train.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment train.max_epochs=50
#     break
#     python eval.py --config-name SNARF_NGP_refine dataset=$dataset experiment=$experiment
#     # break
# done


experiment="baseline-resfields-0.000001"
# SEQUENCES=("female-4-sport" "male-2-sport" "male-3-sport" "male-2-casual")
SEQUENCES=("female-3-casual" "male-4-casual" "male-3-casual" "female-4-casual")
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="peoplesnapshot/$SEQUENCE"
    python train.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment train.max_epochs=50
    # # break
    python eval.py --config-name SNARF_NGP_refine dataset=$dataset experiment=$experiment
    # # break
    # python animate.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose"
    # python novel_view.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose"
done
# experiment="TRes-NeRF"
# SEQUENCES=("female-3-casual" "male-4-casual" "male-3-casual" "female-4-casual" "female-4-sport" "male-2-sport" "male-3-sport" "male-2-casual")
# for SEQUENCE in ${SEQUENCES[@]}; do
#     dataset="peoplesnapshot/$SEQUENCE"
#     python train.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment train.max_epochs=50
#     python eval.py --config-name SNARF_NGP_refine dataset=$dataset experiment=$experiment
#     python animate.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose"
#     python novel_view.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose"
# done
