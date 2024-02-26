# 现在保存结果的执行命令
# experiment="TRes-NeRF"
# SEQUENCES=("seattle" "bike" "lab" "citron" "jogging" "parkinglot")
# for SEQUENCE in ${SEQUENCES[@]}; do
    # dataset="neuman/$SEQUENCE"
    # python train.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose" train.max_epochs=100
    # # break
    # python eval.py --config-name SNARF_NGP_refine dataset=$dataset experiment=$experiment
    # # bash scripts/custom/process-sequence.sh ./data/custom/$SEQUENCE neutral
    # # python fit.py --config-name SNARF_NGP_fitting dataset=$dataset experiment=$experiment deformer=smpl train.max_epochs=200
    # # python train.py --config-name demo dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose" train.max_epochs=200 sampler.dilate=8
    # python novel_view.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose"
    # python animate.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose"
# done

# 按照原论文代码的命令，执行报错
# experiment="TRes-NeRF-test"
# SEQUENCES=("seattle" "bike" "lab" "citron" "jogging" "parkinglot")
# for SEQUENCE in ${SEQUENCES[@]}; do
    # dataset="neuman/$SEQUENCE"
    # python train.py --config-name demo dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose" train.max_epochs=100
    # # break
    # python eval.py --config-name SNARF_NGP_refine dataset=$dataset experiment=$experiment
    # # bash scripts/custom/process-sequence.sh ./data/custom/$SEQUENCE neutral
    # # python fit.py --config-name SNARF_NGP_fitting dataset=$dataset experiment=$experiment deformer=smpl train.max_epochs=200
    # # python train.py --config-name demo dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose" train.max_epochs=200 sampler.dilate=8
    # python novel_view.py --config-name demo dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose"
    # python animate.py --config-name demo dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose"
# done


# baseline下成功执行的代码，但是在这里执行报错
experiment="TRes-NeRF-0.01-10"
SEQUENCES=("seattle" "bike" "lab" "citron" "jogging" "parkinglot")
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="neuman/$SEQUENCE"
    python train.py --config-name demo dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose" train.max_epochs=100
    # break
    python eval.py --config-name SNARF_NGP_refine dataset=$dataset experiment=$experiment
    # bash scripts/custom/process-sequence.sh ./data/custom/$SEQUENCE neutral
    # python fit.py --config-name SNARF_NGP_fitting dataset=$dataset experiment=$experiment deformer=smpl train.max_epochs=200
    # python train.py --config-name demo dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose" train.max_epochs=200 sampler.dilate=8
    # python novel_view.py --config-name demo dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose"
    # python animate.py --config-name demo dataset=$dataset experiment=$experiment deformer.opt.cano_pose="a_pose"
done
