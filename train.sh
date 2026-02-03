export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0


cd src/

datasets=(TNM2P TNP2M TPM2N PNM2T)
# datasets=(TPMN2N)

#  // Dear, MLP, ESM, raw_DEAR, MLP_wAL, MLP_wOtho, MLP_wOtho_wContrastive
# model=DAMMFNDMODEL  # MLP, ESM, Dear, SAFE , MCANV2
# model=MDFEND  #MMDFND MLP, ESM, Dear, SAFE , MCANV2
model=MMDFND  #MMDFND MLP, ESM, Dear, SAFE , MCANV2
epochs=2


# for dataset in ${datasets[@]}; do
#     python main.py \
#         --model $model \
#         --dataset $dataset \
#         --batch_size 32 \
#         --lr 2e-5 \
#         --dropout 0.0 \
#         --max_epoch $epochs \

# done

# datasets=(TNMP2P TNPM2M TPMN2N PNMT2T)
datasets=(TNPM2M)

for dataset in ${datasets[@]}; do
    python main.py \
        --model $model \
        --dataset $dataset \
        --batch_size 32 \
        --lr 2e-5 \
        --dropout 0.0 \
        --max_epoch $epochs \
        
done