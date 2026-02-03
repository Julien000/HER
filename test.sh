
cd src/

datasets=(TMN2P TNP2M TPM2N PNM2T)
# datasets=(TMN2P)


ckpts=(
    "/lwb/MFND/EventDomain/results/MLP_wOtho/uni_MLP_wOtho_TMN2P_train&test/last-v4.ckpt"
    "/lwb/MFND/EventDomain/results/MLP_wOtho/uni_MLP_wOtho_TNP2M_train&test/last-v1.ckpt"
    "/lwb/MFND/EventDomain/results/MLP_wOtho/uni_MLP_wOtho_TPM2N_train&test/last.ckpt"
    "/lwb/MFND/EventDomain/results/MLP_wOtho/uni_MLP_wOtho_PNM2T_train&test/last.ckpt"
)

model=MLP_wOtho

for idx in "${!datasets[@]}"; do
    dataset="${datasets[$idx]}"
    ckpt=${ckpts[$idx]}

    python main.py \
        --model $model \
        --dataset $dataset \
        --ckpt $ckpt \
        --batch_size 128 \
        --lr 2e-5 \
        --phase test \

done

