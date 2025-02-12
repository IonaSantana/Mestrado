#PBS -N my_experiment
#PBS -q testegpu
#PBS -e experiments/logs/my-experiment.train.err
#PBS -o experiments/logs/my-experiment.train.log

#!/bin/bash
# Carrega o módulo Conda fornecido pelo Cenapad
#%Module load miniconda3/22.11.1-gcc-9.4.0
#%Module load cudnn/8.2.0.53-11.3-gcc-9.3.0
# Define o ambiente e o diretório de trabalho
ENV=stgnrde
SCRATCH=/home/lovelace/proj/proj1023/iona
WORK_DIR=$SCRATCH/baseline/STG-NRDE

# Ativa o ambiente Conda
source activate /home/lovelace/proj/proj1023/iona/miniconda3/envs/$ENV
echo "Ambientes disponíveis:"
conda info --envs
# Muda para o diretório de trabalho
cd $WORK_DIR/model
conda list
# Exibe o diretório atual
echo "Diretório atual: $(pwd)"

# Define o PYTHONPATH
export PYTHONPATH=$(pwd)

# Executa o script de treinamento
python3 Run.py --dataset='PEMSD4' --model='GRDE' --model_type='rde2' --embed_dim=10 --hid_dim=64 --hid_hid_dim=64 --num_layers=2 --lr_init=0.001 --weight_decay=1e-3 --epochs=200 --comment="" --input_dim=3 --depth=2 --wnd_len=2 --device=0 --tensorboard

# Desativa o ambiente Conda
# source deactivate
