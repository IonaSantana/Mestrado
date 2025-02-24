import argparse
from train import pipeline_all_sensors 
def main(base_path, model, conf = None):
    list_models = ['conv1D']
    print("lagrimas")
    # Executar o pipeline
    # file_path = 'Bases/PeMSD4/PEMS04.npz'
    file_path = base_path
    results_file = f'sensor_results_{model}.csv'
    results_file_mean = f'sensor_results_{model}_mean.csv'
    pipeline_all_sensors(file_path, model, results_file, results_file_mean, train_ratio=0.6, val_ratio=0.2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Models in a Baseline')
    parser.add_argument('--base_path', metavar='path', required=True,
                        help='the path to workspace')
    parser.add_argument('--model', metavar='path', required=True,
                        help='path to schema')
    parser.add_argument('--conf', metavar='path', required=True,
                        help='path to dem')
    args = parser.parse_args()
    main(base_path=args.base_path, model=args.model, conf=args.conf)
# python3 main.py --base_path '/home/f-msc2023/ra272498/mestrado/baseline/baseline/Bases/PeMSD4/PEMS04.npz' --model 'None' --conf 'None'
# scp -r '/mnt/c/Users/iona/OneDrive - Minerva S.A/Documentos/mestrado/Mestrado/baseline' ra272498@gpus.lab.ic.unicamp.br:/home/f-msc2023/ra272498/mestrado/baseline/

