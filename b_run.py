import swanlab
import argparse
from a_model_train_and_test import load_data,Trainer
import torch
import csv
import os
from datetime import datetime
import yaml
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Random seed fixed to: {seed}")

def get_args():
    parser = argparse.ArgumentParser(description='DBFAD')
    parser.add_argument('--phase', default='train')
    parser.add_argument("--model", type=str, default="ReverseResidual")
    parser.add_argument("--num_of_pattern", type=str, default="7")
    parser.add_argument("--num_of_epochs", type=int, default=150)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--DG", type=bool, default='False')
    parser.add_argument("--experiment_name", type=str, default=None, help="Custom experiment name for SwanLab")
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU ID to use")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat index for multiple runs")
    args = parser.parse_args()
    return args

if __name__=='__main__':
    set_seed(42)

    args=get_args()

    gpu_id = args.gpu_id if args.gpu_id is not None else 0
    device = torch.device(f'cuda:{gpu_id}')

    print(f"Using GPU: {gpu_id}")
    print(f"Pattern: {args.num_of_pattern}")

    if args.experiment_name is None:
        experiment_name = f"DG_off_{args.num_of_epochs}epochs_pattern{args.num_of_pattern}"
    else:
        experiment_name = args.experiment_name

    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        swanlab_project = config.get('swanlab', {}).get('project', 'CLEANZJU_CBAM')
    except:
        swanlab_project = 'CLEANZJU_CBAM'

    swanlab.init(
            project=swanlab_project,
            experiment_name=experiment_name,
            description="Knowledge distillation for anomaly detection with modified model structure"
    )

    if args.phase=="train":
        train_loader, val_loader,test_loader ,pretest_loader = load_data(phase='Train',partial_data=False,divide_by_groups=False,
                                                         num_of_group=False,divide_by_patterns=True,num_of_pattern=args.num_of_pattern,overlap_validation=True)

        trainer = Trainer(train_loader=train_loader, val_loader=val_loader,
                          pretest_loader=pretest_loader,patienceES=args.early_stop_patience
                          ,test_loader=test_loader,num_epochs=args.num_of_epochs,group=args.num_of_pattern,DG=False,gpu_id=gpu_id,device=device,repeat=args.repeat)
        trainer.train()
        img_roc_auc, pixel_roc_auc = trainer.test()

        log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)
        csv_file = os.path.join(log_dir, 'experiment_results.csv')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['数据集编号', '图像级AUCROC', '像素级AUCROC', 'GPU', '完成时间'])
            writer.writerow([
                args.num_of_pattern,
                f"{img_roc_auc:.4f}" if img_roc_auc is not None else "N/A",
                f"{pixel_roc_auc:.4f}" if pixel_roc_auc is not None else "N/A",
                gpu_id,
                timestamp
            ])
        print(f"结果已保存到: {csv_file}")
    elif args.phase=='test':
        train_loader, val_loader,test_loader,pretest_loader = load_data(phase='Test',partial_data=False,divide_by_groups=False,
                                                         num_of_group=False,divide_by_patterns=True,num_of_pattern=args.num_of_pattern,subset_ratio=0.01)

        trainer = Trainer(train_loader=train_loader, val_loader=val_loader,pretest_loader=pretest_loader, patienceES=args.early_stop_patience
                          , test_loader=test_loader, num_epochs=args.num_of_epochs, group=args.num_of_pattern,DG=False,gpu_id=gpu_id,device=device,repeat=args.repeat)
        trainer.test()

    swanlab.finish()