import numpy as np
import os
import time
import random
from a_dataset.clean_dataset  import OurDataset
from c_utils.b_utils import  time_string, convert_secs2time, AverageMeter
from c_utils.a_functions import cal_anomaly_maps,cal_loss
from b_models_and_sub_models.a_teacher_model import TeacherModel
from b_models_and_sub_models.b_student_model import reverse_student18
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split, Subset, SubsetRandomSampler
import swanlab
from c_utils_for_models import calculScoreAndVisualize


def load_data(validation_ratio=0.3,
              partial_data=True,
              divide_by_groups=True,
              num_of_group=1,
              divide_by_patterns=False,
              num_of_pattern=1,
              phase='Train',
              subset_ratio=None,
              subset_num=None,
              overlap_validation=False):
    '''
    这个函数负责导入数据
    Args:
    '''

    kwargs_train = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    kwargs_test = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}

    if phase=='Train':
        # 加载训练数据集
        train_dataset = OurDataset(phase=phase,
                                   divide_by_groups=divide_by_groups,
                                   num_of_group=num_of_group,
                                   divide_by_patterns=divide_by_patterns,
                                   num_of_pattern=num_of_pattern
                                   )

        if partial_data:
            total_train_size = len(train_dataset)
            if subset_num is not None:
                subset_train_size = min(subset_num, total_train_size)
            elif subset_ratio is not None:
                subset_train_size = int(total_train_size * subset_ratio)
            else:
                subset_train_size = total_train_size

            subset_indices = torch.randperm(total_train_size)[:subset_train_size]
            subset_dataset = Subset(train_dataset, subset_indices)

            if overlap_validation:
                num_valid = int(len(subset_dataset) * validation_ratio)
                train_loader = DataLoader(subset_dataset, batch_size=4, shuffle=True, **kwargs_train)
                val_loader = DataLoader(subset_dataset, batch_size=8, shuffle=False, **kwargs_train)
            else:
                num_total = len(subset_dataset)
                num_valid = int(num_total * validation_ratio)
                num_train = num_total - num_valid
                train_data, val_data = random_split(subset_dataset, [num_train, num_valid])
                train_loader = DataLoader(train_data, batch_size=4, shuffle=True, **kwargs_train)
                val_loader = DataLoader(val_data, batch_size=8, shuffle=False, **kwargs_train)
        else:
            if overlap_validation:
                num_valid = int(len(train_dataset) * validation_ratio)
                all_indices = list(range(len(train_dataset)))
                val_indices = random.sample(all_indices, num_valid)
                val_sampler = SubsetRandomSampler(val_indices)
                train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, **kwargs_train)
                val_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, sampler=val_sampler, **kwargs_train)
            else:
                num_total = len(train_dataset)
                num_valid = int(num_total * validation_ratio)
                num_train = num_total - num_valid
                train_data, val_data = random_split(train_dataset, [num_train, num_valid])
                train_loader = DataLoader(train_data, batch_size=4, shuffle=True, **kwargs_train)
                val_loader = DataLoader(val_data, batch_size=8, shuffle=False, **kwargs_train)

    elif phase=='Test':#test phase
        # 不进行训练，只返回空或 None
        train_loader = None
        val_loader = None
    else:
        raise ValueError("Unexpected phase type")

    test_dataset = OurDataset(phase='Test',
                              divide_by_groups=divide_by_groups,
                              num_of_group=num_of_group,
                              divide_by_patterns=divide_by_patterns,
                              num_of_pattern=num_of_pattern
                              )

    pretest_dataset = OurDataset(phase='PreTest',
                              divide_by_groups=divide_by_groups,
                              num_of_group=num_of_group,
                              divide_by_patterns=divide_by_patterns,
                              num_of_pattern=num_of_pattern
                              )

    pretest_loader=DataLoader(pretest_dataset,batch_size=1,shuffle=False,**kwargs_test)

    if partial_data:
        # 测试集也只使用一部分
        total_test_size = len(test_dataset)
        if subset_num is not None:
            subset_test_num = min(subset_num, total_test_size)
        elif subset_ratio is not None:
            subset_test_num = int(total_test_size * subset_ratio)
        else:
            subset_test_num = total_test_size

        # 随机取 subset_test_num 个样本
        test_indices = torch.randperm(total_test_size)[:subset_test_num]
        test_subset = Subset(test_dataset, test_indices)

        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, **kwargs_test)

    else:

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs_test)

    return train_loader, val_loader, test_loader,pretest_loader


class Trainer:
    """
        一个用于知识蒸馏的反向残差网络训练类。

        Attributes:
            device (str): 运行设备，通常是 'cuda' 或 'cpu'。
            data_path (str): 数据集根路径。
            obj (str): 当前处理的对象类别名称。
            img_resize (int): 图像缩放大小。
            img_cropsize (int): 图像裁剪大小。
            validation_ratio (float): 验证集占训练集的比例。
            num_epochs (int): 训练总轮数。
            lr (float): 学习率。
            batch_size (int): 批次大小。
            patienceES (int): 早停机制的耐心值。
            vis (bool): 是否启用可视化。
            save_path (str): 结果保存路径。
            model_dir (str): 模型文件保存目录。
            img_dir (str): 图像输出保存目录。
            DG (bool): 是否启用域泛化。
            model_t (nn.Module): 教师网络模型。
            model_s (nn.Module): 学生网络模型。
            optimizer (torch.optim.Optimizer): 优化器。
            scheduler (torch.optim.lr_scheduler): 学习率调度器。
            train_loader (DataLoader): 训练数据加载器。
            val_loader (DataLoader): 验证数据加载器。
        """
    def __init__(self,train_loader, val_loader,test_loader,pretest_loader,num_epochs=100,device=None
                 ,DG=False,vis=True,group='carpet1',img_dir="./images",patienceES=10,gpu_id=0,repeat=1):
        self.group=group
        self.gpu_id=gpu_id
        self.repeat=repeat
        if device is None:
            device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.DG=DG
        self.num_epochs=num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.pretest_loader = pretest_loader
        self.lr=0.005
        self.model_t,self.model_s=self.load_model()
        self.optimizer = torch.optim.Adam(self.model_s.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min', factor=0.5, patience=5)
        self.model_dir = os.path.join('./results' , 'models', f'{self.group}_gpu{self.gpu_id}_repeat{self.repeat}')
        self.best_acc = 0.0
        self.patienceES=patienceES
        self.vis=vis
        self.img_dir=os.path.join(img_dir,self.group)
        self.manual_set_thresholds=None

    def load_model(self):

        model_t=TeacherModel(backbone_name="resnet34",out_indices=[0,1,2,3]).to(self.device)
        model_s= reverse_student18(DG=self.DG).to(self.device)

        for param in model_t.parameters():
            param.requires_grad = False
        model_t.eval()
        return model_t,model_s


    def validate(self,epoch):
        self.model_s.eval()
        losses = AverageMeter()
        for (data, _, _) in self.val_loader:
            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                features_t = self.model_t(data)
                features_s = self.model_s(features_t)
                loss = cal_loss(features_s, features_t)
                losses.update(loss.item(), data.size(0))
        self.scheduler.step(losses.avg)
        print(f"this is the {epoch}epoch")
        return losses.avg

    def save_checkpoint(self, is_best=False):
        print('Save model !!!')
        os.makedirs(self.model_dir,exist_ok=True)
        state = {'model':self.model_s.state_dict()}
        torch.save(state, os.path.join(self.model_dir, 'model_s.pth'))

    def train(self):
        self.model_s.train()
        best_score = None
        start_time = time.time()
        epoch_time = AverageMeter()

        compteurEarlyStop = 0
        for epoch in range(1, self.num_epochs + 1):
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * ((self.num_epochs + 1) - epoch))
            need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
            print('{:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, self.num_epochs, time_string(), need_time))
            losses = AverageMeter()
            for (data, label, _) in self.train_loader:
                data = data.to(self.device)

                with torch.set_grad_enabled(True):
                    features_t = self.model_t(data)
                    features_s = self.model_s.forward(features_t)
                    loss = cal_loss(features_s, features_t)
                    losses.update(loss.sum().item(), data.size(0))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            print("\n")
            print('Train Epoch: {} loss: {:.6f}'.format(epoch, losses.avg))
            print("\n")
            print("接下来开始validate")
            val_loss = self.validate(epoch)

            swanlab.log({
                "average_loss_for_epoch":losses.avg,
                "val_loss":val_loss
            },step=epoch)

            if best_score is None or val_loss < best_score:
                best_score = val_loss
                self.save_checkpoint()  # 仅在此处保存
                compteurEarlyStop = 0
                print(f"New best model saved with val_loss={val_loss:.6f}")
            else:
                compteurEarlyStop += 1
                print(f"early stop accumulation:{compteurEarlyStop}")
                if compteurEarlyStop >= self.patienceES:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break
                # ---------------------------------------

            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        print('Training end.')

    def test(self):
        try:
            checkpoint = torch.load(os.path.join(self.model_dir,'model_s.pth'))
        except:
            raise Exception('Check saved model path.')
        self.model_s.load_state_dict(checkpoint['model'])
        self.model_s.eval()
        for param in self.model_s.parameters():
            param.requires_grad = False

        scores = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        print('Testing')

        for (data, label, mask) in self.test_loader:
            test_imgs.extend(data.cpu().numpy())
            gt_list.extend(label.cpu().numpy())
            gt_mask_list.extend(mask.cpu().numpy())
            data = data.to(self.device)
            with torch.set_grad_enabled(False):

                features_t = self.model_t.forward(data)
                features_s = self.model_s.forward(features_t)
                score = cal_anomaly_maps(features_s, features_t, 256)
                scores.append(score)

        scores = np.asarray(scores)
        assert not np.isnan(scores).any(), "Input data contains NaN!"

        assert not np.isinf(scores).any(), "Input data contains Inf!"
        """        
        best_sample_f1_threshold,best_sample_f1=calculate_best_sample_f1_cuda(scores=scores, gt_list=gt_list)
        best_regional_f1_threshold,best_regional_f1=calculate_best_regional_f1(score_map=scores,binary_target=np.array(gt_mask_list))
        best_pixel_f1_threshold,best_pixel_f1=calculate_best_pixel_f1(scores=scores,gt_list=gt_mask_list)
        swanlab.log({"best_sample_f1":best_sample_f1})
        swanlab.log({"best_sample_f1_threshold":best_sample_f1_threshold})
        swanlab.log({"best_regional_f1":best_regional_f1})
        swanlab.log({"best_regional_f1_threshold":best_regional_f1_threshold})
        swanlab.log({"best_pixel_f1":best_pixel_f1})
        swanlab.log({"best_pixel_f1_threshold":best_pixel_f1_threshold})
        """

        img_roc_auc, pixel_roc_auc = calculScoreAndVisualize(self, scores=scores, gt_list=gt_list, gt_mask_list=gt_mask_list, test_imgs=test_imgs)
        return img_roc_auc, pixel_roc_auc

