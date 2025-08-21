from __future__ import print_function
import argparse
import time
import os
import datetime
import scipy.io as sio
import numpy
import numpy as np
from utils import utils
from utils.utils import OptimWithSheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models.function import HLoss
from models.function import BetaMixture1D
from models.function import CrossEntropyLoss
from models.basenet import *
import copy
from utils.utils import inverseDecayScheduler, CosineScheduler, StepScheduler, ConstantScheduler
import lmmd
import losses
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import nni
from PIL import Image
import scipy.stats as stats
import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
import numpy as np
from diff import C_net, Diffusion, FE, SS_FE, FE_


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, in_channels)
        self.conv2 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.sigmoid(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)
class GIDDM():
    def __init__(self, args, num_class, src_dset, target_dset,tar_label,test_loader):
        self.model = 'GIDDM'
        self.args = args
        self.all_num_class = num_class
        self.known_num_class = num_class-1
        self.dataset = args.dataset
        self.num_step=10
        self.src_dset = src_dset
        self.target_dset = target_dset
        self.device = self.args.device



        self.build_model_init()
        self.ent_criterion = HLoss()

        # self.bmm_model = self.cont = self.k = 0
        self.bmm_model_maxLoss = torch.log(torch.FloatTensor([self.known_num_class])).to(self.device)
        self.bmm_model_minLoss = torch.FloatTensor([0.0]).to(self.device)
        # self.bmm_update_cnt = 0

        self.src_train_loader = src_dset
        self.target_train_loader= target_dset
        self.target_test_loader = test_loader
        # self.target_train_loader, self.target_val_loader, self.target_test_loader, self.tgt_train_idx = target_dset.get_loaders()
        self.num_batches = min(len(self.src_train_loader), len(self.target_train_loader))
        self.test_dataset=tar_label




    def build_model_init(self):
    #     self.G=FE(
    #     image_size=48,
    #     near_band=1,
    #     num_patches=self.args.patches ** 2,
    #     patch_size=self.args.patches,
    #     num_classes=self.known_num_class,
    #     dim=64,
    #         dim1=self.args.dim1,
    #     pixel_dim=4,
    #     depth=5,
    #     heads=16,
    #     mlp_dim=8,
    #     dropout=0.5,
    #     emb_dropout=0.5,
    #     mode='VIT',
    #     GPU=0,
    #     local_kiner=3
    # )



        self.G=SS_FE( input_dim=48,output_dim=self.args.dim1,hidden_dim1=[self.args.dim2,self.args.dim3],hidden_dim2=[self.args.dim5,self.args.dim6],act= nn.Tanh(),args=self.args)
        self.C = nn.Sequential(
            nn.Linear(in_features=self.args.dim1, out_features=self.args.dim8, bias=True),

            nn.SELU(),

            nn.Linear(in_features=self.args.dim8, out_features=self.all_num_class, bias=True))
        self.E = C_net(self.known_num_class, feature_dim=self.args.dim1, num_step=self.num_step)
        a = torch.zeros(size=(1,)).cuda()
        self.diffusion = Diffusion(noise_steps=self.num_step, device=a.device)
        if self.args.cuda:
            self.G.to(self.args.device)
            self.E.to(self.args.device)
            self.C.to(self.args.device)

        scheduler = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,
                                                                  max_iter=self.args.warmup_iter)

        if 'vgg' == self.args.net:
            for name, param in self.G.named_parameters():
                if 'lower' in name:
                    param.requires_grad = False
                elif 'upper' in name:
                    param.requires_grad = False
            params = list(list(self.G.linear1.parameters()) + list(self.G.linear2.parameters()) + list(
                self.G.bn1.parameters()) + list(self.G.bn2.parameters()))
        else:
            params = list(self.G.parameters())

        self.opt_w_g = OptimWithSheduler(
            optim.Adam(params, lr=self.args.g_lr * self.args.e_lr), scheduler)
        self.opt_w_e = OptimWithSheduler(
            optim.Adam(self.E.parameters(), lr=self.args.e_lr),
            scheduler)
        self.opt_w_c = OptimWithSheduler(
            optim.Adam(self.C.parameters(), lr=self.args.e_lr),
            scheduler)

    def compute_prob_density(self,conf_value, mean, std):
        # 计算给定置信度值的概率密度
        pdf_value = stats.norm.pdf(conf_value, mean, std)
        # 计算该置信度值的累积概率
        cdf_value = stats.norm.cdf(conf_value, mean, std)
        return pdf_value, cdf_value
    def build_model(self):
        def weights_init_bias_zero(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)


        self.DC=nn.Sequential(
            nn.Linear(in_features=self.args.dim1, out_features=5000         , bias=True),

            nn.SELU(),
            nn.Linear(in_features=5000, out_features=2, bias=True))
        self.E=C_net(self.known_num_class,feature_dim=self.args.dim1,num_step=self.num_step)
        self.GCN = GCN(in_channels=self.known_num_class + 1, out_channels=1).cuda()
        a = torch.zeros(size=(1,)).cuda()
        self.diffusion = Diffusion(noise_steps=self.num_step, device=a.device)

        self.DC.apply(weights_init_bias_zero)

        if self.args.cuda:
            self.E.to(self.args.device)
            self.DC.to(self.args.device)

        SCHEDULER = {'cos': CosineScheduler, 'step': StepScheduler, 'id': inverseDecayScheduler, 'constant':ConstantScheduler}
        scheduler = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75,
                                                                            max_iter=self.num_batches*self.args.training_iter)
        scheduler_dc = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75,
                                                                            max_iter=self.num_batches*self.args.training_iter*self.args.update_freq_D)

        if 'vgg' == self.args.net:
            for name,param in self.G.named_parameters():
                if 'lower' in name:
                    param.requires_grad = False
                elif 'upper' in name:
                    param.requires_grad = False
            params = list(list(self.G.linear1.parameters()) + list(self.G.linear2.parameters()) + list(
                self.G.bn1.parameters()) + list(self.G.bn2.parameters()))
        else:
            params = list(self.G.parameters())

        self.opt_g = OptimWithSheduler(
            optim.SGD(params, lr=self.args.g_lr * self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_c = OptimWithSheduler(
            optim.SGD(self.C.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_dc = OptimWithSheduler(
            optim.SGD(self.DC.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler_dc)

        scheduler_e = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,
                                                                     max_iter=self.num_batches*self.args.training_iter)
        self.opt_e = OptimWithSheduler(
            optim.SGD(self.E.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True),
            scheduler_e)

    # def network_initialization(self):
    #     if 'resnet' in self.args.net:
    #         try:
    #             self.E.fc.reset_parameters()
    #             self.E.bottleneck.reset_parameters()
    #         except:
    #             self.E.fc.reset_parameters()
    #     elif 'vgg' in self.args.net:
    #         try:
    #             self.E.fc.reset_parameters()
    #             self.E.bottleneck.reset_parameters()
    #         except:
    #             self.E.fc.reset_parameters()

    def train_init(self):
        print('train_init starts')
        print('...')
        t1 = time.time()
        epoch_cnt =0
        step=0
        best=0
        while step < self.args.warmup_iter + 1:
            self.G.train()
            self.E.train()
            self.C.train()
            epoch_cnt +=1
            for batch_idx, (img_s, label_s) in enumerate(self.src_train_loader):
                img_s_aug= img_s
                img_s_aug=img_s_aug.to(torch.float32)
                if self.args.cuda:
                    # img_s =img_s_aug[0]
                    img_s = Variable(img_s_aug.to(self.args.device))
                    label_s = Variable(label_s.to(self.args.device))

                step += 1
                if step >= self.args.warmup_iter + 1:
                    break

                self.opt_w_g.zero_grad()
                self.opt_w_e.zero_grad()
                self.opt_w_c.zero_grad()
                feat_s = self.G(img_s)
                label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.known_num_class)
                label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                label_s_onehot = label_s_onehot + self.args.ls_eps / (self.known_num_class)
                t = self.diffusion.sample_timesteps(feat_s.shape[0]).cuda()  # 随机拿了一个t
                label_ss, noise = self.diffusion.noise_images(label_s_onehot, t)  # 加噪的代码
                # 10% of the time, don't pass labels, unconditioned diffusion
                predicted_noise = self.E(label_ss, t, feat_s)

                out_s = self.diffusion.sample(self.E, feat_s.shape[0], feat_s, self.known_num_class).reshape(
                    feat_s.shape[0], self.known_num_class)


                label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.known_num_class)
                label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                label_s_onehot = label_s_onehot + self.args.ls_eps / (self.known_num_class)
                loss_s = CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_s,dim=1)) + nn.MSELoss().cuda()(noise,
                                                                                                            predicted_noise)

                out_Cs = self.C(feat_s)
                label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.all_num_class)
                label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                label_s_onehot = label_s_onehot + self.args.ls_eps / (self.all_num_class)
                loss_Cs = CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_Cs, dim=1))

                loss = loss_s + loss_Cs


                loss.backward()
                self.opt_w_g.step()
                self.opt_w_e.step()
                self.opt_w_c.step()
                self.opt_w_g.zero_grad()
                self.opt_w_e.zero_grad()
                self.opt_w_c.zero_grad()
                # print('batch_idx: %s'%batch_idx)
                if step%100==0:
                    self.G.eval()
                    self.E.eval()
                    self.C.eval()

                    C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos, kappa, _, oa,ca = self.test(self.args.training_iter)
                    self.args.logger.info(
                        'Epoch_{:>3}/{:>3}_OS_{:.3f}_OS*_{:.3f}_UNK_{:.3f}_HOS_{:.3f}_OA_{:.3f}_kappa_{:.3f}_Time_{}'.format(
                            self.args.training_iter, self.args.training_iter, C_acc_os * 100, C_acc_os_star * 100,
                                                                              C_acc_unknown * 100, C_acc_hos * 100,
                                                                              oa * 100, kappa * 100,
                            str(datetime.timedelta(seconds=time.time() - t1))[:7]))
                    print("CA:",ca*100)
                    t1 = time.time()


                    nni.report_intermediate_result(C_acc_os*100)
                    if C_acc_os > best:
                        best = C_acc_os
                    print("best:",best)
        # torch.save(self.G.state_dict(),'./chec/HS/G.pkl')
        # torch.save(self.E.state_dict(),'./chec/HS/E.pkl')
        # torch.save(self.C.state_dict(),'./chec/HS/C.pkl')






        duration = str(datetime.timedelta(seconds=time.time() - t1))[:7]
        print('train_init end with duration: %s'%duration)


    def create_graph(self,indices, batch_size):
        edge_index = []
        for i in range(batch_size):
            # 为每个点创建边，使用其K个近邻
            for neighbor in indices[i]:
                edge_index.append([i, neighbor])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index

    def train(self):
        print('Train Starts')
        t1 = time.time()
        # self.G.load_state_dict(torch.load('./chec/HS/G.pkl'))
        # self.E.load_state_dict(torch.load('./chec/HS/E.pkl'))
        # self.C.load_state_dict(torch.load('./chec/HS/C.pkl'))
        best=0
        _, _, _, _, _,banks,_,_= self.test(0)

        for epoch in range(1, self.args.training_iter):
            joint_loader = zip(self.src_train_loader, self.target_train_loader)
            alpha = float((float(2) / (1 + np.exp(-10 * float((float(epoch) / float(self.args.training_iter)))))) - 1)
            for batch_idx, ((img_s, label_s), (img_t, label_t)) in enumerate(joint_loader):
                ######################################################
                #增强原始目标域图像
                # img_t_aug = utils.flip_augmentation(img_t)
                img_t_aug = img_t
                img_t_aug=img_t_aug.to(torch.float32)
                #原始目标域图像
                img_t_og=img_t
                ######################################################
                self.G.train()
                self.C.train()
                self.DC.train()
                self.E.train()
                if self.args.cuda:
                    img_s = Variable(img_s.to(self.args.device))
                    label_s = Variable(label_s.to(self.args.device))
                    img_t = Variable(img_t.to(self.args.device))
                    img_t_og = Variable(img_t_og.to(self.args.device))
                    img_t_aug = Variable(img_t_aug.to(self.args.device))



                out_t_free = self.diffusion.sample(self.E, img_t_og.shape[0], self.G_freezed(img_t_og), self.known_num_class,modal=1).reshape(-1,
                    img_t_og.shape[0], self.known_num_class).detach()
                mean, std = stats.norm.fit(torch.var(out_t_free,0).cpu().detach().numpy())
                _,w_unk_posterior = self.compute_prob_density(torch.var(out_t_free,0).cpu().detach().numpy(), mean, std)
                w_unk_posterior=torch.mean(torch.tensor(w_unk_posterior).cuda(),1)
                w_k_posterior = 1 - w_unk_posterior
                w_k_posterior = w_k_posterior.to(self.args.device)
                w_unk_posterior = w_unk_posterior.to(self.args.device)

                #########################################################################################################
                for l_step in range(self.args.update_freq_D):
                ########################################################################################################

                    self.opt_dc.zero_grad()
                    feat_s = self.G(img_s).detach()
                    out_ds = self.DC(feat_s)









                    label_ds = Variable(torch.zeros(img_s.size()[0], dtype=torch.long).to(self.args.device))
                    label_ds = nn.functional.one_hot(label_ds, num_classes=2)
                    loss_ds = CrossEntropyLoss(label=label_ds, predict_prob=F.softmax(out_ds, dim=1))  # self.criterion(out_ds, label_ds)

                    label_dt = Variable(torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
                    label_dt = nn.functional.one_hot(label_dt, num_classes=2)
                    feat_t = self.G(img_t).detach()

                    out_dt = self.DC(feat_t)

                    loss_dt = CrossEntropyLoss(label=label_dt, predict_prob=F.softmax(out_dt,dim=1))
                    loss_D = 0.5*(loss_ds + loss_dt)
                    loss_D.backward()
                    if self.args.opt_clip >0.0:
                        torch.nn.utils.clip_grad_norm_(self.DC.parameters(), self.args.opt_clip)
                    self.opt_dc.step()
                    self.opt_dc.zero_grad()
                # #########################################################################################################
                for _ in range(self.args.update_freq_G):
                    self.opt_g.zero_grad()
                    self.opt_c.zero_grad()
                    self.opt_e.zero_grad()

                    feat_s = self.G(img_s)
                    out_ds = self.DC(feat_s)
                    loss_ds = CrossEntropyLoss(label=label_ds, predict_prob=F.softmax(out_ds,dim=1))
                    feat_t = self.G(img_t)
                    out_dt = self.DC(feat_t)
                    # label_dt = w_k_posterior[:, None] * label_dt_known - w_unk_posterior[:, None] * label_dt_unknown
                    loss_dt = CrossEntropyLoss(label=label_dt, predict_prob=F.softmax(out_dt,dim=1))
                    loss_G = alpha * (- loss_ds - loss_dt)
                    # LMMD领域适应
                    feat_t_og = self.G(img_t_og)
                    out_ct = self.C(feat_t)
                    probs_w = torch.softmax(out_ct, dim=-1)

                    # pseudo_labels_w, probs_w, _, _ = refine_predictions(feat_t, probs, banks)

                    # train_labels = probs_w[:, :5]
                    # remaining_feat_s, remaining_feat_t, remaining_label_s, tar_labels = Findknownclass(feat_s, feat_t,
                    #                                                                                 label_s, probs_w)
                    # tar_labels = tar_labels[:, :5]
                    dy_iter=20
                    if epoch<=dy_iter:
                        logits_tar_thres=((dy_iter-epoch)/dy_iter)*0.2+0.7
                    else:
                        logits_tar_thres=0.7
                    logits_tar_probs,logits_tar_labels=probs_w.max(-1)
                    high_conf_mask=logits_tar_probs>=logits_tar_thres
                    feat_t_conf=feat_t[high_conf_mask]
                    logits_tar_labels_conf=logits_tar_labels[high_conf_mask]

                    lmmd_=losses.stMMD_loss(class_num=self.known_num_class)

                    loss_supon =lmmd_.get_contrast_loss(torch.concat([feat_s,feat_t_conf],dim=0),torch.concat([label_s,logits_tar_labels_conf],dim=0))
                    # loss_lmmd = lmmd_.get_loss_w(remaining_feat_s,remaining_feat_t, remaining_label_s, tar_labels)
                    loss_lmmd = lmmd_.get_loss_cate(feat_s, feat_t, label_s, probs_w)
                    # unk_loss=lmmd_.push_away_UNK(feat_s, feat_t, label_s, probs_w.max(1)[1])
                    # lmmd_loss = lmmd.LMMD_loss(class_num=self.known_num_class)
                    # loss_lmmd = lmmd_loss.get_loss(remaining_feat_s, remaining_feat_t, remaining_label_s, tar_labels)

                    #########################################################################################################


                    label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.known_num_class)
                    label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                    label_s_onehot = label_s_onehot + self.args.ls_eps / (self.known_num_class)
                    t = self.diffusion.sample_timesteps(feat_s.shape[0]).cuda()  # 随机拿了一个t
                    label_ss, noise = self.diffusion.noise_images(label_s_onehot, t)  # 加噪的代码
                    # 10% of the time, don't pass labels, unconditioned diffusion

                    out_Es = self.diffusion.sample(self.E, feat_s.shape[0], feat_s, self.known_num_class).reshape(
                        feat_s.shape[0], self.known_num_class)
                    predicted_noise = self.E(label_ss, t, feat_s)





                    loss_cls_Es = CrossEntropyLoss(label=label_s_onehot,
                                                   predict_prob=F.softmax(out_Es,dim=1))+ nn.MSELoss().cuda()(noise, predicted_noise)
                    out_Cs = self.C(feat_s)
                    label_Cs_onehot = nn.functional.one_hot(label_s, num_classes=self.all_num_class)
                    label_Cs_onehot = label_Cs_onehot * (1 - self.args.ls_eps)
                    label_Cs_onehot = label_Cs_onehot + self.args.ls_eps / (self.all_num_class)
                    loss_cls_Cs = CrossEntropyLoss(label=label_Cs_onehot, predict_prob=F.softmax(out_Cs, dim=1))

                    label_unknown = (self.known_num_class) * Variable(torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
                    label_unknown = nn.functional.one_hot(label_unknown, num_classes=self.all_num_class)
                    label_unknown_lsr = label_unknown * (1 - self.args.ls_eps)
                    label_unknown_lsr = label_unknown_lsr + self.args.ls_eps / (self.all_num_class)

                    feat_t_aug = self.G(img_t_aug)
                    out_Ct = self.C(feat_t)
                    out_Ct_aug = self.C(feat_t_aug)
                    #*********************************************************
                    #Teaching

                    out_t_f = self.diffusion.sample(self.E, img_t_og.shape[0], self.G_freezed(img_t_og), self.known_num_class).reshape(
                    img_t_og.shape[0], self.known_num_class)#梯度再确定一下到底需要需要去掉
                    ent_Et = self.compute_probabilities_batch(out_t_f, 1)
                    ent_Et=F.sigmoid(ent_Et)

                    teacher_probs=torch.zeros(out_t_f.shape[0],self.all_num_class,dtype=torch.float32)
                    tensor_k=1-ent_Et
                    tensor_k=tensor_k.unsqueeze(1)
                    out_Et_probs = F.softmax(out_t_f)
                    known_probs=out_Et_probs*tensor_k
                    teacher_probs[:,:self.known_num_class]=known_probs
                    teacher_probs[:,self.known_num_class]=ent_Et
                    teacher_probs=teacher_probs.cuda()
                    student_probs=out_Ct_aug.cuda()
                    student_probs=F.softmax(student_probs)
                    knn_S = NearestNeighbors(n_neighbors=11)
                    knn_S.fit(feat_s.cpu().detach().numpy())
                    dist, indices = knn_S.kneighbors(feat_s.cpu().detach().numpy())


                    edge_index = self.create_graph(indices, feat_s.shape[0])
                    W_distance=torch.mean(torch.abs(self.GCN(teacher_probs, edge_index.cuda()).detach()-self.GCN( student_probs, edge_index.cuda())))



                    KL_ts=F.kl_div(F.log_softmax(torch.log(student_probs+(1e-10)),dim=1),teacher_probs,reduction='batchmean')+ 0.1*W_distance


                    #reflective learning
                    # mask_unk=ent_Et>torch.mean(ent_Et)
                    # mask_k = ent_Et <=torch.mean(ent_Et)
                    # unk_samples = feat_t[mask_unk]
                    # k_samples = feat_t[mask_k]
                    # samples_all=torch.cat((unk_samples,k_samples),dim=0)
                    # # 计算未知类别的类别中心
                    # unk_class_center=torch.mean(unk_samples,dim=0)
                    # #计算已知类别的类别中心
                    # k_class_center = torch.mean(k_samples, dim=0)
                    # #计算样本到各自类别中心的距离
                    # distances_unk=torch.norm(unk_samples-unk_class_center,p=2,dim=1)
                    # distances_unk=1-F.sigmoid(distances_unk)
                    # distances_k = torch.norm(k_samples - k_class_center, p=2, dim=1)
                    # distances_k=1-F.sigmoid(distances_k)
                    # distances_all=torch.cat((distances_unk,distances_k),dim=0)
                    #
                    # k_sample_probs=teacher_probs[mask_k]#前面的版本编错了，已经修正
                    # k_sample_probs=torch.sum(k_sample_probs[:,:self.known_num_class],dim=1)
                    # unk_sample_probs=teacher_probs[mask_unk]
                    # unk_sample_probs =unk_sample_probs[:, self.known_num_class]
                    # probs_all=torch.cat((unk_sample_probs,k_sample_probs),dim=0)
                    #
                    # diff=torch.abs(distances_all-probs_all)
                    #
                    # _,indices=torch.topk(diff,max(20-batch_idx,2),largest=True)
                    # fault_samples=samples_all[indices]
                    # #student learning
                    # fault_samples_out=self.C(fault_samples)
                    # fault_samples_out=F.softmax(fault_samples_out)
                    # k_indices=torch.argmax(fault_samples_out,dim=1)
                    # samples_stu_centers = {}
                    # for label in torch.unique(k_indices):
                    #     samples_stu=fault_samples[k_indices==label]
                    #     samples_stu_center = samples_stu.mean(dim=0)
                    #     samples_stu_centers[label.item()] = samples_stu_center
                    # #计算每个样本到其类别中心的距离
                    # distances = torch.zeros_like(k_indices, dtype=torch.float)
                    # for i, label in enumerate(k_indices):
                    #     distances[i] = torch.norm(fault_samples[i]-samples_stu_centers[label.item()])
                    #
                    # L_dis= distances.sum()
                    #*********************************************************
                    # fault_samples_out = self.C(fault_samples)
                    # fault_samples_out = F.softmax(fault_samples_out)
                    # unk_indices = torch.argmax(fault_samples_out, dim=1)
                    # unk_class = unk_indices == self.all_num_class
                    # samples_unk = fault_samples[unk_class]
                    # samples_k = fault_samples[~unk_class]
                    # # 计算未知类别的类别中心
                    # unk_center = torch.mean(samples_unk, dim=0)
                    # # 计算已知类别的类别中心
                    # k_center = torch.mean(samples_k, dim=0)
                    # # 计算样本到各自类别中心的距离
                    # unk_distances = torch.norm(samples_unk - unk_center, p=2, dim=1)
                    # k_distances = torch.norm(samples_k - k_center, p=2, dim=1)
                    # L_dis = sum(torch.cat((unk_distances, k_distances), dim=0))

                    loss_cls_Ctu = alpha*CrossEntropyLoss(label=label_unknown_lsr, predict_prob=F.softmax(out_Ct_aug,dim=1),
                                                    instance_level_weight=w_unk_posterior)

                    pseudo_label = torch.softmax(out_Ct.detach(), dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    targets_u_onehot = nn.functional.one_hot(targets_u, num_classes=self.all_num_class)
                    mask = max_probs.ge(self.args.threshold).float()
                    loss_ent_Ctk = CrossEntropyLoss(label=targets_u_onehot,
                                                    predict_prob=F.softmax(out_Ct_aug,dim=1),
                                                    instance_level_weight=mask)
                    loss =  loss_cls_Es+loss_cls_Cs  +loss_ent_Ctk+ round(self.args.w_ctu,3)*loss_cls_Ctu\
                           +0.5*(loss_G+loss_lmmd*alpha+loss_supon*alpha)+self.args.w_kl*alpha*KL_ts
                    loss.backward()
                    self.opt_g.step()
                    self.opt_c.step()
                    self.opt_e.step()
                    self.opt_g.zero_grad()
                    self.opt_c.zero_grad()
                    self.opt_e.zero_grad()
            # print(
            #     'loss: {:6.4f}, loss_cls_Es:{:6f},loss_cls_Cs:{:6f}, loss_ent_Ctk {:6f}, loss_cls_Ctu: {:6f},loss_lmmd: {:6f},KL_ts: {:6f},L_dis: {:6f}'
            #     .format(loss.item(),loss_cls_Es.item(),loss_cls_Cs.item(),loss_ent_Ctk.item(),loss_cls_Ctu.item(),loss_lmmd.item(),KL_ts.item(),L_dis.item()))
            if (epoch%self.args.update_term==0):
                print("loss_cls_Ctu:",float(loss_cls_Ctu))
                C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos ,kappa,_,oa,ca= self.test(epoch)

                self.args.logger.info(
                    'Epoch_{:>3}/{:>3}_OS_{:.3f}_OS*_{:.3f}_UNK_{:.3f}_HOS_{:.3f}_OA_{:.3f}_kappa_{:.3f}_Time_{}'.format(
                        self.args.training_iter, self.args.training_iter, C_acc_os * 100, C_acc_os_star * 100,
                                                                          C_acc_unknown * 100, C_acc_hos * 100,
                                                                          oa * 100, kappa * 100,
                        str(datetime.timedelta(seconds=time.time() - t1))[:7]))
                t1 = time.time()
                print("CA:", ca * 100)
                aa=float(np.min([ C_acc_os * 100,C_acc_os_star * 100,
                                                                          C_acc_unknown * 100]))

                nni.report_intermediate_result(C_acc_os * 100)
               
                if C_acc_os*100>best :
                    best=C_acc_os* 100
                    # self.test(self.args.training_iter, self.args, draw=True)
                  

        C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos,kappa,_ ,oa,ca= self.test(self.args.training_iter)
        self.args.logger.info(
            'Epoch_{:>3}/{:>3}_OS_{:.3f}_OS*_{:.3f}_UNK_{:.3f}_HOS_{:.3f}_OA_{:.3f}_kappa_{:.3f}_Time_{}'.format(
                self.args.training_iter, self.args.training_iter, C_acc_os*100, C_acc_os_star*100, C_acc_unknown*100, C_acc_hos*100,oa*100,kappa*100,
                str(datetime.timedelta(seconds=time.time() - t1))[:7]))
        print("CA:",ca)

        nni.report_final_result(best)
    def compute_probabilities_batch(self, out_t, unk=1):
        ent_t = self.ent_criterion(out_t)
        batch_ent_t = (ent_t - self.bmm_model_minLoss) / (self.bmm_model_maxLoss - self.bmm_model_minLoss + 1e-6)
        batch_ent_t[batch_ent_t >= 1 - 1e-4] = 1 - 1e-4
        batch_ent_t[batch_ent_t <=  1e-4] = 1e-4
        # B = self.bmm_model.posterior(batch_ent_t.clone().cpu().numpy(), unk)
        # B = torch.FloatTensor(B)
        return batch_ent_t

    def freeze_GE(self):
        self.G_freezed = copy.deepcopy(self.G)
        self.E_freezed = copy.deepcopy(self.E)

    def test(self, epoch,args=None,draw=False):
        known_num_class=self.known_num_class
        self.G.eval()
        self.C.eval()
        self.E.eval()
        total_pred_t = np.array([])
        total_label_t = np.array([])

        all_ent_t = torch.Tensor([])
        probs=[]
        features = []
        m_feature=[]
        pred_label=[]
        indice=[]
        label=[]
        with torch.no_grad():
            for batch_idx, (img_t, label_t) in enumerate(self.target_test_loader):
                if self.args.cuda:
                    img_t, label_t = Variable(img_t.to(self.args.device)), Variable(label_t.to(self.args.device))
                feat_t = self.G(img_t)
                m_fea=self.C(feat_t)
                out_t = F.softmax(self.C(feat_t), dim=1)
                pred = out_t.data.max(1)[1]
                pred_numpy = pred.cpu().numpy()

                total_pred_t = np.append(total_pred_t, pred_numpy)
                total_label_t = np.append(total_label_t, label_t.cpu().numpy())


                # ***********************************************
                features.append(feat_t)
                probs.append(out_t)
                m_feature.append(m_fea)
                pred_label.append(pred)

                label.append(label_t)

        features = torch.cat(features)

        label= torch.cat(label)
        probs=torch.cat(probs)
        m_feature = torch.cat(m_feature)
        m_feature=m_feature.detach().cpu().numpy()
        pred_label = torch.cat(pred_label)
        pred_label=pred_label.squeeze(-1)
        pred_label=pred_label.detach().cpu().numpy()

        rand_idxs = torch.randperm(len(features))
        banks = {
            "features": features[rand_idxs][:3874],# 20736
            "probs": probs[rand_idxs][: 3874],
            "ptr": 0,
        }
        # refine predicted labels
        pred_labels, _, _, _ = refine_predictions(features, probs, banks)
        # **********************************************************
        max_target_label = int(np.max(total_label_t)+1)
        m = utils.extended_confusion_matrix(total_label_t, total_pred_t, true_labels=list(range(max_target_label)), pred_labels=list(range(self.all_num_class)))
        cm = m
        cm = cm.astype(float) / np.sum(cm, axis=1, keepdims=True)
        CA = np.diag(cm) / np.sum(cm, 1, dtype=np.float32)
        oa=len(np.where(total_label_t==total_pred_t)[0])/total_label_t.shape[0]


        acc_os_star = sum([cm[i][i] for i in range(known_num_class)]) / known_num_class



        acc_unknown = sum([cm[i][known_num_class] for i in range(known_num_class, int(np.max(total_label_t)+1))]) / (max_target_label - known_num_class)
        acc_os = (acc_os_star * (known_num_class) + acc_unknown) / (known_num_class+1)
        acc_hos = (2 * acc_os_star * acc_unknown) / (acc_os_star + acc_unknown)

        kappa=metrics.cohen_kappa_score(total_label_t, total_pred_t)

        
        # if acc_os*100>76:
        #
        #     np.save(str('./tsne_data/HS/'+'feature_acc_os:'+str(round(acc_os*100,1))+'acc_uk:'+str(round(acc_unknown*100,1))+'.npy'),features.cpu().detach().numpy())
        #     np.save(
        #             str('./tsne_data/HS/' + 'label_acc_os:' + str(round(acc_os * 100, 1)) + 'acc_uk:' + str(
        #                 round(acc_unknown * 100, 1)) + 'npy'),total_label_t)




        #     # tsen_dir = './tsne_data/' + args.target_name
        #     # if not os.path.exists(tsen_dir):
        #     #     os.makedirs(tsen_dir)
        #     masktar =np.zeros_like(self.test_dataset[3])
        #     RandPerm, Row, Column,G=self.test_dataset
        #     for k in range(len(pred_label)):
        #         masktar[Row[RandPerm[k]], Column[RandPerm[k]]] = pred_label[k] + 1
        #     draw_map(masktar, float(acc_os))



        if epoch%self.args.update_term==0:
            # entropy_list = all_ent_t.data.numpy()
            # loss_tr_t = (entropy_list - self.bmm_model_minLoss.data.cpu().numpy()) / (
            #         self.bmm_model_maxLoss.data.cpu().numpy() - self.bmm_model_minLoss.data.cpu().numpy() + 1e-6)
            # loss_tr_t[loss_tr_t >= 1] = 1 - 10e-4
            # loss_tr_t[loss_tr_t <= 0] = 10e-4
            # self.bmm_model = BetaMixture1D()
            # self.bmm_model.fit(loss_tr_t)
            # self.bmm_model.create_lookup(1)
            # self.bmm_update_cnt += 1
            self.freeze_GE()
            # self.network_initialization()

            return acc_os, acc_os_star, acc_unknown, acc_hos,kappa,banks,oa,CA


def refine_predictions(
        features,
        probs,
        banks):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    pred_labels, probs, pred_labels_all, pred_labels_hard = soft_k_nearest_neighbors(
        features, feature_bank, probs_bank
    )

    return pred_labels, probs, pred_labels_all, pred_labels_hard
    # ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

def get_distances(X, Y, dist_type="cosine"):
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
         distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
         raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances

# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** *

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank):
    pred_probs = []
    pred_probs_all = []

    for feats in features.split(64):
        distances = get_distances(feats, features_bank)
        _, idxs = distances.sort()
        idxs = idxs[:, : 10]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)

    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    # First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard
def findknown(feat_s, feat_t, label_s,probs):
    _,max_indices=torch.max(probs[:,5],dim=0)
    mask=torch.ones_like(probs,dtype=torch.bool)
    mask[max_indices,:]=False
    mask3=torch.ones_like(feat_t,dtype=torch.bool)
    mask3[max_indices, :] = False
#######################################################################
    mask4 = torch.ones_like(feat_s, dtype=torch.bool)
    mask4[max_indices, :] = False
    remaining_feat_s = feat_s[mask3]
    weidu3 = int(len(remaining_feat_s) / 288)
    remaining_feat_s = remaining_feat_s.reshape(weidu3, 288)

    mask2 = torch.ones_like(label_s, dtype=torch.bool)
    mask2[max_indices]=False
    remaining_label_s=label_s[mask2]
    #####################################################################
    remaining_feat_t=feat_t[mask3]
    weidu2 = int(len(remaining_feat_t) / 288)
    remaining_feat_t = remaining_feat_t.reshape(weidu2, 288)


    remaining_samples=probs[mask]
    weidu=int(len(remaining_samples) / 6)
    remaining_samples=remaining_samples.reshape(weidu,6)

    return  remaining_feat_s, remaining_feat_t, remaining_label_s,remaining_samples
def Findknownclass(feat_s, feat_t, label_s,probs):
    unk_indices = torch.argmax(probs, dim=1)
    unk_class = unk_indices == 5
    knownclass = feat_t[~unk_class]
    remaining_probs=probs[~unk_class]
    source_samples=feat_s[~unk_class]
    source_labels=label_s[~unk_class]
    return  source_samples,knownclass,source_labels,remaining_probs
