import json
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from datetime import datetime
import random
import os
from utils import *

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H:%M:%S")

log_path = "./log/log_file_"+now.strftime("%Y_%m_%d")
if not os.path.exists(log_path):
    os.makedirs(log_path)

logging.basicConfig(level=logging.INFO,
                    filename=f'{log_path}/log_{dt_string}.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_weights(model):
	for m in model.modules():
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				torch.nn.init.zeros_(m.bias.data)
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1) 		 
			m.bias.data.zeros_()


def merge_response_by_asnwer(all_hs, all_answer):
    all_scores_merge_by_answer = []
    for i in range(len(all_answer)):
        d = {}
        answers = all_answer[i]
        scores = all_hs[i]
        for j in range(len(answers)):
            answer_tmp = answers[j] 
            if answer_tmp not in d:
                d[answer_tmp] = [scores[j]]
            else:
                d[answer_tmp].append(scores[j])

        all_scores_merge_by_answer.append(d)

    return all_scores_merge_by_answer 


def get_all_scores_merge_answer_more_than_2(all_scores_merge_by_answer):
    return list(map(lambda d:{key: value for key, value in d.items() if len(value) > 1}, all_scores_merge_by_answer))


def sample_loss1_data(all_hs_merge_by_answer):
    loss1_sample = []
    for ans2hs in all_hs_merge_by_answer:
        for _,v in ans2hs.items():
            loss1_sample.append(random.choice(v))

    loss1_sample = torch.stack(loss1_sample)

    return loss1_sample


def sample_loss2_data(all_hs_merge_by_answer):
    response0_pos_neg_list = [] 
    response1_pos_neg_list = [] 
    for ans2hs in all_hs_merge_by_answer:
        for _,v in ans2hs.items():
            if len(v) > 1:
                response0_pos_neg, response1_pos_neg = random.sample(v, 2)
                response0_pos_neg_list.append(response0_pos_neg)
                response1_pos_neg_list.append(response1_pos_neg)

    if len(response0_pos_neg_list) > 0:
        response0_pos_neg_list = torch.stack(response0_pos_neg_list) 
        response1_pos_neg_list = torch.stack(response1_pos_neg_list) 

    return response0_pos_neg_list, response1_pos_neg_list


def sample_loss1_and_loss3_data_sum(all_hs_merge_by_answer:dict):
    loss1_sample = []
    all_pos_data_group = {} 

    for ans2hs in all_hs_merge_by_answer:
        all_pos_data_tmp = [] 
        for _,v in ans2hs.items():
            tmp_data = random.choice(v)
            loss1_sample.append(tmp_data)
            all_pos_data_tmp.append(tmp_data[0, :])

        answer_num = len(all_pos_data_tmp) 
        if answer_num not in all_pos_data_group:
            all_pos_data_group[answer_num] = [torch.stack(all_pos_data_tmp)]
        else:
            all_pos_data_group[answer_num].append(torch.stack(all_pos_data_tmp))

    loss1_sample = torch.stack(loss1_sample)
    loss3_sample = []
    for _,v in all_pos_data_group.items():
        if len(v) > 1:
            loss3_sample.append(torch.stack(v).squeeze(0))
        else:
            loss3_sample.append(torch.stack(v))

    return loss1_sample, loss3_sample


def sample_loss1_and_loss3_data_logic(all_hs_merge_by_answer:dict):
    loss1_sample = []
    all_one_pos_other_neg_data = {}
    for ans2hs in all_hs_merge_by_answer:
        one_pos_other_neg_data_tmp = [] 
        for _,v in ans2hs.items():
            tmp_data = random.choice(v)
            loss1_sample.append(tmp_data)
            one_pos_other_neg_data_tmp.append(random.choice(v))
        
        all_tensors = torch.stack(one_pos_other_neg_data_tmp) 
        first_rows = all_tensors[:, 0, :]  
        second_rows = all_tensors[:, 1, :] 
        comb_tensor = second_rows.repeat(len(one_pos_other_neg_data_tmp), 1, 1) 
        for i in range(len(one_pos_other_neg_data_tmp)):
            comb_tensor[i, i] = first_rows[i]
        comb_list = [comb_tensor[i] for i in range(len(one_pos_other_neg_data_tmp))]

        answer_num = len(comb_list)         
        if answer_num not in all_one_pos_other_neg_data:
            all_one_pos_other_neg_data[answer_num] = [torch.stack(comb_list)]
        else:
            all_one_pos_other_neg_data[answer_num].append(torch.stack(comb_list))

    loss1_sample = torch.stack(loss1_sample)
    loss3_sample = []
    for _,v in all_one_pos_other_neg_data.items():
        loss3_sample.append(torch.stack(v).squeeze(0))

    return loss1_sample, loss3_sample


def all_sample_data(all_hs_merge_by_answer, all_hs_merge_by_answer_answer_more_than_2, use_loss3_logic):
    response0_pos_neg_list, response1_pos_neg_list = sample_loss2_data(all_hs_merge_by_answer_answer_more_than_2)
    if use_loss3_logic:
        loss1_sample, loss3_sample = sample_loss1_and_loss3_data_logic(all_hs_merge_by_answer)
    else:
        loss1_sample, loss3_sample = sample_loss1_and_loss3_data_sum(all_hs_merge_by_answer)

    return loss1_sample, [response0_pos_neg_list, response1_pos_neg_list], loss3_sample


class LOVER(object):
    def __init__(self
                 , all_train_hs
                 , all_train_answers:dict
                 , all_dev_hs
                 , all_dev_model_answers:dict
                 , all_dev_gt_answers:dict
                 , nepochs=100
                 , ntries=10
                 , lr=1e-5
                 , num_batach=1
                 , verbose=True
                 , device="cuda:0"
                 , linear=True
                 , weight_decay=0.01
                 , save_path=f'./model_save/{dt_string}'
                 , model_layer=2
                 , probe_file=None
                 , task=None
                 , data_layer=-1
                 , use_loss3_logic=False
                 , aggregate='sum'):
        # data
        self.all_train_hs = all_train_hs
        self.all_train_answers = all_train_answers
        self.d = self.all_train_hs.shape[-1]
        self.all_dev_hs = all_dev_hs
        self.all_dev_model_answers = all_dev_model_answers
        self.all_dev_gt_answers = all_dev_gt_answers
        self.data_layer = data_layer

        # training
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.num_batach = num_batach
        self.weight_decay = weight_decay
        self.use_loss3_logic = use_loss3_logic
        
        # probe
        self.linear = linear
        self.model_layer = model_layer
        self.probe_file = probe_file
        if probe_file is not None:
            self.probe = torch.load(probe_file)
            self.best_probe = copy.deepcopy(self.probe)
        else:
            self.probe = None
            self.best_probe = None
        
        self.task = task
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.aggregate = aggregate


    def validate(self, loss1_sample_data_dev, loss2_sample_data_dev, loss3_sample_data_dev, probe):
        p0, p1 = self.get_confidence(self.all_dev_hs, probe)
        acc_max, acc_sum, _ = get_batch_acc(self.task, self.all_dev_gt_answers, self.all_dev_model_answers, p0, p1)

        loss_response_pair = self.loss_answer_pair(len(loss1_sample_data_dev), 0, loss1_sample_data_dev, probe)
        loss_in_group = self.loss_in_group(len(loss2_sample_data_dev[0]), 0, loss2_sample_data_dev, probe)
        loss_cross_group = self.loss_cross_group(len(loss3_sample_data_dev), 0, loss3_sample_data_dev, probe)
        return acc_max, acc_sum, loss_response_pair, loss_in_group, loss_cross_group

    def initialize_probe(self):
        init_probe = None
        if self.linear:
            init_probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            init_probe = MLPProbe_2(self.d)


        initialize_weights(init_probe)
        init_probe.to(self.device)  

        return init_probe

    def get_best_initial_probe(self, try_n, threshold, loss1_sample_data_dev, loss2_sample_data_dev, loss3_sample_data_dev):
        logging.info(f"get_best_initial_probe, try at least {try_n} times, threshold is {threshold}")

        max_dev_acc = 0
        current_n = 0

        best_init_probe = None

        while current_n < try_n or max_dev_acc < threshold:
            init_probe = self.initialize_probe()
            dev_acc_max, _, _, _, _= self.validate(loss1_sample_data_dev, loss2_sample_data_dev, loss3_sample_data_dev, init_probe)
            if dev_acc_max > max_dev_acc:
                max_dev_acc = dev_acc_max
                best_init_probe = init_probe

            current_n += 1

        logging.info(f"try {current_n} times, dev_acc of the best initial probe is {max_dev_acc}") 

        return best_init_probe

    def get_loss_answer_pair(self, p0, p1):
        # loss1
        informative_loss = (torch.min(p0, p1)**2).mean(0)
        consistent_loss = ((p0 - (1-p1))**2).mean(0)
        return informative_loss + consistent_loss

    def get_loss_in_group_mse(self, a_0, a_1, b_0, b_1):
        # loss2 mse
        return ((a_0 - b_0)**2).mean(0) + ((a_1 - b_1)**2).mean(0)
        
    def get_confidence(self, all_hs, probe):
        probe.eval()
        with torch.no_grad():
            p = probe(all_hs)
            p0, p1 = p[:, :, 0, :].cpu(), p[:, :, 1, :].cpu()
        return p0, p1
    
    def loss_answer_pair(self, batch_size, j, loss1_sample, probe):
        # loss1
        loss1_sample = loss1_sample[j*batch_size:(j+1)*batch_size]
        p = probe(loss1_sample)
        p0, p1 = p[:, 0, :], p[:, 1, :]
        loss_response_pair = self.get_loss_answer_pair(p0, p1)

        return loss_response_pair
    
    def loss_in_group(self, batch_size, j, loss2_sample, probe):
        # loss2
        response0_pos_neg_list, response1_pos_neg_list = loss2_sample
        response0_pos_neg_list = response0_pos_neg_list[j*batch_size:(j+1)*batch_size]
        response1_pos_neg_list = response1_pos_neg_list[j*batch_size:(j+1)*batch_size]

        p_r0 = probe(response0_pos_neg_list)
        p_r1 = probe(response1_pos_neg_list)

        p_pos0, p_neg0 = p_r0[:, 0, :], p_r0[:, 1, :]
        p_pos1, p_neg1 = p_r1[:, 0, :], p_r1[:, 1, :]

        loss_in_group = self.get_loss_in_group_mse(p_pos0, p_neg0, p_pos1, p_neg1)

        return loss_in_group

    def get_loss_cross_group_batch_sum(self, batch, probe, inference=False):
        # loss3 sum
        all_loss = []
        if inference:
            probe.eval()

        for one_answer_num_pos_data_tensor in batch:
            all_score = probe(one_answer_num_pos_data_tensor) 

            loss_contradict = (1 - torch.sum(all_score, dim=1))**2

            all_score_smooth = all_score + 0.000000000001
            loss_entropy = -torch.sum(all_score_smooth * torch.log2(all_score_smooth), dim=1) / all_score.shape[1]
            all_loss.append((loss_contradict + loss_entropy).mean(0))

        if len(all_loss) > 0:
            mean_loss = torch.stack(all_loss).mean(0)
        else:
            mean_loss = torch.tensor([0], dtype=torch.float).to(self.device)
        return mean_loss


    def get_loss_cross_group_batch_logic(self, batch, probe, inference=False):
        # loss3 logic
        all_loss = []
        if inference:
            probe.eval()

        for one_answer_num_pos_neg_data_tensor in batch:
            all_score = probe(one_answer_num_pos_neg_data_tensor) 
            all_loss.append((1-torch.max(torch.min(all_score, dim=1).values, dim=1).values + torch.min(torch.min(all_score, dim=1).values, dim=1).values).mean(0))

        mean_loss = torch.stack(all_loss).mean(0)
        return mean_loss
    
    def loss_cross_group(self, batch_size, j, all_answer_list, probe):
        batch = all_answer_list[j*batch_size:(j+1)*batch_size]

        if self.use_loss3_logic:
            loss_cross_group = self.get_loss_cross_group_batch_logic(batch, probe)
        else:
            loss_cross_group = self.get_loss_cross_group_batch_sum(batch, probe)

        return loss_cross_group

    
    def verbose_log(self, *args):
        loss_all, loss_response_pair, loss_in_group, loss_cross_group, dev_acc_max, dev_loss_response_pair, dev_loss_in_group, dev_loss_cross_group, epoch, dev_acc_sum = args
        loss_response_pair = loss_response_pair.item() if loss_response_pair !=0 else 0
        loss_in_group = loss_in_group.item() if loss_in_group !=0 else 0
        loss_cross_group = loss_cross_group.item() if loss_cross_group !=0 else 0
        loss_all = loss_all.item() if loss_all !=0 else 0

        dev_loss_response_pair = dev_loss_response_pair.item() if dev_loss_response_pair !=0 else 0
        dev_loss_in_group = dev_loss_in_group.item() if dev_loss_in_group !=0 else 0
        dev_loss_cross_group = dev_loss_cross_group.item() if dev_loss_cross_group !=0 else 0
        dev_loss_all = dev_loss_response_pair + dev_loss_in_group + dev_loss_cross_group


        logging.info(f'epoch:{epoch+1}, loss:{loss_all:.8f}, dev_acc_max:{dev_acc_max:.4f}, dev_acc_sum:{dev_acc_sum:.4f},train_loss_response_pair:{loss_response_pair:.8f}, train_loss_in_group:{loss_in_group:.8f}, train_loss_cross_group:{loss_cross_group:.8f},dev_loss_response_pair:{dev_loss_response_pair:.8f}, dev_loss_in_group:{dev_loss_in_group:.8f}, dev_loss_cross_group:{dev_loss_cross_group:.8f}')

    def get_loss_sample(self, nbatches, j, loss1_sample, loss2_sample, loss3_sample, probe=None):
        probe = self.probe if probe == None else probe
        loss_answer_pair = self.loss_answer_pair(len(loss1_sample)//nbatches, j, loss1_sample, probe)

        if loss2_sample[0] != []:
            loss_in_group = self.loss_in_group(len(loss2_sample[0])//nbatches, j, loss2_sample, probe)
        else:
            loss_in_group = torch.tensor([0], dtype=torch.float).to(self.device)

        loss_cross_group = self.loss_cross_group(len(loss3_sample)//nbatches, j, loss3_sample, probe)
        loss_list = []

        if loss_cross_group > 0.2:
            loss_all = loss_cross_group
            loss_list = [loss_answer_pair*0, loss_in_group*0, loss_cross_group]
        elif loss_answer_pair > 0.2:
            loss_all = loss_answer_pair + loss_in_group
            loss_list = [loss_answer_pair, loss_in_group, loss_cross_group*0] 
        else:
            loss_all = loss_answer_pair + loss_cross_group + loss_in_group
            loss_list = [loss_answer_pair, loss_in_group, loss_cross_group]

        return loss_all, loss_list, loss_answer_pair, loss_in_group, loss_cross_group


    def train(self, all_hs_merge_by_answer, all_hs_merge_by_answer_answer_more_than_2, loss1_sample_data_dev, loss2_sample_data_dev, loss3_sample_data_dev):
        """
        Does a single training run of nepochs epochs
        """
        
        # Sample data
        loss1_sample_data, loss2_sample_data, loss3_sample_data = all_sample_data(all_hs_merge_by_answer, all_hs_merge_by_answer_answer_more_than_2, self.use_loss3_logic)
        show_data_size(loss1_sample_data, loss2_sample_data, loss3_sample_data)

        optimizer = torch.optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Print loss and other info every verbose_step epochs, and test on dev set, print acc, etc. If acc reaches a new high, save the corresponding model
        verbose_step = 10
        # Resample data every data_sample_step epochs to increase training data diversity
        data_sample_step = 2000
        # Number of batches (not batch size)
        nbatches = self.num_batach
        # Record of highest dev_acc
        dev_acc_max_record = 0

        # Early-stop related configuration below
        # Record the number of times dev_acc is below max_dev_acc; reset to 0 if current epoch breaks the record, otherwise +1
        less_than_max_count = 0
        # If less_than_max_count exceeds less_than_max_threshold, training is not improving, normal training ends (not abnormal)
        less_than_max_threshold = (self.nepochs / 4) / verbose_step
        # Record loss, maintain at most loss_same_threshold latest losses; if exceeded, delete the oldest one
        loss_record_list = []
        # If loss_record_list reaches loss_same_threshold and all losses are the same, training loss is not decreasing, abnormal training, training ends
        loss_same_threshold = 300
        # Lower bound for dev_acc. Note: This value should be adjusted for different datasets, models, encode_format. 0.45 is for mistra-7b-ins-v0.3, instruct, GSM8K.
        dev_acc_lower_bound = 0.65
        dev_acc_lower_bound = 0.45
        dev_acc_lower_bound = 0.1
        # Record the number of times dev_acc is below the lower bound
        dev_acc_lower_than_bound_count = 0
        # If dev_acc_lower_than_bound_count exceeds dev_acc_lower_than_bound_count_threshold, training is going in the wrong direction, abnormal training, training ends
        dev_acc_lower_than_bound_count_threshold = 30
        # Early stop flag due to abnormal training.
        # If early_stop_error is True, this training run is invalid due to abnormal training, and will be automatically retried later
        early_stop_error = False

        for epoch in range(self.nepochs):
            if (epoch+1) % data_sample_step == 0:
                loss1_sample_data, loss2_sample_data, loss3_sample_data = all_sample_data(all_hs_merge_by_answer, all_hs_merge_by_answer_answer_more_than_2, self.use_loss3_logic)

            for j in range(nbatches):
                all_loss, all_loss_list, loss_1, loss_2, loss_3 = self.get_loss_sample(nbatches, j, loss1_sample_data, loss2_sample_data, loss3_sample_data)
                
                optimizer.zero_grad()
                all_loss.backward()

                optimizer.step()
            
            if self.verbose and (epoch==0 or (epoch+1)%verbose_step == 0):
                dev_acc_max, dev_acc_sum, dev_loss_response_pair, dev_loss_in_group, dev_loss_cross_group = self.validate(loss1_sample_data_dev, loss2_sample_data_dev, loss3_sample_data_dev, self.probe)
                self.verbose_log(all_loss, loss_1, loss_2, loss_3, dev_acc_max, dev_loss_response_pair, dev_loss_in_group, dev_loss_cross_group, epoch, dev_acc_sum) 

                if dev_acc_max > dev_acc_max_record:
                    dev_acc_max_record = dev_acc_max
                    best_model_file = f'{self.save_path}/model_max_dev_acc_{dt_string}_data_layer_{self.data_layer}_nepochs_{self.nepochs}_model_layer_{self.model_layer}_lr_{self.lr}_linear_{self.linear}.pt'
                    torch.save(self.probe, best_model_file)
                    self.best_probe = copy.deepcopy(self.probe)
                    less_than_max_count = 0 
                else:
                    less_than_max_count += 1

                if len(loss_record_list) >= loss_same_threshold:
                    del(loss_record_list[0])
                loss_record_list.append(all_loss.item())

                if dev_acc_max < dev_acc_lower_bound:
                    dev_acc_lower_than_bound_count += 1

                if dev_acc_lower_than_bound_count > dev_acc_lower_than_bound_count_threshold \
                    or (len(loss_record_list) == loss_same_threshold and len(list(set(loss_record_list))) == 1):
                    early_stop_error = True
                    break

                if less_than_max_count > less_than_max_threshold:
                    break

        return dev_acc_max_record, early_stop_error
    

    def repeated_train(self):
        logging.info('start repeated trianing ...')
        best_dev_acc = 0

        self.all_train_hs = torch.tensor(self.all_train_hs).to(self.device)
        self.all_dev_hs = torch.tensor(self.all_dev_hs).to(self.device)

        all_hs_merge_by_answer = merge_response_by_asnwer(self.all_train_hs, self.all_train_answers)
        all_hs_merge_by_answer_answer_more_than_2 = get_all_scores_merge_answer_more_than_2(all_hs_merge_by_answer)

        all_dev_hs_merge_by_answer = merge_response_by_asnwer(self.all_dev_hs, self.all_dev_model_answers)
        all_dev_hs_merge_by_answer_answer_more_than_2 = get_all_scores_merge_answer_more_than_2(all_dev_hs_merge_by_answer)
        loss1_sample_data_dev, loss2_sample_data_dev, loss3_sample_data_dev = all_sample_data(all_dev_hs_merge_by_answer, all_dev_hs_merge_by_answer_answer_more_than_2, self.use_loss3_logic)
        
        # Number of normal training runs. When the number of normal training runs reaches self.ntries, training ends. The model with the highest dev_acc is selected as the final model.
        normal_train_count = 0
        # Number of attempts to get the best initial probe.
        ntries_initial_probe = 30
        # Lower bound for initial probe accuracy. When below this bound, the model often gets worse with more training.
        # Note: This value should be adjusted for different datasets, models, and encode_format. 0.5 is for mistra-7b-ins-v0.3, instruct, GSM8K.
        acc_lower_bound_initial_probe = 0.65
        acc_lower_bound_initial_probe = 0.5
        acc_lower_bound_initial_probe = 0.15

        while normal_train_count < self.ntries:
            self.probe = self.get_best_initial_probe(ntries_initial_probe, acc_lower_bound_initial_probe, loss1_sample_data_dev, loss2_sample_data_dev, loss3_sample_data_dev) 
            dev_acc, early_stop = self.train(all_hs_merge_by_answer, all_hs_merge_by_answer_answer_more_than_2, loss1_sample_data_dev, loss2_sample_data_dev, loss3_sample_data_dev)
            if dev_acc > best_dev_acc:
                self.best_probe = copy.deepcopy(self.probe)
                best_dev_acc = dev_acc
            
            if not early_stop:
                normal_train_count += 1
            
            logging.info(f'train count:{normal_train_count}, threshold:{self.ntries}')

        torch.save(self.best_probe, f'{self.save_path}/final_model_{dt_string}_data_layer_{self.data_layer}_nepochs_{self.nepochs}_model_layer_{self.model_layer}_lr_{self.lr}_linear_{self.linear}.pt')
        logging.info('repeated training SUCCESS!')
        return dev_acc
    

def show_data_size(loss1_sample_data, loss2_sample_data, loss3_sample_data):
    logging.info(f'sample data from all data')
    logging.info(f'loss1_sample_data size: {str(list(loss1_sample_data.shape))}')
    logging.info(f'loss2_sample_data size: {str(list(loss2_sample_data[0].shape))}')
    all_loss3_data_num = 0
    for i in loss3_sample_data:
        all_loss3_data_num += i.shape[0]
    logging.info(f'loss3_sample_data size: {all_loss3_data_num}×{loss1_sample_data.shape[-1]}')


def get_loss_specific_probe(loss1_sample_data, loss2_sample_data, loss3_sample_data, probe, model:LOVER, single_prompt_hs_layer):
    _, _, loss_1, loss_2, loss_3 = model.get_loss_sample(1, 0, loss1_sample_data, loss2_sample_data, loss3_sample_data, probe)
    p0, p1 = model.get_confidence(single_prompt_hs_layer, probe)
    confidence = (p0+(1-p1))*0.5

    return confidence, p0, p1, loss_1, loss_2, loss_3


def get_loss_all(model:LOVER, single_prompt_model_response_answers, single_prompt_hs_layer):
    model.best_probe.eval()

    all_hs_merge_by_answer = merge_response_by_asnwer(single_prompt_hs_layer, single_prompt_model_response_answers)
    all_hs_merge_by_answer_answer_more_than_2 = get_all_scores_merge_answer_more_than_2(all_hs_merge_by_answer)
    loss1_sample_data, loss2_sample_data, loss3_sample_data = all_sample_data(all_hs_merge_by_answer, all_hs_merge_by_answer_answer_more_than_2, model.use_loss3_logic)
    best_confidence, best_p0, best_p1, best_loss_1, best_loss_2, best_loss_3 = get_loss_specific_probe(loss1_sample_data, loss2_sample_data, loss3_sample_data, model.best_probe, model, single_prompt_hs_layer)

    best_confidence = best_confidence.squeeze(0)
    best_p0 = best_p0.squeeze(0)
    best_p1 = best_p1.squeeze(0)

    return best_confidence, best_p0, best_p1, best_loss_1, best_loss_2, best_loss_3


def get_batch_acc(task, all_gt_answers, all_model_answers, p0, p1):
    our_correct_max = 0
    our_correct_sum = 0

    p0 = p0.reshape(len(all_gt_answers), -1)
    p1 = p1.reshape(len(all_gt_answers), -1)

    all_scores = (p0+1-p1)*0.5
    indices_max = torch.max(all_scores, dim=-1).indices

    for i, index_max in enumerate(indices_max):
        choose_answer_max = all_model_answers[i][index_max]
        gt_answer = task.extract_gt_answer(all_gt_answers[i])
        if choose_answer_max == gt_answer:
            our_correct_max += 1

     
    for i in range(len(all_model_answers)):
        model_answers_score = all_scores[i].unsqueeze(-1)
        model_answers = all_model_answers[i]
        choose_answer_sum = choose_answer_by_sum(model_answers, model_answers_score)
        gt_answer = task.extract_gt_answer(all_gt_answers[i])
        if choose_answer_sum == gt_answer:
            our_correct_sum += 1

    acc_max = our_correct_max / len(all_gt_answers)
    acc_sum = our_correct_sum / len(all_gt_answers)

    return acc_max, acc_sum, all_scores

def choose_answer(model_answers, confidence, task, aggregate):

    assert aggregate in ['sum', 'max']

    if aggregate == 'sum':
        model_answer = choose_answer_by_sum(model_answers, confidence)
    elif aggregate == 'max':
        model_answer = choose_answer_by_max(model_answers, confidence)

    if ',' in model_answer:
        model_answer = model_answer.replace(',', '')

    if model_answer == task.INVALID_ANS:
        model_answer = '-1000000000'
    
    return model_answer 
    

def choose_answer_by_sum(model_answers, confidence):
    answer2score = {}
    for answer, score in zip(model_answers, confidence):
        if answer not in answer2score:
            answer2score[answer] = [score[0]]
        else:
            answer2score[answer].append(score[0])

    max_dev = 0
    model_answer = None
    for k,v in answer2score.items():
        mean_dev_acc = torch.stack(v).sum()
        if mean_dev_acc > max_dev:
            max_dev = mean_dev_acc
            model_answer = k
    
    return model_answer

def choose_answer_by_max(model_answers, confidence):
    path_id = np.where(confidence == confidence.max())[0][0]
    model_answer = model_answers[path_id]

    return model_answer 

def decode_with_rewad_model(task, args, model:LOVER, all_gt_response_answers, all_model_response_answers, all_hs_layer):
    result = {}

    count = 0
    correct = 0
    best_correct = 0

    correct_max = 0
    best_correct_max = 0

    best_loss1_all = []
    best_loss2_all = []
    best_loss3_all = []

    for i in tqdm(range(len(all_gt_response_answers))):
        model_answers = all_model_response_answers[i]
        single_prompt_hs = torch.tensor(all_hs_layer[i]).to(model.device).unsqueeze(0)
        best_confidence, best_p0, best_p1, best_loss1, best_loss2, best_loss3  = get_loss_all(model, [model_answers], single_prompt_hs) 

        best_loss1_all.append(best_loss1)
        best_loss2_all.append(best_loss2)
        best_loss3_all.append(best_loss3)

        for best_score, answer, best_p0_tmp, best_p1_tmp in zip(best_confidence, model_answers, best_p0, best_p1):
            logging.info(f'answer: {answer}\tbest_score: {best_score.item():.4f}\tbest_socre_right:{best_p0_tmp.item():.4f}\tbest_score_wrong:{best_p1_tmp.item():.4f}')      

        best_model_answer = choose_answer(model_answers, best_confidence, task, 'sum')
        best_model_answer_max = choose_answer(model_answers, best_confidence, task, 'max')
        gt_answer = task.extract_gt_answer(all_gt_response_answers[i])

        try:
            best_correct += (1 if task.correct(gt_answer,best_model_answer) else 0)
            best_correct_max += (1 if task.correct(gt_answer,best_model_answer_max) else 0)
        except Exception as E:
            print(E)
            continue

        count += 1
        logging.info(f'best_loss1: {best_loss1[0].item():.8f}, loss2:{best_loss2.item():.8f}, loss3:{best_loss3.item():.8f}')
        logging.info(f'ALL:{count}, final_model_acc_sum:{100 * correct / count :.2f}, best_mdoel_acc_sum:{100 * best_correct / count :.2f}, gt_answer:{gt_answer}, best_model_answer:{best_model_answer}')
        logging.info(f'ALL:{count}, final_model_acc_max:{100 * correct_max / count :.2f}, best_mdoel_acc_max:{100 * best_correct_max / count :.2f}, gt_answer:{gt_answer}, best_model_answer:{best_model_answer_max}')

    best_loss1_mean = torch.stack(best_loss1_all).mean(0).item()
    best_loss2_mean = torch.stack(best_loss2_all).mean(0).item()
    best_loss3_mean = torch.stack(best_loss3_all).mean(0).item()

    logging.info(f'All mean final losses: loss1:{best_loss1_mean:.8f}, loss2:{best_loss2_mean:.8f}, loss3:{best_loss3_mean:.8f}')
    logging.info(f'Number of test samples: {count}, sum final correct samples: {correct}, sum best correct samples: {best_correct}')
    logging.info(f'Final acc sum of this method: {100 * correct / count :.2f}')
    logging.info(f'Best acc sum of this method: {100 * best_correct / count :.2f}')

    logging.info(f'Number of test samples: {count}, max final correct samples: {correct_max}, max best correct samples: {best_correct_max}')
    logging.info(f'Final acc max of this method: {100 * correct_max / count :.2f}')
    logging.info(f'Best acc max of this method: {100 * best_correct_max / count :.2f}')

    logging.info(f'data_layer:{args.layer}, lr:{args.lr}, epoches:{args.nepochs}')

    result['best_acc'] = best_correct / count
    result['acc_max'] = correct_max / count
    result['best_acc_max'] = best_correct_max / count
    result['data_layer'] = args.layer
    result['lr'] = args.lr
    result['epoches'] = args.nepochs
    result['train_num_examples'] = len(all_model_response_answers)
    result['encode_format'] = args.encode_format
    result['use_loss3_logic'] = args.use_loss3_logic
    result['exp_description"'] = args.exp_description
    result['ntries"'] = args.ntries

    return result


def load_hs_answer_data(all_hs_layer, all_model_answers, all_gt_answers, index_list):
    hs_layer_choose = all_hs_layer[index_list, ...] 

    model_answers_choose = {}
    index = 0
    for k,v in all_model_answers.items():
        if int(k) in index_list:
            model_answers_choose[index] = v
            index += 1

    gt_answers_choose = {}
    index = 0
    for k,v in all_gt_answers.items():
        if int(k) in index_list:
            gt_answers_choose[index] = v
            index += 1

    logging.info(f'load data SUCCESS!')
    logging.info('hs data shape: '+str(list(hs_layer_choose.shape)))
    logging.info('model answer size: ' + str(len(model_answers_choose)))
    logging.info('gt answer size: ' + str(len(gt_answers_choose)))
    logging.info('')

    return hs_layer_choose, model_answers_choose, gt_answers_choose


def load_og_data(hs_file, model_answer_file, gt_answer_file, layer):
    all_hs_layer = np.load(hs_file, mmap_mode='r')[:, :, :, :, layer]
    all_data_size = all_hs_layer.shape[0]
    with open(model_answer_file, 'r') as f:
        all_model_answers = json.load(f)

    with open(gt_answer_file, 'r') as f:
        all_gt_answers = json.load(f)

    logging.info(f'load data from: {hs_file}')
    logging.info(f'load data size: {all_data_size}')
    logging.info(f'load data from layer {layer}')
    
    return all_hs_layer, all_data_size, all_model_answers, all_gt_answers


def load_train_and_dev_data(args):
    '''
    gsm8k+7b all data shape: (7473, 10, 2, 4096, 33)

    data_num: number of examples to use
    layer: which layer of data to select

    '''
    temp_random = random.Random(1029) 
    hs_file = args.hs_train_file_path
    model_answer_file = args.model_answer_train_file_path
    gt_answer_file = args.gt_answer_train_file_path
    layer = args.layer
    data_num = args.train_num_examples

    all_hs_layer, all_data_size, all_model_answers, all_gt_answers = load_og_data(hs_file, model_answer_file, gt_answer_file, layer)

    training_data_size = data_num
    validation_data_size = min(int(data_num / 3), len(all_gt_answers) - data_num)

    assert all_data_size +1 > training_data_size + validation_data_size

    logging.info(f'training data size: {training_data_size}')
    logging.info(f'validation data size: {validation_data_size}')
    logging.info('====='*10)

    random_choice_training_list = sorted(temp_random.sample(range(all_data_size), training_data_size))
    random_choice_validation_list = sorted(temp_random.sample(list(set(list(range(all_data_size)))-set(random_choice_training_list)), validation_data_size))

    all_trainning_hs_data, all_training_model_answers, all_training_gt_answers = load_hs_answer_data(all_hs_layer, all_model_answers, all_gt_answers, random_choice_training_list)
    all_validation_hs_data, all_validation_model_answers, all_validation_gt_answers = load_hs_answer_data(all_hs_layer, all_model_answers, all_gt_answers, random_choice_validation_list)

    return all_trainning_hs_data, all_training_model_answers, all_training_gt_answers, all_validation_hs_data, all_validation_model_answers, all_validation_gt_answers


def load_test_data(args):
    '''
    gsm8k+7b all data shape: (1319, 10, 2, 4096, 33)

    data_num: number of examples to use
    layer: which layer of data to select

    '''
    temp_random = random.Random(1029) 
    hs_file = args.hs_test_file_path
    model_answer_file = args.model_answer_test_file_path
    gt_answer_file = args.gt_answer_test_file_path
    layer = args.layer
    data_num = args.test_num_examples

    all_hs_layer, all_data_size, all_model_answers, all_gt_answers = load_og_data(hs_file, model_answer_file, gt_answer_file, layer)

    random_choice_training_list = sorted(temp_random.sample(range(all_data_size), data_num))
    all_hs_layer_choose, all_model_answers_choose, all_gt_answers_choose = load_hs_answer_data(all_hs_layer, all_model_answers, all_gt_answers, random_choice_training_list)

    return all_hs_layer_choose, all_model_answers_choose, all_gt_answers_choose


def store_result(args, result, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = f'{output_path}/result_{args.exp_description}.txt' 
    with open(output_file, 'a') as f:
        f.write(dt_string + '\t' + '\t'.join([str(i) for i in list(result.values())])+'\n')
