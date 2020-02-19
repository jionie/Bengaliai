# import os and define graphic card
import os
os.environ["OMP_NUM_THREADS"] = "1"

# import common libraries
import gc
import random
import argparse
import pandas as pd
import numpy as np
from functools import partial

# import pytorch related libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter


# import apex for mix precision training
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from apex.optimizers import FusedAdam

# import dataset class
from dataset.dataset import *

# import utils
from utils.ranger import *
from utils.lrs_scheduler import * 
from utils.transformers_lr_scheduler import * 
from utils.loss_function import *
from utils.metric import *
from utils.file import *

# import model
from model.model import *



############################################################################## Define Argument
parser = argparse.ArgumentParser(description="arg parser")

parser.add_argument('--data_path', type=str, \
    default="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19", \
    required=False, help='specify the path of train dataset')
parser.add_argument('--df_path', type=str, \
    default="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/train.csv", \
    required=False, help='specify the df path of train dataset')
parser.add_argument('--test_data_path', type=str, \
    default="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19", \
    required=False, help='specify the path data of test dataset')
parser.add_argument('--test_df_path', type=str, \
    default="/media/jionie/my_disk/Kaggle/Bengaliai/input/bengaliai-cv19/test.csv", \
    required=False, help='specify the df path of test dataset')
parser.add_argument('--model_type', type=str, default="seresnext50", \
    required=False, help='specify the model_type for BengaliaiNet')
parser.add_argument('--loss', type=str, default="bce", required=True, help="specify the loss for training")
parser.add_argument('--split', type=str, default="MultilabelStratifiedKFold", required=False, help="specify the splitting dataset way")
parser.add_argument('--n_splits', type=int, default=5, \
    required=False, help='specify the number of folds')
parser.add_argument('--seed', type=int, default=42, \
    required=False, help='specify the random seed for splitting dataset')
parser.add_argument('--save_path', type=str, default="/media/jionie/my_disk/Kaggle/Bengaliai/input/", \
    required=False, help='specify the path for saving splitted csv')
parser.add_argument('--fold', type=int, default=0, required=False, help="specify the fold for training")
parser.add_argument('--optimizer', type=str, default='AdamW', required=False, help='specify the optimizer')
parser.add_argument("--lr_scheduler", type=str, default='WarmupCosineAnealingWithHardRestart', required=False, help="specify the lr scheduler")
parser.add_argument("--warmup_proportion",  type=float, default=0.05, required=False, \
    help="Proportion of training to perform linear learning rate warmup for. " "E.g., 0.1 = 10%% of training.")
parser.add_argument("--lr", type=float, default=4e-3, required=False, help="specify the initial learning rate for training")
parser.add_argument("--num_epoch", type=int, default=20, required=False, help="specify the total epoch")
parser.add_argument('--batch_size', type=int, default=4, \
    required=False, help='specify the batch_size for testing dataloader')
parser.add_argument("--valid_batch_size", type=int, default=32, required=False, help="specify the batch size for validating")
parser.add_argument("--accumulation_steps", type=int, default=4, required=False, help="specify the accumulation steps")
parser.add_argument('--num_workers', type=int, default=4, \
    required=False, help='specify the num_workers for testing dataloader')
parser.add_argument("--start_epoch", type=int, default=0, required=False, help="specify the start epoch for continue training")
parser.add_argument("--checkpoint_folder", type=str, default="/media/jionie/my_disk/Kaggle/Bengaliai/model", \
    required=False, help="specify the folder for checkpoint")
parser.add_argument('--load_pretrain', action='store_true', default=False, help='whether to load pretrain model')
parser.add_argument('--early_stopping', type=int, default=3, required=False, help="specify how many epochs for early stopping doesn't increase")



############################################################################## Define Constant
NUM_CLASS=1
WEIGHT_DECAY = 0.01
NUM_CLASSES = [168, 11, 7, 1295]

############################################################################## seed all
SEED = 42
base_dir = '../input/'
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)



############################################################################## define function for training
def training(
            n_splits,
            fold,
            train_data_loader, 
            val_data_loader,
            model_type,
            optimizer_name,
            lr_scheduler_name,
            lr,
            warmup_proportion,
            batch_size,
            valid_batch_size,
            num_epoch,
            start_epoch,
            accumulation_steps,
            checkpoint_folder,
            load_pretrain,
            seed,
            loss_type,
            early_stopping
            ):
    
    torch.cuda.empty_cache()

    COMMON_STRING ='@%s:  \n' % os.path.basename(__file__)
    COMMON_STRING += '\tset random seed\n'
    COMMON_STRING += '\t\tseed = %d\n'%seed

    COMMON_STRING += '\tset cuda environment\n'
    COMMON_STRING += '\t\ttorch.__version__              = %s\n'%torch.__version__
    COMMON_STRING += '\t\ttorch.version.cuda             = %s\n'%torch.version.cuda
    COMMON_STRING += '\t\ttorch.backends.cudnn.version() = %s\n'%torch.backends.cudnn.version()
    try:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = %s\n'%os.environ['CUDA_VISIBLE_DEVICES']
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = None\n'
        NUM_CUDA_DEVICES = 1

    COMMON_STRING += '\t\ttorch.cuda.device_count()      = %d\n'%torch.cuda.device_count()
    COMMON_STRING += '\n'
    
    checkpoint_folder = os.path.join(checkpoint_folder, model_type + '/' + loss_type + '-' + \
        optimizer_name + '-' + lr_scheduler_name)
    os.makedirs(checkpoint_folder, exist_ok=True)
        
    
        
    checkpoint_filename = 'fold_' + str(fold) + "_checkpoint.pth"
    checkpoint_filepath = os.path.join(checkpoint_folder, checkpoint_filename)
    
    log = Logger()
    log.open(os.path.join(checkpoint_folder, 'fold_' + str(fold) + '_log_train.txt'), mode='a+')
    
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tseed         = %u\n' % seed)
    log.write('\tFOLD         = %s\n' % fold)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % checkpoint_folder)
    log.write('\n')


    ############################################################################## define unet model with backbone
    def load(model, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = model.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip): continue
            try:
                state_dict[key] = pretrain_state_dict[key]
            except:
                print(key)
        model.load_state_dict(state_dict, strict=False)
        
        return model


    ############################################################################### model
    model = BengaliaiNet(model_type=model_type, \
            n_classes=NUM_CLASSES)
    
    model = model.cuda()
    
    if load_pretrain:
        model = load(model, checkpoint_filepath)

    ############################################################################### optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], \
            'lr': lr, \
            'weight_decay': WEIGHT_DECAY}, \
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], \
            'lr': lr, \
            'weight_decay': 0.0}
    ]

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters)
    elif optimizer_name == "Ranger":
        optimizer = Ranger(optimizer_grouped_parameters)
    elif optimizer_name == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, eps=4e-5)
    elif optimizer_name == "FusedAdam":
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              bias_correction=False)
    else:
        raise NotImplementedError
    
    ############################################################################### lr_scheduler   
    if lr_scheduler_name == "WarmupCosineAnealing":
        num_train_optimization_steps = num_epoch * len(train_data_loader) // accumulation_steps
        scheduler = get_cosine_schedule_with_warmup(optimizer, \
            num_warmup_steps=int(num_train_optimization_steps*warmup_proportion), \
            num_training_steps=num_train_optimization_steps)
        lr_scheduler_each_iter = True
    elif lr_scheduler_name == "WarmupLinearSchedule":
        num_train_optimization_steps = num_epoch * len(train_data_loader) // accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, \
                                        num_warmup_steps=int(num_train_optimization_steps*warmup_proportion), \
                                        num_training_steps=num_train_optimization_steps)
        lr_scheduler_each_iter = True
    elif lr_scheduler_name == "WarmupCosineAnealingWithHardRestart":
        num_train_optimization_steps = num_epoch * len(train_data_loader) // accumulation_steps
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, \
                                        num_warmup_steps=int(num_train_optimization_steps*warmup_proportion), \
                                        num_training_steps=num_train_optimization_steps)
        lr_scheduler_each_iter = True
    else:
        raise NotImplementedError

    log.write('net\n  %s\n'%(model_type))
    log.write('optimizer\n  %s\n'%(optimizer_name))
    log.write('schduler\n  %s\n'%(lr_scheduler_name))
    log.write('\n')

    ###############################################################################  mix precision
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    # model = nn.DataParallel(model)

    ############################################################################### eval setting
    eval_step = len(train_data_loader) # or len(train_data_loader) 
    log_step = int(len(train_data_loader) / 20)
    eval_count = 0
    count = 0

    ############################################################################### training
    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  accumulation_steps=%d\n'%(batch_size, accumulation_steps))
    log.write('   experiment  = %s\n' % str(__file__.split('/')[-2:]))
    
    valid_loss = np.zeros(1, np.float32)
    train_loss = np.zeros(1, np.float32)
    valid_metric_optimal = -np.inf
    
    # define tensorboard writer and timer
    writer = SummaryWriter()
    
    # define criterion
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'mae':
        criterion = nn.SmoothL1Loss()
    elif loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == 'focal':
        criterion = FocalLoss()
    else:
        raise NotImplementedError
    
    for epoch in range(1, num_epoch+1):

        # init in-epoch statistics
        grapheme_root_recall_train = []
        vowel_diacritic_recall_train = []
        consonant_diacritic_recall_train = []
        grapheme_recall_train = []
        average_recall_train = []
        
        grapheme_root_recall_val = []
        vowel_diacritic_recall_val = []
        consonant_diacritic_recall_val = []
        grapheme_recall_val = []
        average_recall_val = []
        
        # update lr and start from start_epoch  
        if ((epoch > 1) and (not lr_scheduler_each_iter)):
            scheduler.step()
           
        if (epoch < start_epoch):
            if (lr_scheduler_each_iter):
                scheduler.step(len(train_data_loader))
            continue
        
        log.write("Epoch%s\n" % epoch)
        log.write('\n')

        sum_train_loss = np.zeros_like(train_loss)
        sum_train = np.zeros_like(train_loss)

        # init optimizer
        torch.cuda.empty_cache()
        model.zero_grad()
        
        
        for tr_batch_i, (image, grapheme_root, vowel_diacritic, consonant_diacritic, grapheme) in enumerate(train_data_loader):

            rate = 0
            for param_group in optimizer.param_groups:
                rate += param_group['lr'] / len(optimizer.param_groups)
                
            # set model training mode
            model.train() 

            # set input to cuda mode
            image = image.cuda()
            grapheme_root    = grapheme_root.cuda().float()
            vowel_diacritic    = vowel_diacritic.cuda().float()
            consonant_diacritic    = consonant_diacritic.cuda().float()
            grapheme    = grapheme.cuda().float()
            
            if ((loss_type == 'mae') or (loss_type == 'mse')):
                
                _, grapheme_root  = torch.max(grapheme_root, 1)
                grapheme_root = torch.squeeze(grapheme_root.float())
                
                _, vowel_diacritic  = torch.max(vowel_diacritic, 1)
                vowel_diacritic = torch.squeeze(vowel_diacritic.float())
                
                _, consonant_diacritic  = torch.max(consonant_diacritic, 1)
                consonant_diacritic = torch.squeeze(consonant_diacritic.float())
                
                _, grapheme  = torch.max(grapheme, 1)
                grapheme = torch.squeeze(grapheme.float())
                
                

            # predict and calculate loss (only need torch.sigmoid when inference)
            predictions = model(image)  
            
            grapheme_root_prediction = torch.squeeze(predictions[0])
            vowel_diacritic_prediction = torch.squeeze(predictions[1])
            consonant_diacritic_prediction = torch.squeeze(predictions[2])
            grapheme_prediction = torch.squeeze(predictions[3])
            
        
            loss = criterion(grapheme_root_prediction, grapheme_root) + \
                criterion(vowel_diacritic_prediction, vowel_diacritic) + \
                criterion(consonant_diacritic_prediction, consonant_diacritic) + \
                0.1 * criterion(grapheme_prediction, grapheme) 
            
            # use apex
            with amp.scale_loss(loss/accumulation_steps, optimizer) as scaled_loss:
                scaled_loss.backward()

            # don't use apex
            #loss.backward()
        
            if ((tr_batch_i+1) % accumulation_steps == 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
                optimizer.step()
                model.zero_grad()
                # adjust lr
                if (lr_scheduler_each_iter):
                    scheduler.step()

                writer.add_scalar('train_loss_' + str(fold), loss.item(), (epoch-1)*len(train_data_loader)*batch_size+tr_batch_i*batch_size)
            
            # calculate statistics
            if ((loss_type == "bce") or (loss_type == "focal")):
                # prediction = torch.sigmoid(prediction)
                grapheme_root  = torch.argmax(grapheme_root, 1)
                grapheme_root = torch.squeeze(grapheme_root.float())
                
                vowel_diacritic  = torch.argmax(vowel_diacritic, 1)
                vowel_diacritic = torch.squeeze(vowel_diacritic.float())
                
                consonant_diacritic  = torch.argmax(consonant_diacritic, 1)
                consonant_diacritic = torch.squeeze(consonant_diacritic.float())
                
                grapheme  = torch.argmax(grapheme, 1)
                grapheme = torch.squeeze(grapheme.float())
                
                grapheme_root_prediction  = torch.argmax(grapheme_root_prediction, 1)
                grapheme_root_prediction = torch.squeeze(grapheme_root_prediction.float())
                
                vowel_diacritic_prediction  = torch.argmax(vowel_diacritic_prediction, 1)
                vowel_diacritic_prediction = torch.squeeze(vowel_diacritic_prediction.float())
                
                consonant_diacritic_prediction  = torch.argmax(consonant_diacritic_prediction, 1)
                consonant_diacritic_prediction = torch.squeeze(consonant_diacritic_prediction.float())
                
                grapheme_prediction  = torch.argmax(grapheme_prediction, 1)
                grapheme_prediction = torch.squeeze(grapheme_prediction.float())
                

            
            grapheme_root_train = grapheme_root.cpu().detach().numpy()
            vowel_diacritic_train = vowel_diacritic.cpu().detach().numpy()
            consonant_diacritic_train = consonant_diacritic.cpu().detach().numpy()
            grapheme_train = grapheme.cpu().detach().numpy()
            
            grapheme_root_prediction_train = grapheme_root_prediction.cpu().detach().numpy()
            vowel_diacritic_prediction_train = vowel_diacritic_prediction.cpu().detach().numpy()
            consonant_diacritic_prediction_train = consonant_diacritic_prediction.cpu().detach().numpy()
            grapheme_prediction_train = grapheme_prediction.cpu().detach().numpy()
            
            
            l = np.array([loss.item() * batch_size])
            n = np.array([batch_size])
            sum_train_loss = sum_train_loss + l
            sum_train      = sum_train + n
            
            # log for training
            if (tr_batch_i+1) % log_step == 0:  
                train_loss          = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train[...]      = 0
                
                grapheme_root_recall_train.append(metric(grapheme_root_prediction_train, grapheme_root_train))
                vowel_diacritic_recall_train.append(metric(vowel_diacritic_prediction_train, vowel_diacritic_train))
                consonant_diacritic_recall_train.append(metric(consonant_diacritic_prediction_train, consonant_diacritic_train))
                grapheme_recall_train.append(metric(grapheme_prediction_train, grapheme_train))
                
                average_recall_train.append(np.average([grapheme_root_recall_train[-1], \
                    vowel_diacritic_recall_train[-1], \
                    consonant_diacritic_recall_train[-1]], weights=[2, 1, 1]))
                
                log.write('lr: %f train loss: %f average_recall: %f grapheme_root_recall: %f vowel_diacritic_recall: %f consonant_diacritic_recall: %f grapheme_recall: %f\n' % \
                    (rate, train_loss[0], np.mean(average_recall_train), \
                        np.mean(grapheme_root_recall_train), \
                        np.mean(vowel_diacritic_recall_train), \
                        np.mean(consonant_diacritic_recall_train), \
                        np.mean(grapheme_recall_train)))
            
            if (tr_batch_i + 1) % eval_step == 0:  
                
                eval_count += 1
                
                valid_loss = np.zeros(1, np.float32)
                valid_num  = np.zeros_like(valid_loss)
                
                with torch.no_grad():
                    
                    # init cache
                    torch.cuda.empty_cache()

                    for val_batch_i, (image, grapheme_root, vowel_diacritic, consonant_diacritic, grapheme) in enumerate(val_data_loader):
                        
                        # set model to eval mode
                        model.eval()

                        # set input to cuda mode
                        image = image.cuda()
                        grapheme_root    = grapheme_root.cuda().float()
                        vowel_diacritic    = vowel_diacritic.cuda().float()
                        consonant_diacritic    = consonant_diacritic.cuda().float()
                        grapheme    = grapheme.cuda().float()
                        
                        if ((loss_type == 'mae') or (loss_type == 'mse')):
                            
                            _, grapheme_root  = torch.max(grapheme_root, 1)
                            grapheme_root = torch.squeeze(grapheme_root.float())
                            
                            _, vowel_diacritic  = torch.max(vowel_diacritic, 1)
                            vowel_diacritic = torch.squeeze(vowel_diacritic.float())
                            
                            _, consonant_diacritic  = torch.max(consonant_diacritic, 1)
                            consonant_diacritic = torch.squeeze(consonant_diacritic.float())
                            
                            _, grapheme  = torch.max(grapheme, 1)
                            grapheme = torch.squeeze(grapheme.float())

                        # predict and calculate loss (only need torch.sigmoid when inference)
                        predictions = model(image)  
            
                        grapheme_root_prediction = torch.squeeze(predictions[0])
                        vowel_diacritic_prediction = torch.squeeze(predictions[1])
                        consonant_diacritic_prediction = torch.squeeze(predictions[2])
                        grapheme_prediction = torch.squeeze(predictions[3])
                        
                    
                        loss = criterion(grapheme_root_prediction, grapheme_root) + \
                            criterion(vowel_diacritic_prediction, vowel_diacritic) + \
                            criterion(consonant_diacritic_prediction, consonant_diacritic) + \
                            0.1 * criterion(grapheme_prediction, grapheme) 
                            
                        writer.add_scalar('val_loss_' + str(fold), loss.item(), (eval_count-1)*len(val_data_loader)*valid_batch_size+val_batch_i*valid_batch_size)
                        
                        # calculate statistics
                        if ((loss_type == "bce") or (loss_type == "focal")):
                            # prediction = torch.sigmoid(prediction)
                            grapheme_root  = torch.argmax(grapheme_root, 1)
                            grapheme_root = torch.squeeze(grapheme_root.float())
                            
                            vowel_diacritic  = torch.argmax(vowel_diacritic, 1)
                            vowel_diacritic = torch.squeeze(vowel_diacritic.float())
                            
                            consonant_diacritic  = torch.argmax(consonant_diacritic, 1)
                            consonant_diacritic = torch.squeeze(consonant_diacritic.float())
                            
                            grapheme  = torch.argmax(grapheme, 1)
                            grapheme = torch.squeeze(grapheme.float())
                            
                            grapheme_root_prediction  = torch.argmax(grapheme_root_prediction, 1)
                            grapheme_root_prediction = torch.squeeze(grapheme_root_prediction.float())
                            
                            vowel_diacritic_prediction  = torch.argmax(vowel_diacritic_prediction, 1)
                            vowel_diacritic_prediction = torch.squeeze(vowel_diacritic_prediction.float())
                            
                            consonant_diacritic_prediction  = torch.argmax(consonant_diacritic_prediction, 1)
                            consonant_diacritic_prediction = torch.squeeze(consonant_diacritic_prediction.float())
                            
                            grapheme_prediction  = torch.argmax(grapheme_prediction, 1)
                            grapheme_prediction = torch.squeeze(grapheme_prediction.float())

                        grapheme_root_val = grapheme_root.cpu().detach().numpy()
                        vowel_diacritic_val = vowel_diacritic.cpu().detach().numpy()
                        consonant_diacritic_val = consonant_diacritic.cpu().detach().numpy()
                        grapheme_val = grapheme.cpu().detach().numpy()
                        
                        grapheme_root_prediction_val = grapheme_root_prediction.cpu().detach().numpy()
                        vowel_diacritic_prediction_val = vowel_diacritic_prediction.cpu().detach().numpy()
                        consonant_diacritic_prediction_val = consonant_diacritic_prediction.cpu().detach().numpy()
                        grapheme_prediction_val = grapheme_prediction.cpu().detach().numpy()

                        l = np.array([loss.item()*valid_batch_size])
                        n = np.array([valid_batch_size])
                        valid_loss = valid_loss + l
                        valid_num  = valid_num + n
                        
                        grapheme_root_recall_val.append(metric(grapheme_root_prediction_val, grapheme_root_val))
                        vowel_diacritic_recall_val.append(metric(vowel_diacritic_prediction_val, vowel_diacritic_val))
                        consonant_diacritic_recall_val.append(metric(consonant_diacritic_prediction_val, consonant_diacritic_val))
                        grapheme_recall_val.append(metric(grapheme_prediction_val, grapheme_val))
                        
                        average_recall_val.append(np.average([grapheme_root_recall_val[-1], \
                            vowel_diacritic_recall_val[-1], \
                            consonant_diacritic_recall_val[-1]], weights=[2, 1, 1]))
            
                        
                    valid_loss = valid_loss / valid_num
                    
                    log.write('valid loss: %f average_recall: %f grapheme_root_recall: %f vowel_diacritic_recall: %f consonant_diacritic_recall: %f grapheme_recall: %f\n' % \
                            (valid_loss[0], np.mean(average_recall_val), \
                                np.mean(grapheme_root_recall_val), \
                                np.mean(vowel_diacritic_recall_val), \
                                np.mean(consonant_diacritic_recall_val), \
                                np.mean(grapheme_recall_val)))

        val_metric_epoch = np.mean(average_recall_val)

        if (val_metric_epoch >= valid_metric_optimal):
            
            log.write('Validation metric improved ({:.6f} --> {:.6f}).  Saving model ...'.format(\
                    valid_metric_optimal, val_metric_epoch))

            valid_metric_optimal = val_metric_epoch
            torch.save(model.state_dict(), checkpoint_filepath)
            
            count = 0
        
        else:
            count += 1
            
        if (count == early_stopping):
            break
        
        
        
if __name__ == "__main__":
    
    # torch.multiprocessing.set_start_method('spawn')

    args = parser.parse_args()

    seed_everything(args.seed)

    # get train val split
    
    get_train_val_split(df_path=args.df_path, \
                        save_path=args.save_path, \
                        n_splits=args.n_splits, \
                        seed=args.seed, \
                        split=args.split)

    # get train_data_loader and val_data_loader
    data_df_train = args.save_path + '/split/train_fold_%s_seed_%s.csv'%(args.fold, args.seed)
    data_df_val = args.save_path + '/split/val_fold_%s_seed_%s.csv'%(args.fold, args.seed)

        
    train_data_loader, val_data_loader = get_train_val_loaders(data_path=args.data_path, \
                    train_df=data_df_train, \
                    val_data_path=args.data_path, \
                    val_df=data_df_val, \
                    batch_size=args.batch_size, \
                    val_batch_size=args.valid_batch_size, \
                    num_workers=args.num_workers, \
                    train_transform=train_transform, \
                    val_transform=test_transform)
        

    # start training
    training(
            args.n_splits, \
            args.fold, \
            train_data_loader, \
            val_data_loader, \
            args.model_type, \
            args.optimizer, \
            args.lr_scheduler, \
            args.lr, \
            args.warmup_proportion, \
            args.batch_size, \
            args.valid_batch_size, \
            args.num_epoch, \
            args.start_epoch, \
            args.accumulation_steps, \
            args.checkpoint_folder, \
            args.load_pretrain, \
            args.seed, \
            args.loss, \
            args.early_stopping)

    gc.collect()

