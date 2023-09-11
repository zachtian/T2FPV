import argparse
import os
import math
import sys
import pickle
import time
import numpy as np
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import *
from torch.autograd import Variable
from torch import nn
import torch
import torch.utils
import torch.utils.data


# 1. Train NAOMI on our data, once for each fold
# 2. We use NAOMI one time at test to output a new dataset
# 3. We re-run baseline stuff (SGNet, VRNN, A-VRNN) on the new dataset, with NAOMI
# 4. potentially try incorporating our stuff too?

from helpers import *

def get_model(fold):
    params = {
        'task' : fold,
        'batch' : 128,
        'y_dim' : 2,
        'rnn_dim' : 64,
        'dec1_dim' : 64,
        'dec2_dim' : 64,
        'dec4_dim' : 64,
        'dec8_dim' : 64,
        'dec16_dim' : 64,
        'n_layers' : 2,
        'discrim_rnn_dim' : 64,
        'discrim_num_layers' : 2,
        'cuda' : True,
        'highest' : 1,
    }

    save_path = f'out/saved/NAOMI_{fold}_001/model/policy_step1_state_dict_best_pretrain.pth'
    model = NAOMI(params)
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)
    model = model.cuda()
    return model, save_path

if __name__ == '__main__':
    Tensor = torch.FloatTensor
    torch.set_default_tensor_type('torch.FloatTensor')

    def printlog(line):
        print(line)
        with open(save_path+'log.txt', 'a') as file:
            file.write(line+'\n')

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trial', type=int, default=1)
    parser.add_argument('--model', type=str, default='NAOMI', help='NAOMI, SingleRes')
    parser.add_argument('--task', type=str, default='eth', help='basketball, billiard, eth')
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--rnn_dim', type=int, default=64)
    parser.add_argument('--dec1_dim', type=int, default=64)
    parser.add_argument('--dec2_dim', type=int, default=64)
    parser.add_argument('--dec4_dim', type=int, default=64)
    parser.add_argument('--dec8_dim', type=int, default=64)
    parser.add_argument('--dec16_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2, required=False)
    parser.add_argument('--seed', type=int, required=False, default=123)
    parser.add_argument('--clip', type=int, default=10, help='gradient clipping')
    parser.add_argument('--pre_start_lr', type=float, default=1e-3, help='pretrain starting learning rate')
    parser.add_argument('--batch_size', type=int, required=False, default=256)
    parser.add_argument('--save_every', type=int, required=False, default=50, help='periodically save model')
    parser.add_argument('--pretrain', type=int, required=False, default=300, help='num epochs to use supervised learning to pretrain')
    parser.add_argument('--highest', type=int, required=False, default=1, help='highest resolution in terms of step size in NAOMI')
    parser.add_argument('--cuda', action='store_true', default=True, help='use GPU')

    parser.add_argument('--discrim_rnn_dim', type=int, default=64)
    parser.add_argument('--discrim_layers', type=int, default=2)
    parser.add_argument('--policy_learning_rate', type=float, default=1e-6, help='policy network learning rate for GAN training')
    parser.add_argument('--discrim_learning_rate', type=float, default=1e-3, help='discriminator learning rate for GAN training')
    parser.add_argument('--max_iter_num', type=int, default=60000, help='maximal number of main iterations (default: 60000)')
    parser.add_argument('--log_interval', type=int, default=1, help='interval between training status logs (default: 1)')
    parser.add_argument('--draw_interval', type=int, default=200, help='interval between drawing and more detailed information (default: 50)')
    parser.add_argument('--pretrain_disc_iter', type=int, default=2000, help="pretrain discriminator iteration (default: 2000)")
    parser.add_argument('--save_model_interval', type=int, default=50, help="interval between saving model (default: 50)")

    parser.add_argument('--test_only', action='store_true', help='Whether or not to just get test')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.cuda = False
        
    # model parameters
    params = {
        'task' : args.task,
        'batch' : args.batch_size,
        'y_dim' : args.y_dim,
        'rnn_dim' : args.rnn_dim,
        'dec1_dim' : args.dec1_dim,
        'dec2_dim' : args.dec2_dim,
        'dec4_dim' : args.dec4_dim,
        'dec8_dim' : args.dec8_dim,
        'dec16_dim' : args.dec16_dim,
        'n_layers' : args.n_layers,
        'discrim_rnn_dim' : args.discrim_rnn_dim,
        'discrim_num_layers' : args.discrim_layers,
        'cuda' : args.cuda,
        'highest' : args.highest,
    }

    # hyperparameters
    pretrain_epochs = args.pretrain
    clip = args.clip
    start_lr = args.pre_start_lr
    batch_size = args.batch_size
    save_every = args.save_every

    # manual seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)

    # build model
    policy_net = eval(args.model)(params)
    discrim_net = Discriminator(params).double()
    if args.cuda:
        policy_net, discrim_net = policy_net.cuda(), discrim_net.cuda()
    params['total_params'] = num_trainable_params(policy_net)
    print(params)

    # create save path and saving parameters
    save_path = 'out/saved/' + args.model + '_' + args.task + '_%03d/' % args.trial
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path+'model/')

    if args.test_only:
        filename = save_path+'model/policy_step'+str(args.highest)+'_state_dict_best_pretrain.pth'
        model = torch.load(filename, map_location=torch.device('cpu'))
        policy_net.load_state_dict(model)
        out_name = filename.replace('.pth', '_entire.pth')
        torch.save(policy_net, out_name)
        exit()

    pretrained_path = "./config/fpv_det_train/sgnet_cvae.json"
    from run import get_exp_config
    fold = args.task
    pre_config = get_exp_config(pretrained_path, run_type='test', ckpt=None, fold=fold, gpu_id=0, use_cpu=False,
                                max_test_epoch=10000, corr=False, epochs=100, no_tqdm=False)
    pre_config['load_ckpt'] = True
    pre_config['ckpt_name'] = False
    from vrnntools.trajpred_trainers.module import ModuleTrainer
    from vrnntools.trajpred_trainers.sgnet_cvae import SGNetCVAETrainer
    from vrnntools.trajpred_trainers.ego_avrnn import EgoAVRNNTrainer
    from vrnntools.trajpred_trainers.ego_vrnn import EgoVRNNTrainer
    if pre_config['trainer'] == 'module':
        trainer = ModuleTrainer(config=pre_config)
    elif pre_config['trainer'] == 'ego_vrnn':
        trainer = EgoVRNNTrainer(config=pre_config)
    elif pre_config['trainer'] == 'ego_avrnn':
        trainer = EgoAVRNNTrainer(config=pre_config)
    elif pre_config['trainer'] == 'sgnet':
        trainer = SGNetCVAETrainer(config=pre_config)
    _ = trainer.eval(do_eval=False, load_only=True)
    test_data = trainer.test_data
    val_data = trainer.val_data
    train_data = trainer.train_data
    # figures and statistics
    if os.path.exists('imgs'):
        shutil.rmtree('imgs')
    if not os.path.exists('imgs'):
        os.makedirs('imgs')

    # pretrain
    best_test_loss = 0
    lr = start_lr
    teacher_forcing = True
    best_epoch = 0
    for e in range(pretrain_epochs):
        epoch = e+1
        print("Epoch: {}".format(epoch))
  
        update = 'append' if epoch > 1 else None

        # control learning rate
        if epoch == pretrain_epochs // 2:
            lr = lr / 10
            print(lr)
            
        if epoch == pretrain_epochs * 2 // 3:
            teacher_forcing = False

        # train
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, policy_net.parameters()),
            lr=lr)

        start_time = time.time()

        train_loss = run_epoch(True, policy_net, train_data, clip, optimizer, teacher_forcing=teacher_forcing)
        printlog('Train:\t' + str(train_loss))

        val_loss = run_epoch(False, policy_net, val_data, clip, optimizer, teacher_forcing=teacher_forcing)
        printlog('Val:\t' + str(val_loss))


        epoch_time = time.time() - start_time
        printlog('Time:\t {:.3f}'.format(epoch_time))

        total_test_loss = val_loss
        
        update = 'append' if epoch > 1 else None

        # best model on test set
        if (best_test_loss == 0 or total_test_loss < best_test_loss) and not teacher_forcing:    
            best_test_loss = total_test_loss
            filename = save_path+'model/policy_step'+str(args.highest)+'_state_dict_best_pretrain.pth'
            torch.save(policy_net.state_dict(), filename)
            best_epoch = epoch
            printlog('Best model at epoch '+str(epoch))

        # periodically save model
        if epoch % save_every == 0:
            filename = save_path+'model/policy_step'+str(args.highest)+'_state_dict_'+str(epoch)+'.pth'
            torch.save(policy_net.state_dict(), filename)
            printlog('Saved model')
        
    printlog('End of Pretrain, Best Val Loss: {:.4f}'.format(best_test_loss))
    filename = save_path+'model/policy_step'+str(args.highest)+'_state_dict_best_pretrain.pth'
    model = torch.load(filename, map_location=torch.device('cpu'))
    policy_net.load_state_dict(model)
    test_loss = run_epoch(False, policy_net, test_data, clip, optimizer, teacher_forcing=teacher_forcing)
    printlog(f'Test at epoch {best_epoch}:\t' + str(test_loss))

    test_loss = run_epoch(False, None, test_data, clip, optimizer, teacher_forcing=teacher_forcing)
    printlog(f'Base Test at epoch {best_epoch}:\t' + str(test_loss))

    exit()
