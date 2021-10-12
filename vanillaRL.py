import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch_optimizer as optim

from tqdm.auto import tqdm


#data + embedding

#ddpg

#soft update
#ddpg update

import sys
import os
sys.path.append(os.getcwd())

import recnn2
from recnn2.nn.models import Actor, Critic
from recnn2.nn.utils import soft_update
from recnn2.nn.update.ddpg import ddpg_update
from recnn2.data.env import DataPath, FrameEnv
from recnn2.utils.plot import Plotter
cuda = torch.device('cuda')

frame_size = 9
batch_size = 25
n_epochs   = 100
plot_every = 30
#
# dirs = DataPath(
#     base=os.path.join(os.getcwd(), 'data/'),
#     embeddings="embeddings/ml20_pca128.pkl",
#     ratings="ml-20m/ratings.csv",
#     cache="cache/frame_env.pkl", # cache will generate after you run
#     use_cache=True
# )

dirs = DataPath(
    base='/data/workspace/holly0015/test_project1/project1/src/thesis/',
    embeddings="embedding/thesis_pca128.pkl",
    ratings="parsed/research_rec_tfidf_v2.3.csv",
    cache="cache/frame_env.pkl", # cache will generate after you run
    use_cache=False
)

env = FrameEnv(dirs, frame_size, batch_size)

def run_tests(nets, optimizer):
    step = 0
    debug = {}

    test_batch = next(iter(env.test_dataloader))
    losses, debug = ddpg_update(test_batch, params, nets, optimizer,learn=False, step=step)

    gen_actions = debug['next_action']
    true_actions = env.base.embeddings.detach().cpu().numpy()


    #f = plotter.kde_reconstruction_error(ad, gen_actions, true_actions, cuda)
    #writer.add_figure('rec_error', f, losses['step'])
    return losses



# === ddpg settings ===

params = {
    'gamma': 0.99,
    'min_value': -10,
    'max_value': 10,
    'policy_step': 10,
    'soft_tau': 0.001,

    'policy_lr': 1e-5,
    'value_lr': 1e-5,
    'actor_weight_init': 54e-2,
    'critic_weight_init': 6e-1,
}

def ddpg_model():
    value_net  = Critic(1290, 128, 256, params['critic_weight_init'])#.to(cuda) cuda is too old
    policy_net = Actor(1290, 128, 256, params['actor_weight_init'])#.to(cuda)


    target_value_net = Critic(1290, 128, 256)#.to(cuda)
    target_policy_net = Actor(1290, 128, 256)#.to(cuda)

    # ad = recnn.nn.models.AnomalyDetector().to(cuda)
    # ad.load_state_dict(torch.load('../../models/anomaly.pt'))
    # ad.eval()

    target_policy_net.eval()
    target_value_net.eval()

    soft_update(value_net, target_value_net, soft_tau=1.0)
    soft_update(policy_net, target_policy_net, soft_tau=1.0)

    value_criterion = nn.MSELoss()

    # from good to bad: Ranger Radam Adam RMSprop
    value_optimizer = optim.Ranger(value_net.parameters(),
                                  lr=params['value_lr'], weight_decay=1e-2)
    policy_optimizer = optim.Ranger(policy_net.parameters(),
                                   lr=params['policy_lr'], weight_decay=1e-5)

    loss = {
        'test': {'value': [], 'policy': [], 'step': []},
        'train': {'value': [], 'policy': [], 'step': []}
        }


    plotter = Plotter(loss, [['value', 'policy']],)

    step = 0
    optimizer = dict()
    optimizer["value_optimizer"] = value_optimizer
    optimizer["policy_optimizer"] = policy_optimizer
    nets = {
        'value_net': value_net,
        'target_value_net': target_value_net,
        'policy_net': policy_net,
        'target_policy_net': target_policy_net,
    }
    for epoch in range(100):

        for batch in tqdm(env.train_dataloader):
            loss, debug = ddpg_update(batch, params, nets,optimizer, step=step)

            plotter.log_losses(loss)
            step += 1
            if step % 500 == 0:#plot_every == 0:
                #clear_output(True)
                print('step', step)
                test_loss = run_tests(nets,optimizer)
                plotter.log_losses(test_loss, test=True)
                plotter.plot_loss()
            # if step > 1000:
            #     assert False


    torch.save(value_net.state_dict(), os.path.join(os.getcwd(), 'thesis/models/ddpg_value.pt'))
    torch.save(policy_net.state_dict(), os.path.join(os.getcwd(), 'thesis/models/ddpg_policy.pt'))