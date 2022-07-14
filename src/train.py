import torch
import wandb
import tqdm
from data import getAdvSample

def logEpochResult(avg_epoch_loss, avg_epoch_score, epoch, step, phase):
    print('{} Loss: {:.4f} Score(Acc or Err): {:.4f}'.format(
           phase, avg_epoch_loss, avg_epoch_score))
    wandb.log({"Epoch" : epoch,
                phase + " Loss" : avg_epoch_loss,
                phase + " Score" : avg_epoch_score},
                step = step)

def log_dp(eps, alpha, step):
    wandb.log({"epsilon"    : eps,
               "alpha"      : alpha},
               step = step)

def _detachedPredict(model_ft, img):
    with torch.no_grad():
        model_ft.eval()
        pred = list(model_ft(img).to("cpu").detach().numpy()[0])
        model_ft.train()
        return pred

def runPhase(phase, dl, model_ft, optimizer, criterion, learn, score_fn, device):
    ds_size = dl.ds_size
    if phase == 'train':
        model_ft.train()
    else:
        model_ft.eval()

    loss_sum = 0
    score_sum = 0
    step_ctr = 0
    for inputs, labels in dl:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        loss = criterion(outputs, labels)
        score_sum += score_fn(outputs, labels)
        loss_sum += loss.detach().sum().item()

        if phase == 'train' and learn == True:
            loss.backward()
            optimizer.step()

        step_ctr += 1

    avg_epoch_loss = loss_sum / ds_size
    avg_epoch_score = float(score_sum) / ds_size
    return step_ctr, avg_epoch_loss, avg_epoch_score


def train_model(cfg, model, criterion, optimizer, dataloaders, score_fn,
                scheduler, device):
    epochs = cfg['epochs']
    do_mal_pred = cfg['pred_mal_samples']
    delta = cfg['delta']

    phases = ['train', 'val']

    model_ft = model.nn
    model_ft.to(device)

    mal_pred_arr = []
    nonmal_pred_arr = []
    loss_arr = {'train':[],'val':[]}
    score_arr = {'train':[],'val':[]}
    eps_arr = []
    step = 0

    if do_mal_pred:
        mal_img = getAdvSample(cfg, True)[0][None, :].to(device)
        mal_pred_arr.append(_detachedPredict(model_ft, mal_img))
        nonmal_img = getAdvSample(cfg, False)[0][None, :].to(device)
        nonmal_pred_arr.append(_detachedPredict(model_ft, nonmal_img))

    for phase in phases:
        ret = runPhase(phase, dataloaders[phase], model_ft, optimizer,
                       criterion, False, score_fn, device)
        _, rloss, rscore = ret
        loss_arr[phase].append(rloss)
        score_arr[phase].append(rscore)
        logEpochResult(rloss, rscore, 0, step, phase)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in phases:
            ret = runPhase(phase, dataloaders[phase], model_ft, optimizer,
                           criterion, True, score_fn, device)
            step_ctr, rloss, rscore = ret
            loss_arr[phase].append(rloss)
            score_arr[phase].append(rscore)
            if phase == 'train':
                step += step_ctr

            logEpochResult(rloss, rscore, epoch, step, phase)

        if 'privacy_engine' in optimizer.__dict__:
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
            print(
                f"(epsilon = {epsilon:.2f}, delta = {delta}) for alpha = {best_alpha}"
            )
            eps_arr.append((epsilon, best_alpha))
            log_dp(epsilon, best_alpha, step)

        if do_mal_pred:
            mal_pred_arr.append(_detachedPredict(model_ft, mal_img))
            nonmal_pred_arr.append(_detachedPredict(model_ft, nonmal_img))

        if not scheduler is None:
            scheduler.step()

    meta = {
        'loss_arr'          : loss_arr,
        'score_arr'         : score_arr,
        'mal_pred_arr'      : mal_pred_arr,
        'nonmal_pred_arr'   : nonmal_pred_arr,
        'epsilon_arr'       : eps_arr,
    }
    return meta
