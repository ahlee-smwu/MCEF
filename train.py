import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime
from tqdm import tqdm
torch.backends.cudnn.benchmark = False
from torch.optim.lr_scheduler import ReduceLROnPlateau
from SoftHGRLoss import SoftHGRLoss
import torch.nn.functional as F

def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('data/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, kl_loss, dataloader, epoch, optimizer=None, train=False, gamma_1=1.0,
                        gamma_2=1.0, gamma_3=1.0):
    losses, preds, labels, masks = [], [], [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
            kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob = model(textf, visuf, acouf, umask, qmask, lengths)

        lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)

        kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[2])
        kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[2])
        kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[2])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[2])

        loss = gamma_1 * loss_function(lp_all, labels_, umask)
            #    gamma_2 * (loss_function(lp_1, labels_, umask) + loss_function(lp_2, labels_, umask) + loss_function(
            # lp_3, labels_, umask)) \
                # KL-loss가 Nan을 만들면서 exploit
               # + gamma_3 * (kl_loss(kl_lp_1, kl_p_all, umask) + kl_loss(kl_lp_2, kl_p_all, umask) + kl_loss(kl_lp_3,
               #                                                                                            kl_p_all,
               #                                                                                            umask))

        lp_ = all_prob.view(-1, all_prob.size()[2])

        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore

def train_or_eval_model_HGR(model, loss_function, kl_loss, hgr_loss, dataloader, epoch, optimizer=None, train=False, gamma_1=1.0,
                        gamma_2=1.0, gamma_3=1.0):
    losses, preds, labels, masks = [], [], [], []
    preds_t, preds_a, preds_v = [], [], []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        t_transformer_out, a_transformer_out, v_transformer_out, \
            t_final_out, a_final_out, v_final_out, all_final_out = model(textf, visuf, acouf, umask, qmask, lengths)

        '''softmax/log for final output'''
        t_log_prob = F.log_softmax(t_final_out, 2)
        a_log_prob = F.log_softmax(a_final_out, 2)
        v_log_prob = F.log_softmax(v_final_out, 2)
        all_log_prob = F.log_softmax(all_final_out, 2)
        t_prob = F.softmax(t_final_out, 2)
        a_prob = F.softmax(a_final_out, 2)
        v_prob = F.softmax(v_final_out, 2)
        all_prob = F.softmax(all_final_out, 2)

        kl_t_log_prob = F.log_softmax(t_final_out / args.temp, 2)
        kl_a_log_prob = F.log_softmax(a_final_out / args.temp, 2)
        kl_v_log_prob = F.log_softmax(v_final_out / args.temp, 2)
        kl_all_prob = F.softmax(all_final_out / args.temp, 2)

        '''reshaping'''
        lp_1 = t_log_prob.view(-1, t_log_prob.size()[2])
        lp_2 = a_log_prob.view(-1, a_log_prob.size()[2])
        lp_3 = v_log_prob.view(-1, v_log_prob.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)

        kl_lp_1 = kl_t_log_prob.view(-1, kl_t_log_prob.size()[2])
        kl_lp_2 = kl_a_log_prob.view(-1, kl_a_log_prob.size()[2])
        kl_lp_3 = kl_v_log_prob.view(-1, kl_v_log_prob.size()[2])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[2])

        # print('mask', umask.shape)
        mask = umask.view(-1,1)
        t_output = t_transformer_out.reshape(-1, t_transformer_out.shape[-1])
        a_output = a_transformer_out.reshape(-1, a_transformer_out.shape[-1])
        v_output = v_transformer_out.reshape(-1, v_transformer_out.shape[-1])
        t_output = t_output * mask
        a_output = a_output * mask
        v_output = v_output * mask

        # print('shape\n', t_output.size(), a_output.size(), v_output.size())
        # torch.Size([1264, 1024])

        #     def forward(self, f_t, f_a, f_v):
        #         self.projection_matrix = nn.Linear(num_heads * V_dim, model_dim) == (batch까지 뭉쳐진 dim, output_dim)
        #           torch.Size([16, 79, 256]) -> torch.Size([1264, 256])

        ''''loss'''
        CE = loss_function(lp_all, labels_, umask)
        HGR = hgr_loss(F.softmax(t_output, dim=1), F.softmax(a_output, dim=1), F.softmax(v_output, dim=1))
        HGR = torch.abs(HGR)
        T_loss = loss_function(lp_1, labels_, umask)
        A_loss = loss_function(lp_2, labels_, umask)
        V_loss = loss_function(lp_3, labels_, umask)

        loss = 0.3 * CE + 0.4 * HGR + 0.1 * (T_loss + A_loss + V_loss)
        # print(f'Loss: CE:{CE}, HGR:{HGR}, T:{T_loss}, A:{A_loss}, V:{V_loss}')

        '''accuracy'''
        lp_ = all_prob.view(-1, all_prob.size()[2])
        lp_t = t_prob.view(-1, t_prob.size()[2])
        lp_a = a_prob.view(-1, a_prob.size()[2])
        lp_v = v_prob.view(-1, v_prob.size()[2])

        pred_ = torch.argmax(lp_, 1)
        pred_t = torch.argmax(lp_t, 1)
        pred_a = torch.argmax(lp_a, 1)
        pred_v = torch.argmax(lp_v, 1)
        preds.append(pred_.data.cpu().numpy())
        preds_t.append(pred_t.data.cpu().numpy())
        preds_a.append(pred_a.data.cpu().numpy())
        preds_v.append(pred_v.data.cpu().numpy())

        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        pass
                        #writer.add_histogram(name, param.grad, epoch)
                # for param in model.named_parameters():
                #         writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    preds_t = np.concatenate(preds_t) if preds_t != [] else print('Error for preds_t!')
    preds_a = np.concatenate(preds_a) if preds_a != [] else print('Error for preds_a!')
    preds_v = np.concatenate(preds_v) if preds_v != [] else print('Error for preds_v!')

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    t_acc = round(accuracy_score(labels, preds_t, sample_weight=masks) * 100, 2)
    a_acc = round(accuracy_score(labels, preds_a, sample_weight=masks) * 100, 2)
    v_acc = round(accuracy_score(labels, preds_v, sample_weight=masks) * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, \
        CE,HGR,T_loss,A_loss,V_loss, t_acc, a_acc, v_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=150, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')

    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda = args.cuda
    print("cuda: ", cuda)
    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10':1582, 'denseface':342, 'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    D_m = D_audio + D_visual + D_text

    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1

    print('temp {}'.format(args.temp))

    model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
                                        n_classes=n_classes,
                                        hidden_dim=args.hidden_dim,
                                        n_speakers=n_speakers,
                                        dropout=args.dropout)

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    if cuda:
        model.cuda() #여기서 안 넘어가서 아래 코드로 수정함
        #model = model.to('cuda')
    print('cuda')
        
    kl_loss = MaskedKLDivLoss()
    hgr_loss = SoftHGRLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6, verbose=True)

    #optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    for e in range(n_epochs):
        start_time = time.time()

        # train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(model, loss_function, kl_loss, train_loader, e, optimizer, True)
        # valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(model, loss_function, kl_loss, valid_loader, e)
        # test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, kl_loss, test_loader, e)

        train_loss,train_acc, _,_,_, train_fscore,\
            train_CE,train_HGR,train_T_loss,train_A_loss,train_V_loss,\
            train_t_acc, train_a_acc, train_v_acc = train_or_eval_model_HGR(model, loss_function, kl_loss, hgr_loss, train_loader, e, optimizer, True)
        #valid_loss, valid_acc, _, _, _, valid_fscore,_,_,_,_,_ = train_or_eval_model_HGR(model, loss_function, kl_loss, hgr_loss, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, \
            test_CE, test_HGR, test_T_loss, test_A_loss, test_V_loss, \
            test_t_acc, test_a_acc, test_v_acc = train_or_eval_model_HGR(model, loss_function, kl_loss, hgr_loss, test_loader, e)

        all_fscore.append(test_fscore)
        scheduler.step(test_loss.item())

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask = test_label, test_pred, test_mask

        if args.tensorboard:
            # writer.add_scalars('train/Loss', {
            #     'CE_loss': train_CE,
            #     'HGR_loss': train_HGR,
            #     'T_loss': train_T_loss,
            #     'A_loss': train_A_loss,
            #     'V_loss': train_V_loss,
            # }, e)
            # writer.add_scalars('train/Acc', {
            #     'top1_acc': train_acc,
            #     'fscore_acc': train_fscore,
            #     'T_acc': train_t_acc,
            #     'A_acc': train_a_acc,
            #     'V_acc': train_v_acc,
            # }, e)
            # writer.add_scalars('test/Loss', {
            #     'CE_loss': test_CE,
            #     'HGR_loss': test_HGR,
            #     'T_loss': test_T_loss,
            #     'A_loss': test_A_loss,
            #     'V_loss': test_V_loss,
            # }, e)
            # writer.add_scalars('test/Acc', {
            #     'top1_acc': test_acc,
            #     'fscore_acc': test_fscore,
            #     'T_acc': test_t_acc,
            #     'A_acc': test_a_acc,
            #     'V_acc': test_v_acc,
            # }, e)

            '''original'''
            writer.add_scalar('train: loss', train_loss, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)
            writer.add_scalar('test: loss', test_loss, e)
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)

            '''detailed'''
            writer.add_scalar('train: CE_loss', train_CE, e)
            writer.add_scalar('train: HGR_loss', train_HGR, e)
            writer.add_scalar('train: T_loss', train_T_loss, e)
            writer.add_scalar('train: A_loss', train_A_loss, e)
            writer.add_scalar('train: V_loss', train_V_loss, e)
            writer.add_scalar('train: T_acc', train_t_acc, e)
            writer.add_scalar('train: A_acc', train_a_acc, e)
            writer.add_scalar('train: V_acc', train_v_acc, e)

            writer.add_scalar('test: CE_loss', test_CE, e)
            writer.add_scalar('test: HGR_loss', test_HGR, e)
            writer.add_scalar('test: T_loss', test_T_loss, e)
            writer.add_scalar('test: A_loss', test_A_loss, e)
            writer.add_scalar('test: V_loss', test_V_loss, e)
            writer.add_scalar('test: T_acc', test_t_acc, e)
            writer.add_scalar('test: A_acc', test_a_acc, e)
            writer.add_scalar('test: V_acc', test_v_acc, e)

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        if (e+1)%10 == 0:
            print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
            print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))


    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('F-Score: {}'.format(max(all_fscore)))
    print('F-Score-index: {}'.format(all_fscore.index(max(all_fscore)) + 1))
    
    if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
            pk.dump({}, f)
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
        record = pk.load(f)
    key_ = 'name_'
    if record.get(key_, False):
        record[key_].append(max(all_fscore))
    else:
        record[key_] = [max(all_fscore)]
    if record.get(key_+'record', False):
        record[key_+'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
    else:
        record[key_+'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask,digits=4)]
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
        pk.dump(record, f)

    print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))


