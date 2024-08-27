import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import copy
import torch
import joblib
import argparse
import numpy as np
from torch.optim import Adam
from MIL_model import MCFN
from torch import nn, optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
import random
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,roc_curve, auc
from sklearn.preprocessing import label_binarize
def prediction(v_model, val_data, all_data, batch_size, optimizer, loss_fn_c,  args):

    v_model.eval()

    lbl_batch = None
    pre_batch = None
    lbl_epoch = []
    pre_epoch = []
    
    with torch.no_grad():
        for i_batch, id in enumerate(val_data):


            data = all_data[id[0]]
            class_label= torch.tensor([id[1]], dtype=torch.int64).to(device)

            lbl_pred, class_hat = v_model(data)

            if iter == 0 or lbl_batch == None:
                pre_batch = lbl_pred
                lbl_batch = class_label
            else:
                pre_batch = torch.cat([pre_batch, lbl_pred])
                lbl_batch = torch.cat([lbl_batch, class_label])

            predicted = class_hat.detach().cpu().numpy()[0]
            pre_epoch.append(predicted)
            lbl_epoch.append(id[1])


        all_loss = loss_fn_c(pre_batch, lbl_batch)

        all_loss =all_loss / len(val_data)

        pre_epoch = np.argmax(pre_epoch, axis=1)
        Acc = accuracy_score(lbl_epoch, pre_epoch)


        return all_loss, Acc
    




def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
def adjust_learning_rate(optimizer, lr, epoch, lr_step=20, lr_gamma=0.5):
    lr = lr * (lr_gamma ** (epoch // lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_a_epoch(model,train_data,all_data,batch_size,optimizer, loss_fn_c,epoch,args):
    model.train() 

    
    iter = 0

    all_loss = 0.0
    lbl_batch = None
    pre_batch = None

    lbl_epoch = []
    pre_epoch = []
    random.shuffle(train_data)
    for i_batch,id in enumerate(train_data):
        
        iter += 1 

        data = all_data[id[0]]
        class_label= torch.tensor([id[1]], dtype=torch.int64).to(device)

        lbl_pred,class_hat= model(data)

        if iter == 0 or lbl_batch == None:
            pre_batch = lbl_pred
            lbl_batch=class_label
        else:
            pre_batch = torch.cat([pre_batch, lbl_pred])
            lbl_batch = torch.cat([lbl_batch, class_label])

        predicted = class_hat.detach().cpu().numpy()[0]
        pre_epoch.append(predicted)
        lbl_epoch.append(id[1])


        if iter % batch_size == 0 or i_batch == len(train_data)-1:


            optimizer.zero_grad()

            batch_loss_class =  loss_fn_c(pre_batch, lbl_batch)

            loss = batch_loss_class

            all_loss += loss.item()
            loss.backward()
            if epoch == 0:
                print('*',end='')
            else:
                optimizer.step()

            torch.cuda.empty_cache()

            iter = 0
            lbl_batch = None
            pre_batch = None

    all_loss = all_loss/len(train_data)*batch_size

    pre_epoch = np.argmax(pre_epoch, axis=1)
    Acc = accuracy_score(lbl_epoch, pre_epoch)


    return all_loss, Acc

class Data:
    def __init__(self, x_img, x_rna, x_cli):
        self.x_img = x_img
        self.x_rna = x_rna
        self.x_cli = x_cli

def main(args): 
    start_seed = args.start_seed
    cancer_type = args.cancer_type
    repeat_num = args.repeat_num
    drop_out_ratio = args.drop_out_ratio
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    details = args.details
    fusion_model = args.fusion_model
    if_adjust_lr = args.if_adjust_lr
    keep_rate=args.keep_rate
    fusion=args.fusion
    out_classes = args.out_classes

    label = "{} {} lr_{}".format(cancer_type, details, lr)

    print(label)



    if cancer_type == 'brca':
        all_data = joblib.load('your path')
        seed_fit_split = joblib.load('your path')
        omic_sizes=[]
    elif cancer_type == 'nsclc':
        all_data = joblib.load('your path')
        seed_fit_split = joblib.load('your path')
        omic_sizes = []
    elif cancer_type == 'rcc':
        all_data = joblib.load('your path')
        seed_fit_split = joblib.load('your path')
        omic_sizes = []




    repeat = -1
    for seed in range(start_seed,start_seed+repeat_num):
        repeat+=1
        setup_seed(0)
        test_fold_Acc=[]
        test_fold_F1 = []
        test_fold_AUC = []


        for n_fold in range(5):
            fold_patients = []
            n_fold+=1
            print('fold: ',n_fold)
             
            if fusion_model == 'MCFN':
                model = MCFN.MCFN(in_feats=768,
                               n_hidden=args.n_hidden,
                               dropout=drop_out_ratio,
                               keep_rate = keep_rate,
                               omic_sizes=omic_sizes,
                               fusion=fusion,
                               out_classes=out_classes
                                           ).to(device)

            model_log_folder = 'log/' +cancer_type+'/'
            if not os.path.exists(model_log_folder):
                os.makedirs(model_log_folder)
            path_dir = os.path.join(model_log_folder, 'best_model_' + str(n_fold)+'_'+str(lr) + '.pth')
            result_dir = os.path.join(model_log_folder, 'result_' +str(lr) + '.pkl')

            optimizer=Adam(model.parameters(),lr=lr,weight_decay=5e-4)
            loss_fn_c = nn.CrossEntropyLoss()
            
            if args.if_fit_split:
                train_data = seed_fit_split[n_fold-1][0]
                val_data = seed_fit_split[n_fold-1][1]
                test_data = seed_fit_split[n_fold-1][2]

            print(len(train_data),len(val_data),len(test_data))

   

            best_val_acc = 0
            count = 0

            for epoch in range(epochs):
                
                if if_adjust_lr:
                    adjust_learning_rate(optimizer, lr, epoch, lr_step=30, lr_gamma=args.adjust_lr_ratio)
                start_t = time.perf_counter()
                train_loss,t_train_acc = train_a_epoch(model,train_data,all_data,batch_size,optimizer, loss_fn_c,epoch,args)
                end_t = time.perf_counter()
                v_loss, val_acc = prediction(model, val_data, all_data, batch_size, optimizer, loss_fn_c,args)
                print("epoch：{:2d}，train_loos：{:.4f},train_acc：{:.4f},val_loos：{:.4f},val_acc：{:.4f},time：{:.4f},".format(epoch, train_loss, t_train_acc,v_loss,val_acc,round(end_t - start_t)))

                if val_acc >= best_val_acc:

                    count = 0
                    best_val_acc=val_acc
                    torch.save(model.state_dict(), path_dir)
                    t_model = copy.deepcopy(model)
                else:
                    count = count + 1
                    if count > 20 and epoch>60 :
                        break


            t_model.eval()


            lbl_epoch = []
            pre_epoch = []

            with torch.no_grad():
                for i_batch, id in enumerate(test_data):

                    data = all_data[id[0]]

                    _, class_hat = t_model(data)

                    predicted = class_hat.detach().cpu().numpy()[0]

                    lbl_epoch.append(id[1])
                    if out_classes<=2:
                        pre_epoch.extend(predicted[1:])
                    else:
                        pre_epoch.append(predicted)

                if out_classes<=2:

                    y_true = lbl_epoch
                    y_pred = pre_epoch
                    AUC = roc_auc_score(y_true, y_pred)
                    y_pred = np.array(y_pred)
                    y_pred[y_pred > 0.5] = 1
                    y_pred[y_pred < 0.5] = 0
                    Acc = accuracy_score(y_true, y_pred)
                    F1 = f1_score(y_true, y_pred)
                else:
                    y_true=lbl_epoch
                    y_pred = np.argmax(pre_epoch, axis=1)
                    Acc = accuracy_score(lbl_epoch, y_pred)
                    # 计算F1-score（使用宏平均）
                    F1 = f1_score(y_true, y_pred, average='macro')
                    # 计算AUC
                    # 将标签进行二进制编码
                    y_true_binary = label_binarize(y_true, classes=np.unique(y_true))
                    y_probs_binary = label_binarize(np.argmax(pre_epoch, axis=1), classes=np.unique(y_true))

                    # 计算每个类别的ROC曲线和AUC
                    n_classes = len(np.unique(y_true))
                    auc_scores = []
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_probs_binary[:, i])
                        auc_i = auc(fpr, tpr)
                        auc_scores.append(auc_i)
                    # macro_auc
                    AUC = np.mean(auc_scores)

            test_fold_Acc.append(Acc)
            test_fold_F1.append(F1)
            test_fold_AUC.append(AUC)
            torch.save(t_model.state_dict(), path_dir)
            del model, train_data, val_data,test_data, t_model
            print("Acc：%0.2f" % (Acc * 100))
            print("F1：%0.2f" % (F1 * 100))
            print("Auc：%0.2f" % (AUC * 100))
        acc_mean = np.mean(test_fold_Acc)
        acc_std = np.std(test_fold_Acc)
        f1_mean = np.mean(test_fold_F1)
        f1_std = np.std(test_fold_F1)
        auc_mean = np.mean(test_fold_AUC)
        auc_std = np.std(test_fold_AUC)
        print("cancer type：", cancer_type)
        print("Acc：%0.2f±%0.2f" % (acc_mean * 100, acc_std * 100))
        print("F1：%0.2f±%0.2f" % (f1_mean * 100, f1_std * 100))
        print("Auc：%0.2f±%0.2f" % (auc_mean * 100, auc_std * 100))

        result_list = {'Acc':test_fold_Acc,'F1':test_fold_F1,'AUC':test_fold_AUC}

        joblib.dump(result_list, result_dir)

    
def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer_type", type=str, default="brca", help="Cancer type")
    parser.add_argument("--start_seed", type=int, default=1, help="start_seed")
    parser.add_argument("--repeat_num", type=int, default=1, help="Number of repetitions of the experiment")
    parser.add_argument("--fusion_model", type=str, default="MCFN", help="Which model to use")
    parser.add_argument("--drop_out_ratio", type=float, default=0.4, help="Drop_out_ratio")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate of model training")
    parser.add_argument("--epochs", type=int, default=90, help="Cycle times of model training")
    parser.add_argument("--batch_size", type=int, default=32, help="Data volume of model training once")
    parser.add_argument("--n_hidden", type=int, default=256, help="Model middle dimension")
    parser.add_argument("--out_classes", type=int, default=2, help="Model out dimension")
    parser.add_argument("--if_adjust_lr", action='store_true', default=True, help="if_adjust_lr")
    parser.add_argument("--adjust_lr_ratio", type=float, default=0.5, help="adjust_lr_ratio")
    parser.add_argument("--if_fit_split", action='store_true', default=True, help="fixed division/random division")
    parser.add_argument("--keep_rate", type=float, default=0.3, help="The keep rate of instance is used for ODA module")
    parser.add_argument("--fusion", type=str, default="bilinear", help="bilinear or concat")
    parser.add_argument("--details", type=str, default='', help="Experimental details")
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        args=get_params()
        main(args)
    except Exception as exception:
        raise
    
       

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
