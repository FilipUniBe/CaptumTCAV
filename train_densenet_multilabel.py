import os
import pandas as pd
import random
import torch
import pandas as pd

from main_on_CheXpert import preprocess_excel
from model import DenseNet121
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
from data_loading import Load_from_path_Dataset
from model import DenseNet121
from sampler import StratifiedSampler
from tqdm import tqdm


model_name="densenet"

#Training settings
parser = argparse.ArgumentParser(description='graph visual')
parser.add_argument('--name', type=str, default=f"{model_name}" )
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--imgw', type=int, default=320)
parser.add_argument('--imgh', type=int, default=320)
parser.add_argument('--bs', type=int, default=16, help="batch_size")
parser.add_argument('--n_epochs_stop',type=float, default=10, help="the number of epoch waiting before early stopping")
parser.add_argument('--location', type=str)
parser.add_argument("--interp", type=str, default='bilinear')
parser.add_argument('--device', required=True, type=int, help="CUDA device number")
parser.add_argument('--nc', default=5, type=int, help="number of classes")
parser.add_argument('--usemarker', action='store_true', help="use marker images")
parser.add_argument('--useconcepts', action='store_true', help="use concept images in DS")
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

homepath="/home/fkraehenbuehl/projects/"

print("args.usemaker",args.usemarker)
if args.usemarker==True:
    dspath="CheXpert-v1.0-marker"
else:
    dspath="CheXpert-v1.0"
print("dspath",dspath)

train_labels_meta=pd.read_csv(homepath+f"{dspath}/train.csv")
val_labels_meta=pd.read_csv(homepath+f"{dspath}/valid.csv")
test_labels_meta=pd.read_csv(homepath+f"{dspath}/test.csv")

print("training data:", train_labels_meta.shape[0])
print("validation data", val_labels_meta.shape[0])
print("test_data", test_labels_meta.shape[0])

exist_labels=["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

print("args.useconcepts",args.useconcepts)
train_labels_meta=preprocess_excel(train_labels_meta,exist_labels,args.useconcepts)
val_labels_meta=preprocess_excel(val_labels_meta,exist_labels,args.useconcepts)
test_labels_meta=preprocess_excel(test_labels_meta,exist_labels,args.useconcepts)

print("training data:", train_labels_meta.shape[0])
print("validation data", val_labels_meta.shape[0])
print("test_data", test_labels_meta.shape[0])

y_train=train_labels_meta[exist_labels].values
y_val=val_labels_meta[exist_labels].values
y_test=test_labels_meta[exist_labels].values

x_train_path=train_labels_meta.Path
x_val_path=val_labels_meta.Path
x_test_path=test_labels_meta.Path


train_dataset=Load_from_path_Dataset(x_train_path, homepath+f"{dspath}/", y_train,args.imgw, args.imgh,mode="train")
val_dataset=Load_from_path_Dataset(x_val_path, homepath+f"{dspath}/", y_val,args.imgw, args.imgh,mode="test")
test_dataset=Load_from_path_Dataset(x_test_path, homepath+f"{dspath}/", y_test, args.imgw, args.imgh,mode="test")


class_train=np.argmax(y_train, axis=1)
sampler_train=StratifiedSampler(class_train,batch_size=args.bs)


train_dataloader=DataLoader(train_dataset, batch_size=args.bs, num_workers=20, sampler=sampler_train)
val_dataloader=DataLoader(val_dataset, batch_size=args.bs, shuffle=False,num_workers=20)
test_dataloader=DataLoader(test_dataset, batch_size=args.bs, shuffle=False,num_workers=20)



def save_checkpoint(state, filepath):
    torch.save(state, filepath)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    return checkpoint

def train(epoch,train_dataloader, model, optimizer, writer,step, n_classes,device):
    model.train()



    train_loss=[]
    labels_list=[]
    preds_list=[]
        
    for batch_indx, data_batch in enumerate(tqdm(train_dataloader)):
        image, labels=data_batch[0].float(),data_batch[1].float()

        # calculating class weight
        class_proportion=1-(torch.sum(labels, 0)/labels.shape[0])
        class_weights=class_proportion/class_proportion.sum()

        if args.cuda:
            #image, labels, class_weights=image.cuda(), labels.cuda(), class_weights.cuda()
            image, labels, class_weights = image.to(device), labels.to(device), class_weights.to(device)

        optimizer.zero_grad()

        logits=model(image)

        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean',pos_weight=class_weights.data)

        loss=criterion(logits,target=labels)

        preds=torch.sigmoid(logits)

      
        labels_list.append(labels)
        preds_list.append(preds)
        train_loss.append(loss.cpu().data.numpy())

        loss.backward()
        optimizer.step()

        writer.add_scalar('training loss', loss, step)
        step+=1

    mean_train_loss=np.mean(train_loss)
    print('Epoch: {}, Loss: {:.4f}'.format(epoch,mean_train_loss))

    targets=torch.cat(labels_list).cpu().data.numpy()
    outputs=torch.cat(preds_list).cpu().data.numpy()
    roc_auc=roc_auc_score(targets, outputs, average=None)
    print("auroc", roc_auc)

    return mean_train_loss, step


def val(writer, step_val, valid_dataloader,model, n_classes,device):
    model.eval()

    val_loss_list=[]
    labels_list=[]
    preds_list=[]

    with torch.no_grad():

        for batch_idx, data_batch in enumerate(tqdm(valid_dataloader)):
            image, labels=data_batch[0].float(),data_batch[1].float()

            # calculating class weight
            class_proportion=1-(torch.sum(labels, 0)/labels.shape[0])
            class_weights=class_proportion/class_proportion.sum()

            if args.cuda:
                #image, labels, class_weights=image.cuda(), labels.cuda(), class_weights.cuda()
                image, labels, class_weights = image.to(device), labels.to(device), class_weights.to(device)

            logits=model(image)
            preds=torch.sigmoid(logits)


            criterion = torch.nn.BCEWithLogitsLoss(reduction='mean',pos_weight=class_weights.data)
            loss=criterion(logits,target=labels)

            val_loss_list.append(loss.cpu().data.numpy())
            writer.add_scalar('validation loss',loss, step_val)

            try:
                roc_auc_batch=roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
                writer.add_scalar('roc_auc_val',roc_auc_batch, step_val)

            except ValueError:
                pass

            labels_list.append(labels.cpu().data.numpy())
            preds_list.append(preds.cpu().data.numpy())

            step_val+=1


        val_loss=np.mean(val_loss_list)
        print("validation loss", val_loss)

        targets=np.concatenate(labels_list)
        outputs=np.concatenate(preds_list)

        roc_auc=roc_auc_score(targets, outputs, average=None)
        ap=average_precision_score(targets, outputs)

        print("auroc", roc_auc)
        print("ap",round(ap,4))

        return val_loss, step_val,np.mean(roc_auc)


    
def test(test_dataloader, n_classes,model,device):
    model.eval()

    test_loss=[]
    labels_list=[]
    preds_list=[]

    with torch.no_grad():

        for batch_idx, data_batch in enumerate(tqdm(test_dataloader)):
            image, labels=data_batch[0].float(),data_batch[1].float()

            # calculating class weight
            class_proportion=1-(torch.sum(labels, 0)/labels.shape[0])
            class_weights=class_proportion/class_proportion.sum()

            if args.cuda:
                image, labels, class_weights=image.cuda(), labels.cuda(), class_weights.cuda()
                image, labels, class_weights = image.to(device), labels.to(device), class_weights.to(device)

            logits=model(image)
            preds=torch.sigmoid(logits)

            criterion = torch.nn.BCEWithLogitsLoss(reduction='mean',pos_weight=class_weights.data)
            # criterion = torch.nn.BCELoss()
            loss=criterion(logits,target=labels)

            labels_list.append(labels.cpu().data.numpy())
            preds_list.append(preds.cpu().data.numpy())
            test_loss.append(loss.cpu().data.numpy())

        test_loss_mean=np.mean(test_loss)
        print("test_loss", test_loss_mean)

        targets=np.concatenate(labels_list)
        outputs=np.concatenate(preds_list)

        roc_auc=roc_auc_score(targets, outputs)
        ap=average_precision_score(targets, outputs)

        return roc_auc, ap, test_loss_mean
    

if __name__ == "__main__":
     
    storinghome="prep-model"

    # Step 3: Write all argument variables to a text file
    textfilename=f"args_output_{args.name}.txt"
    textfilefullpath=os.path.join(storinghome,textfilename)
    with open(textfilefullpath, "w") as file:
        # Convert the Namespace to a dictionary for easy writing
        for arg, value in vars(args).items():
            file.write(f"{arg}: {value}\n")

    PATH_TB=storinghome+'/tensorboard_results'
    os.makedirs(PATH_TB,exist_ok=True)
    os.makedirs(PATH_TB+"/",exist_ok=True)

    writer=SummaryWriter(PATH_TB+"/tensorboard_logs_{}_{}_lr{}_epoches{}".format(args.name,args.interp,args.lr, args.epochs))

    step_val=0
    step=0
    max_auc=0

    model=DenseNet121(num_classes=args.nc)
    if args.cuda:
        #model.cuda()
        model.to(args.device)
        print(f"Model is running on: {next(model.parameters()).device}")
    optimizer=optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, f'min', patience=5)

    # Check if a checkpoint exists
    os.makedirs(storinghome, exist_ok=True)
    checkpoint_name = f'checkpoint_latest_{args.name}.pth.tar'
    checkpoint_path=os.path.join(storinghome,checkpoint_name) #todo changed from last time this

    if os.path.isfile(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = load_checkpoint(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_auc = checkpoint['best_auc']
        epoch_min='epoch_min' #todo current running model doesn't have this
        print(f"Checkpoint loaded, resuming from epoch {start_epoch} with best AUC {max_auc:.4f}")
    else:
        start_epoch = 0
        max_auc = 0.0
        print("No checkpoint found, starting training from scratch.")


    for epoch in range(start_epoch, args.epochs):
        print('Start training')
        training_loss, step=train(epoch=epoch, train_dataloader=train_dataloader, model=model, optimizer=optimizer, writer=writer, step=step, n_classes=args.nc,device=args.device)
        
        print('Start Validation')
        val_loss, step_val,roc_auc=val(writer=writer, step_val=step_val, model=model,valid_dataloader=val_dataloader, n_classes=args.nc,device=args.device)
        scheduler.step(val_loss)

        #early stopping
        if (roc_auc-max_auc)>0.0001:
            best_model=model
            max_auc=roc_auc
            epoch_min=epoch
            epochs_no_improve=0

            model_state=model.state_dict()
            optimizer_state=optimizer.state_dict()

        else:
            epochs_no_improve += 1

        # Save the latest checkpoint every epoch
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_auc': max_auc,
            'optimizer': optimizer.state_dict(),
            'epoch_min': epoch_min,  # Add this to the checkpoint #todo current running model doesn't have this
        }, checkpoint_path)

        if epoch > 10 and epochs_no_improve ==args.n_epochs_stop:
            print('Early stopping!' )
            break
        else:
            continue

    writer.close()

    # saving the model
    PATH_SAVE=storinghome+'/models'
    os.makedirs(PATH_SAVE, exist_ok=True)
    PATH_SAVE_MODEL=PATH_SAVE+"/{}_model_{}_lr{}_epoches{}.pt".format(args.name, args.interp,args.lr, args.epochs)
    torch.save({'epoch':epoch_min, 'model_state_dict': model_state,'optimizer_state_dict': optimizer_state,'AUC':max_auc}, PATH_SAVE_MODEL)

    print('Start Testing')
    roc_auc, ap, test_loss=test(test_dataloader=test_dataloader,n_classes=args.nc,model=best_model,device=args.device)
    print("test roc_auc:", roc_auc,np.mean(roc_auc), "average precision:", ap,"test_loss",test_loss)

    # outputing results
    RESULT_SAVING=storinghome+'/evalution_metrices'
    os.makedirs(RESULT_SAVING,exist_ok=True)

    os.makedirs(RESULT_SAVING+"/seed"+str(args.seed),exist_ok=True)
    file_name="{}_evaluation_metries_{}_lr{}_epoches{}.csv".format(args.name, args.interp, args.lr, args.epochs)
    file_path=RESULT_SAVING+"/seed"+"/"+file_name
    with open(file_path, 'w') as csvfile:
        writer=csv.DictWriter(csvfile, fieldnames=["roc_auc","average precision" ,"accuracy", "test_loss"])
        writer.writeheader()
        writer.writerows([{"roc_auc":roc_auc, "average precision":ap, "test_loss":test_loss}])













        









    
    







