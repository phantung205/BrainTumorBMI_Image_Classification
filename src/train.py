from argparse import ArgumentParser
import os
from src import config,dataset
from src.model import BrainTumorMRICNN
import torch
import shutil
import torch.nn as nn
from tqdm.autonotebook import  tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from sklearn.metrics import accuracy_score


def get_args():
    parser = ArgumentParser(description="train brain tumorMBI")
    parser.add_argument("--root", "-r", type=str, default=config.data_dir, help="dataset root path")
    parser.add_argument("--batch_size", "-b", type=int, default=config.batch_size, help="batch size")
    parser.add_argument("--image_size", "-i", type=int, default=config.image_size, help="input image size")
    parser.add_argument("--logging", "-l", type=str, default=config.path_tensorboard, help="tensorboard log directory")
    parser.add_argument("--trained_models", "-t", type=str, default=config.model_dir, help="directory to save models")
    parser.add_argument("--learning_rate", "-lr", type=float, default=config.learning_rate, help="learning rate")
    parser.add_argument("--momentum", "-m", type=float, default=config.momentum, help="momentum")
    parser.add_argument("--epochs", "-e", type=int, default=config.epochs, help="number epoch")
    parser.add_argument("--checkpoint","-c",type=str,default=None)
    args = parser.parse_args()
    return args


def train(args):
    # use th GPU if the CPU is unavailable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get dataloader
    train_dataloader, test_dataloader = dataset.BrainTumorMRIDataloafers(data_dir=args.root,batch_size=args.batch_size,image_size=args.image_size)

    # remove tensorboard old
    if  os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    # check if the model exists
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)

    #initialize tensorboard
    writer = SummaryWriter(args.logging)

    # initialize model
    model = BrainTumorMRICNN().to(device)

    # initialize criterion
    criterion = nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = SGD(model.parameters(),lr=args.learning_rate,momentum=args.momentum)

    num_iters = len(train_dataloader)

    #load epoch old
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_acc = 0

    for epoch in range(start_epoch,args.epochs):
        # turn on mode train
        model.train()
        progress_bar = tqdm(train_dataloader,colour="cyan")
        for iter ,(images,labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            # loss
            loss_value = criterion(outputs,labels)

            # add tensorboard
            writer.add_scalar("train/loss",loss_value.item(),epoch*num_iters+iter)

            # proget bar
            progress_bar.set_description("Epoch {}/{} , iteration {}/{} , losss {:.3f}".format(epoch+1,args.epochs,iter+1,num_iters,loss_value.item()))


            # back ward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        #turn on model validation
        model.eval()
        progress_bar = tqdm(test_dataloader,colour="red")
        all_labels = []
        all_predictions = []
        for images, labels in progress_bar:
            all_labels.extend(labels)

            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions.cpu(),dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(predictions,labels)

        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]

        accuracy = accuracy_score(all_labels,all_predictions)
        print("Epoch : {}, accuaracy :{} ".format(epoch + 1, accuracy))
        writer.add_scalar("val/Accuracy", accuracy, epoch)

        # save last model ,learning rate ,epochs
        checkpoint = {
            "epoch":epoch+1,
            "best_acc":accuracy,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        # save best model ,learning rate ,epochs
        if accuracy > best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": accuracy,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))
            best_acc = accuracy



if __name__ == '__main__':
    args = get_args()
    train(args)