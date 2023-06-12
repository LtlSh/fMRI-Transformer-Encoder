"""
Trasnfromer Encdoer for fMRI data calssficiation
By Lital Shytrit and Ariel Eltanov 2023
"""

#general imports
import torch
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix
from torchmetrics import ConfusionMatrix
import random
#my imports
from preprocess_TEC import create_dataset



#Attention block
class AttentionBlock(nn.Module):
    def __init__(self, num_heads=2, head_size=128, ff_dim=None, dropout=0):
        super(AttentionBlock, self).__init__()

        if ff_dim is None:
            ff_dim = head_size

        self.attention = nn.MultiheadAttention(embed_dim=head_size, num_heads=num_heads, dropout=dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(head_size, eps=1e-6)

        self.ff_conv1 = nn.Conv1d(head_size, ff_dim, kernel_size=1)
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(ff_dim, eps=1e-6)

    def forward(self, inputs):
        x, attention_scores = self.attention(inputs, inputs, inputs, need_weights=True)
        x = self.attention_dropout(x)
        x = self.attention_norm(inputs + x)

        x = self.ff_conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.ff_dropout(x)
        x = self.ff_norm(inputs + x)
        return x, attention_scores

#Full model
class TransformerEncoder(nn.Module):
    def __init__(self,classes, num_heads=2, head_size=128, ff_dim=None, num_layers=1,
                 dropout=0):
        super(TransformerEncoder, self).__init__()
        if ff_dim is None:
            ff_dim = head_size
        self.dropout = dropout
        self.classes = classes
        self.attention_layers = nn.ModuleList(
            [AttentionBlock(num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in
             range(num_layers)])
        self.dense_layer = nn.Linear(head_size, 512)
        self.dropout_layer = nn.Dropout(dropout)
        self.final_layer = nn.Linear(512, classes)

    def forward(self, inputs):
        x=inputs
        for attention_layer in self.attention_layers:
            x, attention_scores = attention_layer(x)
        x = self.dense_layer(x)
        x = self.dropout_layer(x)
        #Reducing dimensions fo classification
        x = x.mean(dim=1)
        x = self.final_layer(x)
        return x






#Get item calss for dataloaer
class TimeSeriesDataset(Dataset):
    def __init__(self, subjects_dict):
        self.subjects_dict = subjects_dict
        self.subjects = list(subjects_dict.keys())

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_dict = None
        subject = self.subjects[idx]
        subject_dict = self.subjects_dict[subject]

        # #FOR SANITY CHECK:
        # # Generate a random index
        # random_index = random.randint(0, 15 - 1)
        # # Create a list of all zeros
        # random_label = [0.] * 15
        # # Set the random index to one
        # random_label[random_index] = 1.
        # #Reassign label
        # subject_dict['clip_idx'] = torch.tensor(random_label)#todo
        return subject_dict

#Calculation of evaluation metrices
def calc_metrix(predicted_labels,true_labels, flag=False):
    # Calculate true positives, false positives, false negatives per class
    class_labels = set(true_labels)
    metrics = {}
    for label in class_labels:
        tp = sum((p == label and t == label) for p, t in zip(predicted_labels, true_labels))
        fp = sum((p == label and t != label) for p, t in zip(predicted_labels, true_labels))
        fn = sum((p != label and t == label) for p, t in zip(predicted_labels, true_labels))

        # Calculate precision, recall, F1-score for each class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1
        }

    # Calculate overall accuracy
    correct_predictions = sum(p == t for p, t in zip(predicted_labels, true_labels))
    accuracy = correct_predictions / len(predicted_labels)
    #if flag is raised, it means we're testing and need to calc confusion matrix
    if flag:
        #figure size
        plt.figure(figsize=(10, 7))
        target = torch.tensor(true_labels)
        preds = torch.tensor(predicted_labels)
        #calc cm
        confmat = ConfusionMatrix(task="multiclass", num_classes=15)
        cm = confmat(preds, target)
        #visualize cm
        cm_heatmap = sns.heatmap(cm, cbar=True,  annot=True)
        figure_cm = cm_heatmap.get_figure()
        #turn into wandb image
        wandb_cm = wandb.Image(figure_cm, caption="Confusion Matrix | Movies")
        return {'Accuracy': accuracy, 'Confusion Matrix': wandb_cm}
    else:
        return {'Accuracy': accuracy}


# Defining defalut hyperparameters
learning_rate = 1e-3
epochs =100#todo
batch_size = 16
num_heads = 1
dropout=0.1
embedding_dim = 128

# Instantiating the dataset
directory = r"D:\Final Project\TASK_RH_vis2\dataset" #todo


def train_loop():
    global num_heads

    #creating dataset
    train_subjects_dict, num_voxels = create_dataset(directory, 'train', NET, NET_idx, H)
    train_dataset = TimeSeriesDataset(train_subjects_dict)

    eval_subjects_dict,_ = create_dataset(directory, 'eval', NET, NET_idx, H)
    eval_dataset = TimeSeriesDataset(eval_subjects_dict)

    test_subjects_dict,_ = create_dataset(directory, 'test', NET, NET_idx, H)
    test_dataset = TimeSeriesDataset(test_subjects_dict)

    # Creating the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    #Defining the model architechture in accordance to the current net
    model = nn.Sequential(
        TransformerEncoder(classes = 15, time2vec_dim=1, num_heads=num_heads, head_size=num_voxels, ff_dim=None, num_layers=1,
                 dropout=dropout))
    #sending model to GPU
    device = torch.device('cuda:0')
    model.to(device)
    print('cuda: ', torch.cuda.is_available())

    # Reseting loss
    best_loss = 100

    # Defining loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Asking wandb to watch model weights
    wandb.watch(model)

    ### Train ###
    # Setting the model to training mode
    model.train()

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []

        for idx, train_batch in enumerate(tqdm(train_dataloader)):
            # Forward pass
            # training the model on train batch
            data = train_batch['vis_values']
            data = data.cuda()
            gt = train_batch['clip_idx'].cuda()
            outputs = model(data)

            #calculating loss
            loss = loss_fn(outputs, gt)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Metric calculation and logging
            batch_accuracy =  calc_metrix(torch.argmax(outputs,1), torch.argmax(gt,1))['Accuracy']
            wandb.log({f'Train/Accuracy': batch_accuracy})
            train_losses.append(loss.item())
            print(f"batch [{idx}/{len(train_dataloader)}], Batch Loss: {loss.item()}")

           #logging results to wandb
            if idx % 100 == 0:
                wandb.log({'Train/loss': loss.item(),
                           'Train/epoch': epoch,
                           'Train/step': idx})



        # Printing the training loss
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {sum(train_losses) / len(train_losses)}")
        wandb.log({'Train/loss per epoch': sum(train_losses) / len(train_losses),
                   'Train/epoch': epoch,
                   'Train/step': idx})


        #validation
        print('Valditaion')
        eval_losses = []
        eval_pds, eval_gts = [],[]
        #Stop gradient update in evalatio mode
        with torch.no_grad():
            for idx_val, eval_batch in enumerate(eval_dataloader):
                # setting the model to eval mode
                model.eval()
                # testing the model on eval batch
                eval_data = eval_batch['vis_values']
                eval_data = eval_data.cuda()
                eval_output = model(eval_data)
                eval_gt = eval_batch['clip_idx'].cuda()

                #Saving batch losses for metric calculation
                eval_gts.append(torch.argmax(eval_gt, 1).item())
                eval_pds.append(torch.argmax(eval_output, 1).item())
                # calculating loss
                eval_loss = loss_fn(eval_output, eval_gt)
                eval_losses.append(eval_loss.item())

                #logging results to wandb
                if idx_val % 100 == 0:
                    wandb.log({'Eval/loss': eval_loss,
                           'Eval/step': idx_val})


        #saving best model (with loswest loss)
        if sum(eval_losses) / len(eval_losses) < best_loss:
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()
                        }, model_path)
            best_loss = sum(eval_losses) / len(eval_losses)
            print('saving best model')
        print(f"Eval Loss: {best_loss}")
        #calculating accuracy
        for metric in calc_metrix(eval_pds, eval_gts).items():
            wandb.log({f'Eval/{metric[0]}': metric[1]})

    # Test
    test_losses = []
    test_pds, test_gts = [], []
    print('Testing')
    test_model = model
    checkpoint = torch.load(model_path)
    test_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    test_model.to(device)
    with torch.no_grad():
        for idx_test, test_batch in enumerate(test_dataloader):
            # setting the model to eval mode
            test_model.eval()
            # testing the model on test batch
            test_data = test_batch['vis_values']
            test_data = test_data.cuda()
            test_output = test_model(test_data)
            test_gt = test_batch['clip_idx'].cuda()

            # Saving batch losses for metric calculation
            test_gts.append(torch.argmax(test_gt, 1).item())
            test_pds.append(torch.argmax(test_output, 1).item())
            # calculating loss
            test_loss = loss_fn(test_output, test_gt)
            test_losses.append(test_loss.item())

    #calc and log metrics to wandb
    print(f"Test Loss: {sum(test_losses) / len(test_losses)}")
    wandb.log({'Test/loss': sum(test_losses) / len(test_losses)})
    for metric in calc_metrix(test_pds, test_gts, True).items():
        wandb.log({f'Test/{metric[0]}': metric[1]})


#Loop for training the model on all nets and fine tuning hyperparameters
#Net parameters
NET_list = ['Default_pCunPCC']#todo
NET_indexes = [1,2,3,4,5,6]#todo
H_list = ['RH', 'LH']#todo
for NET in NET_list:
    for NET_idx in NET_indexes:
        for H in H_list:
            for bs in [1,16,32]:
                for lr in [0.001, 0.0001, 0.00001]:
                    for num_heads in [1,4,8]:
                        print(f"Running training on {H}_{NET}_{NET_idx} for {epochs} epochs - Batch Size: {bs}, Learning Rate: {lr}")
                        wandb.login()
                        wandb.init(
                            # set the wandb project where this run will be logged
                            project="fmri_project",
                            group='encoder_nets',#todo
                            name=f'{NET}_{NET_idx}_{H}',#todo
                            # track hyperparameters and run metadata
                            config={
                                "learning_rate": lr, "epochs": epochs, "batch_size": bs, "dropout":0.1, "loss": 'CE', "optimizer": 'Adam',
                                'attention heads': num_heads,
                                "embedding dim": embedding_dim
                            }
                        )
                        #define model path for saving best model
                        model_path = f'models/best_model_{NET}_{NET_idx}_{H}.pth'#todo
                        #call train loop
                        train_loop()
                        #end run logging
                        wandb.finish()



# accuracy_fn = torchmetrics.Accuracy(task='multiclass', num_classes=15).to(device)
# precision_fn = torchmetrics.Precision(task='multiclass', num_classes=15).to(device)
# recall_fn = torchmetrics.Recall(task='multiclass', num_classes=15).to(device)
# f1_fn = torchmetrics.F1Score(task='multiclass', num_classes=15).to(device)
# metric_dict = {'Accuracy': accuracy_fn, 'Precision': precision_fn, 'Recall': recall_fn, 'F1-Score': f1_fn}
