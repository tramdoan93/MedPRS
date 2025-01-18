# %%writefile Main_TAKS.py
# -*- coding: utf-8 -*-
import re
import argparse
from datetime import datetime
import os # file management
import torch
from torch import nn, Tensor # neural network
# numerical matrix processing
import numpy as np
from numpy import ndarray

# data/parameter loading
import pandas as pd
# visualization
from tqdm import trange, tqdm
# transfomers
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

# code instruction
from typing import Union, List, Dict
# filter out warnings
import warnings
warnings.filterwarnings('ignore')


"""## Some useful functions"""


def sim_matrix(a, b, eps=1e-8):
    """
    Calculate cosine similarity between two matrices.
    Note: added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def batch2device(batch, device):
    """
    Transfer batch of training to GPU/CPU
    Args:
        batch: Dict[str, Tensor], represent for transformer input (input_ids, attention_mask)
        device: torch.device, GPU or CPU
    """
    for key, value in batch.items():
        batch[key] = batch[key].to(device)
    return batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# GPU accelerator
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

"""## Data loader"""


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        x = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        y = torch.tensor(self.labels[idx])
        return x, y

    def __len__(self):
        return len(self.labels)


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp



class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

"""## Sentence Embedder"""


class ModelForSE(nn.Module):
    def __init__(self, model_name_or_path, pooler_type):
        super(ModelForSE, self).__init__()
        '''
        Model for sentence embedding
        '''
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.pooler_type = pooler_type
        self.pooler = Pooler(self.pooler_type)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if self.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=return_dict,
        )
        if self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"]:
            pooler_output = self.pooler(attention_mask, outputs)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )
    def encode(self, sentences: Union[str, List[str]],
               batch_size: int = 8,
               show_progress_bar: bool = None,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None) -> Union[List[Tensor], ndarray, Tensor]:
        self.eval()

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False

        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.to(device)

        all_embeddings = []
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentence_batch = sentences[start_index: start_index+batch_size]
            features = tokenizer(sentence_batch,
                       padding='max_length',
                       truncation=True,
                       max_length=300,
                       return_tensors='pt').to(device)

            with torch.no_grad():
                out_features = self.forward(**features)
                embeddings = []
                # gather the embedding vectors
                for row in out_features.pooler_output:
                    embeddings.append(row.cpu())
                all_embeddings.extend(embeddings)
        if convert_to_tensor:
            all_embeddings = torch.vstack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings

"""## Model for downstream task"""

class WithAim_Classifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(WithAim_Classifier, self).__init__()
        self.base_model = base_model
        self.linear1_1 = nn.Linear(768, 512)
        self.act1_1 = nn.ReLU()
        self.drop1_1 = nn.Dropout(0.1)

        self.linear2_1 = nn.Linear(768, 512)
        self.act2_1 = nn.ReLU()

        self.linear_main_1 = nn.Linear(512 + num_classes, num_classes)
        self.act_main_1 = nn.LogSoftmax(dim=1)

    def forward(self, inputs_tak, inputs_aims):
        '''
        Args:
            inputs_tak: (Dict) batch of TAK samples, shape as [bs, n_samples, encoding_dim]
            inputs_aims: (Tensor) batch of aims embeddings taken by cls tokens, shape as [bs, n_samples, hidden_size]
        '''
        output_tak = self.base_model(**inputs_tak)
        last_hidden = output_tak.last_hidden_state[:, 0, :]  # cls tokens
        x = self.linear1_1(last_hidden)
        x = self.act1_1(x)
        x = self.drop1_1(x)

        if inputs_aims is not None:  # Aims
            y = self.linear2_1(inputs_aims)
            y = self.act2_1(y)

            cosine_feats = sim_matrix(x, y)
            concat_feats = torch.cat((x, cosine_feats), dim=1)

            out = self.linear_main_1(concat_feats)
            out = self.act_main_1(out)

            return out
        else:
            return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning script with LoRA and early stopping")
    parser.add_argument("--working_path", type=str, default="./QLoRAPSR/", help="working path")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Pretrained model name")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the contrastive learning checkpoint",
                        default="./QLoRAPSR/checkpoint/saved_model/Epoch:09 QLoRa_CL_DistilRoberta_CL.pth")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--pooler_type", type=str, default='cls', help="Pooler type")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--num_epoch", type=int, default=10, help="number of epochs")
    parser.add_argument("--name", type=str, default=10, help="model name when save checkpoint")
    parser.add_argument("--folder", type=str, default=10, help="folder")
    args = parser.parse_args()

    # Run the main part of the script
    print(f"Using model: {args.model_name}")
    # print(f"Loading checkpoint from: {args.checkpoint_path}")
    print(f"Using batch size: {args.batch_size}")
    print(f"Using pooler type: {args.pooler_type}")

    """# Data preparation"""

    working_path = args.working_path

    data_train = pd.read_csv(working_path + "data/preprocessed_data/01_train.csv", encoding="ISO-8859-1")
    data_validate = pd.read_csv(working_path + "data/preprocessed_data/01_validate.csv", encoding="ISO-8859-1")
    data_test = pd.read_csv(working_path + "data/preprocessed_data/01_test.csv", encoding="ISO-8859-1")
    data_aims = pd.read_csv(working_path + "data/preprocessed_data/01_aims.csv", encoding="ISO-8859-1")

    data_train.fillna("", inplace=True)
    data_validate.fillna("", inplace=True)
    data_test.fillna("", inplace=True)
    data_aims.fillna("", inplace=True)

    n_classes = len(data_aims)

    """## Feature selection"""
    print("Feature selection")
    X_train = (
            data_train['Title']
            + " " + data_train['Abstract']
            + " " + data_train['Keywords']
    ).tolist()
    X_valid = (
            data_validate['Title']
            + " " + data_validate['Abstract']
            + " " + data_validate['Keywords']
    ).tolist()
    X_test = (
            data_test['Title']
            + " " + data_test['Abstract']
            + " " + data_test['Keywords']
    ).tolist()

    X_aims = data_aims["Aims"].tolist()

    Y_train = data_train['Label'].tolist()
    Y_validate = data_validate['Label'].tolist()
    Y_test = data_test['Label'].tolist()

    """## Tokenization"""
    print("Tokenization")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_encodings = tokenizer(
        X_train,
        truncation=True,
        padding="max_length",
        max_length=300,
        return_tensors="pt"
    )
    valid_encodings = tokenizer(
        X_valid,
        truncation=True,
        padding="max_length",
        max_length=300,
        return_tensors="pt"
    )
    test_encodings = tokenizer(
        X_test,
        truncation=True,
        padding="max_length",
        max_length=300,
        return_tensors="pt"
    )

    # Dataset
    train_dataset = Dataset(train_encodings, Y_train)
    valid_dataset = Dataset(valid_encodings, Y_validate)
    test_dataset = Dataset(test_encodings, Y_test)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False)
    print("Finish encoding data")
    """# Model definition"""
    """## Load fine-tuned LM"""

    # # Fine-tuned LM checkpoint (by contrastive learning)
    checkpoint_cl_finetuned = torch.load(args.checkpoint_path, map_location=device,
                                         weights_only=True)  # ensure device for loading

    model_args = {
        'model_name_or_path': args.model_name,
        'pooler_type': args.pooler_type
    }
    base_model = ModelForSE(**model_args)
    base_model.load_state_dict(checkpoint_cl_finetuned["model_state_dict"], strict=False)

    model = WithAim_Classifier(base_model, n_classes)
    model.to(device)

    """# Training

    Firstly, we encode Aims&Scopes into the embedding features as external features for training
    """

    aims_embeddings = base_model.encode(X_aims, show_progress_bar=True, convert_to_tensor=True)
    if torch.cuda.is_available():
        aims_embeddings = aims_embeddings.cuda()

    """## Optimizer and Loss function"""

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)

    # Loss function
    loss_fn = nn.NLLLoss().to(device)

    """## Training settings"""

    max_epochs = args.num_epoch
    topks = [1, 3, 5, 10]
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc@k": [],
        "val_acc@k": [],
    }
    min_valid_loss = np.inf
    patience = 3  # Early stopping patience
    trigger_times = 0

    def get_latest_checkpoint(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if not checkpoints:
            return None  # No checkpoints found

        def extract_epoch(filename):
            match = re.search(r'Epoch:(\d+)', filename)
            return int(match.group(1)) if match else -1  # Default to -1 if no epoch found

        sorted_checkpoints = sorted(checkpoints, key=extract_epoch, reverse=True)
        return sorted_checkpoints[0] if sorted_checkpoints else None

    checkpoint_dir = args.working_path + args.folder + "/weight/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_name = get_latest_checkpoint(checkpoint_dir)

    if checkpoint_name:
        print(f"Latest checkpoint found: {checkpoint_name}")
        try:
            checkpoint_cl = torch.load(os.path.join(checkpoint_dir, checkpoint_name), map_location=device,
                                       weights_only=True)  # ensure device for loading
            model.load_state_dict(checkpoint_cl["model_state_dict"], strict=False)
            optimizer.load_state_dict(checkpoint_cl['optimizer_state_dict'])
            history = checkpoint_cl['history']
            epoch = checkpoint_cl["epoch"]
            train_loss = history["train_loss"][-1]
            min_valid_loss = history["val_loss"][-1]
            print("Model loaded successfully.")
            print("\tTraining loss: {}".format(train_loss))
            print("\tValidating loss: {}".format(min_valid_loss))
            print("\n")
            for k in topks:
                print("\tTrain accuracy@{}: {}".format(k, history["train_acc@k"][-1][k]))
            print("\n")
            for k in topks:
                print("\tValidate accuracy@{}: {}".format(k, history["val_acc@k"][-1][k]))
            epoch += 1
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch.")
            epoch = 0
            min_valid_loss = np.inf
    else:
        epoch = 0
        print("No checkpoints found. Starting training from scratch.")
        min_valid_loss = np.inf

    # Create the result file
    result_file_path = os.path.join(checkpoint_dir, "results.txt")

    # Check if the results file exists
    if os.path.exists(result_file_path):
        # If it exists, open in append mode ('a')
        with open(result_file_path, "a") as f:
            f.write(f"Resuming training on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    else:
        # If it doesn't exist, open in write mode ('w') and write the headers
        with open(result_file_path, "w") as f:
            f.write(f"Training started on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Name: {args.model_name}\n")
            f.write(f"Checkpoint Path: {args.working_path}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Pooler type: {args.pooler_type}\n")
            f.write(f"================================================\n")
            f.write(
                f"Epoch,Train Loss,Val Loss,Train Acc@1,Train Acc@3,Train Acc@5,Train Acc@10,Val Acc@1,Val Acc@3,Val Acc@5,Val Acc@10\n")

    """## Training loop"""

    for epoch in range(epoch, max_epochs):
        train_loss = 0.0
        train_loop = tqdm(train_loader, leave=True,
                          desc=f"Epoch: {epoch} - lr: {optimizer.param_groups[0]['lr']}, Training")  # removed format and added desc to tqdm
        batch_train_accuracy = {k: 0 for k in topks}
        batch_valid_accuracy = {k: 0 for k in topks}
        num_correct_at_k = {
            "train": {k: 0 for k in topks},
            "val": {k: 0 for k in topks}
        }
        # Training
        model.train()

        for features, labels in train_loop:

            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                features, labels = batch2device(features, device), labels.to(device)
            # forward pass
            logits = model(features, aims_embeddings)
            # Clear the gradients
            optimizer.zero_grad()
            # Find the Loss
            loss = loss_fn(logits, labels)
            # Calculate gradients
            loss.backward()
            # Update Weights
            optimizer.step()
            # Calculate accuracy
            probs_des = torch.argsort(torch.exp(logits), axis=1, descending=True)
            for k in topks:
                batch_num_correct = 0
                nPoints = len(labels)
                for i in range(nPoints):
                    if labels[i] in probs_des[i, 0:k]:
                        batch_num_correct += 1
                        num_correct_at_k["train"][
                            k] += 1  # globally counting number of correct at each k's for whole valid set
                batch_train_accuracy[k] = batch_num_correct / nPoints
            # Calculate Loss
            train_loss += loss.item()
            train_loop.set_postfix(train_loss=loss.item(),
                                   top01=batch_train_accuracy[1],
                                   top03=batch_train_accuracy[3],
                                   top05=batch_train_accuracy[5],
                                   top10=batch_train_accuracy[10])
        train_loss = train_loss / len(train_loader)
        history["train_loss"].append(train_loss)
        history["train_acc@k"].append(
            {k: val / len(X_train) for k, val in num_correct_at_k["train"].items()}
        )

        # Validation
        valid_loss = 0.0
        valid_loop = tqdm(valid_loader, leave=True,
                          desc=f"Epoch: {epoch} - lr: {optimizer.param_groups[0]['lr']}, Validating")  # removed format and added desc to tqdm
        with torch.no_grad():
            model.eval()
            # Transfer Data to GPU if available
            for features, labels in valid_loop:

                if torch.cuda.is_available():
                    features, labels = batch2device(features, device), labels.to(device)
                # Forward pass
                logits = model(features, aims_embeddings)

                # Find the Loss
                loss = loss_fn(logits, labels)
                # Calculate accuracy
                probs_des = torch.argsort(torch.exp(logits), axis=1, descending=True)
                for k in topks:
                    num_correct = 0
                    nPoints = len(labels)
                    for i in range(nPoints):
                        if labels[i] in probs_des[i, 0:k]:
                            num_correct += 1
                            num_correct_at_k["val"][
                                k] += 1  # globally counting number of correct at each k's for whole valid set
                    batch_valid_accuracy[k] = num_correct / nPoints
                # Calculate Loss
                valid_loss += loss.item()
                valid_loop.set_postfix(val_loss=loss.item(),
                                       val_top01=batch_valid_accuracy[1],
                                       val_top03=batch_valid_accuracy[3],
                                       val_top05=batch_valid_accuracy[5],
                                       val_top10=batch_valid_accuracy[10])
            valid_loss = valid_loss / len(valid_loader)
            history["val_loss"].append(valid_loss)
            history["val_acc@k"].append(
                {k: val / len(X_valid) for k, val in num_correct_at_k["val"].items()}
            )
            print(f'>> Epoch {epoch} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')
            lr_scheduler.step()

            # Write the information to the result file
            with open(result_file_path, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{valid_loss:.6f},")
                f.write(f"{history['train_acc@k'][-1][1]:.4f},")
                f.write(f"{history['train_acc@k'][-1][3]:.4f},")
                f.write(f"{history['train_acc@k'][-1][5]:.4f},")
                f.write(f"{history['train_acc@k'][-1][10]:.4f},")
                f.write(f"{history['val_acc@k'][-1][1]:.4f},")
                f.write(f"{history['val_acc@k'][-1][3]:.4f},")
                f.write(f"{history['val_acc@k'][-1][5]:.4f},")
                f.write(f"{history['val_acc@k'][-1][10]:.4f}\n")

            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                trigger_times = 0
                # Saving State Dict
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'history': history,
                        'epoch': epoch
                    }, checkpoint_dir + args.name + "Epoch:{:0>2} QLoRa_CL_DistilRoberta.pth".format(epoch)
                )
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping triggered")
                    break  # break the training process early if no validation loss improvement

    # load checkpoint and testing
    checkpoints = os.listdir(checkpoint_dir)
    checkpoint_name = checkpoints[-1]
    checkpoint = torch.load(checkpoint_dir + checkpoint_name, map_location=device)  # ensure device for loading

    model = WithAim_Classifier(base_model, n_classes)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False )
    model.to(device)

    history = checkpoint['history']

    # Loss function
    loss_fn = nn.NLLLoss().to(device)

    # Test
    topks = [1, 3, 5, 10]
    num_correct_at_k = {}
    test_loop = tqdm(test_loader, leave=True, desc='Testing...')  # removed format and added desc to tqdm
    num_correct_at_k["test"] = {k: 0 for k in topks}
    batch_test_accuracy = {k: [] for k in topks}
    history["test_acc@k"] = []
    history["test_loss"] = []
    test_loss = 0.0

    with torch.no_grad():
        model.eval()
        for features, labels in test_loop:
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                features, labels = batch2device(features, device), labels.to(device)
            logits = model(features, aims_embeddings)
            # Find the Loss
            loss = loss_fn(logits, labels)
            # Calculate accuracy
            probs_des = torch.argsort(torch.exp(logits), axis=1, descending=True)
            for k in topks:
                num_correct = 0
                nPoints = len(labels)
                for i in range(nPoints):
                    if labels[i] in probs_des[i, 0:k]:
                        num_correct += 1
                        num_correct_at_k["test"][
                            k] += 1  # globally counting number of correct at each k's for whole valid set
                batch_test_accuracy[k] = num_correct / nPoints
            # Calculate Loss
            test_loss += loss.item()

            test_loop.set_postfix(test_loss=loss.item(),
                                  test_top01=batch_test_accuracy[1],
                                  test_top03=batch_test_accuracy[3],
                                  test_top05=batch_test_accuracy[5],
                                  test_top10=batch_test_accuracy[10])

        test_loss = test_loss / len(test_loader)
        history["test_loss"].append(test_loss)
        history["test_acc@k"].append(
            {k: val / len(X_test) for k, val in num_correct_at_k["test"].items()}
        )

    # Add test results to result.txt
    with open(result_file_path, "a") as f:
        f.write(f"================================================\n")
        f.write(f"Testing Loss: {history['test_loss'][-1]:.6f}\n")
        f.write(f"Testing Acc@1: {history['test_acc@k'][-1][1]:.4f}\n")
        f.write(f"Testing Acc@3: {history['test_acc@k'][-1][3]:.4f}\n")
        f.write(f"Testing Acc@5: {history['test_acc@k'][-1][5]:.4f}\n")
        f.write(f"Testing Acc@10: {history['test_acc@k'][-1][10]:.4f}\n")
        f.write(f"================================================\n")

