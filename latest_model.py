import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from network import Network

from sklearn import metrics
from sklearn.model_selection import train_test_split

import numpy as np
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(self, n_epochs: int, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
    gpu_id: int) -> None:
        self.n_epochs = n_epochs
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
        # self.lr_lambda = lambda epoch: max(0.01, 0.003 - (0.003 - 0.01) * epoch / self.n_epochs)
        # self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,  lr_lambda=self.lr_lambda)
        self.save_every = 500
        self.model = DDP(model, device_ids=[self.gpu_id])
        self.loss_fn = nn.BCELoss()
        self.loss_avg = []
        self.val_avg = []
        self.accuracy_avg = []
        self.gradient_avg = []

    def _run_batch(self, data, targets):
        self.optimizer.zero_grad()
        output = self.model(data)
        targets = targets[:, None]
        loss = self.loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()
        grad = self.get_norm()
        return loss, grad

    def _run_epoch(self, epoch):
        loss_sum = 0
        gradient_sum = 0
        acc_sum = 0
        val_sum = 0

        self.train_loader.sampler.set_epoch(epoch)
        self.val_loader.sampler.set_epoch(epoch)
        s1 = len(self.train_loader)

        for data, targets in self.train_loader:
            data = data.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            info = self._run_batch(data, targets)
            loss_sum += info[0]
            gradient_sum += info[1]

        s2 = len(self.val_loader)
        with torch.no_grad():
            for data, targets in self.val_loader:
                output = self.model(data)
                targets = targets[:, None]
                val_sum += self.loss_fn(output.cpu(), targets.cpu())

                acc_sum += self.get_accuracy(targets, output)

        if self.gpu_id == 0:
            self.loss_avg.append(100 * loss_sum/s1)
            self.gradient_avg.append(100 * gradient_sum/s1)
            self.accuracy_avg.append(100 * acc_sum/s2)
            self.val_avg.append(100 * val_sum/s2)


    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoints/checkpoint_{0}.pt".format(epoch)
        torch.save(ckp, PATH)
        # print(f"Epoch {epoch} | Learning Rate {self.scheduler.get_last_lr()[0]} |Training checkpoint saved at {PATH}")

    def train(self):
        for epoch in range(self.n_epochs + 1):
            self._run_epoch(epoch)
            # self.scheduler.step()
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                np.save("logs/loss.npy", np.array(torch.tensor(self.loss_avg, device="cpu")))
                np.save("logs/accuracy.npy", np.array(torch.tensor(self.accuracy_avg, device="cpu")))
                np.save("logs/gradient.npy", np.array(torch.tensor(self.gradient_avg, device="cpu")))
                np.save("logs/validation.npy", np.array(torch.tensor(self.val_avg, device="cpu")))
        if self.gpu_id == 0:
            self.test()


    def test(self):
        s = len(test_loader)
        test_loss = 0
        test_sum = 0
        with torch.no_grad():
            for data, targets in test_loader:
                output = model(data)
                targets = targets[:, None]
                test_loss += self.loss_fn(output.cpu(), targets.cpu())
                test_sum += self.get_accuracy(targets, output)

        test_loss /= s
        test_sum /= s
        with open("logs/test.txt", "w") as file:
            file.write("Test Accuracy: {}".format(test_sum))
            file.write("Test Loss: {}".format(test_loss))


    def get_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            else:
                return 0
        return total_norm ** (1./2)

    def get_accuracy(self, y_true, y_prob):
        accuracy = metrics.accuracy_score(y_true.cpu(), y_prob.cpu() > 0.5)
        return accuracy


def load_data():
    # Loading Data
    inp = np.load("input_test.npy")
    opt = np.load("output_test.npy")

    # Normalization
    normalization = np.linalg.norm(inp, axis=2)
    max_norm = np.max(normalization, axis=1)
    inp_norm = inp/max_norm[:, None,None]

    # Convert to Pytorch Tensors
    inp_tensor = torch.tensor(inp_norm, dtype=torch.float32)
    opt_tensor = torch.tensor(opt, dtype=torch.float32)

    # Splitting Data into Training, Validation, and Test Sets
    data_train, data_temp, label_train, label_temp = train_test_split(inp_tensor, opt_tensor, test_size=0.3, random_state=42)
    val_inputs, data_test, val_outputs, label_test = train_test_split(data_temp, label_temp, test_size=0.66, random_state=42)

    train_dataset = TensorDataset(data_train, label_train)
    val_dataset = TensorDataset(val_inputs, val_outputs)
    test_dataset = TensorDataset(data_test, label_test)

    return train_dataset, val_dataset, test_dataset

def main(rank: int, world_size: int):
    n_epochs = 10000
    ddp_setup(rank, world_size)
    model = Network()
    train_dataset, val_dataset, test_dataset = load_data()
    train_loader = DataLoader(dataset=train_dataset, batch_size=5, pin_memory=True, shuffle=False, drop_last=True, sampler=DistributedSampler(train_dataset))
    val_loader = DataLoader(dataset=train_dataset, batch_size=5, pin_memory=True, shuffle=False, drop_last=True, sampler=DistributedSampler(val_dataset))
    test_loader = DataLoader(dataset=train_dataset, batch_size=5, pin_memory=True, shuffle=False, drop_last=True, sampler=DistributedSampler(test_dataset))
    trainer = Trainer(n_epochs, model, train_loader, val_loader, test_loader, rank)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    mp.spawn(main, args=(world_size,), nprocs=world_size)
