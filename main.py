import torch
import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from model import Transformer
from tqdm import tqdm


class ToyDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


def generate_data(num_samples, vocab_size, sequence_len):
    src_data = []
    tgt_data = []
    for _ in range(num_samples):
        src_seq = np.random.randint(1, vocab_size - 1, sequence_len)
        tgt_seq = src_seq[::-1]
        src_data.append(src_seq)
        tgt_data.append(tgt_seq)
    return np.array(src_data), np.array(tgt_data)


def train(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for src, tgt in tqdm(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(
                output.view(-1, vocab_size), tgt[:, 1:].contiguous().view(-1)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in tqdm(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(
                output.view(-1, vocab_size), tgt[:, 1:].contiguous().view(-1)
            )
            total_loss += loss.item()

            if np.random.rand() < 0.01:
                pred = torch.argmax(output, dim=-1)
                print("Input: ", src[0].cpu().numpy())
                print("Target: ", tgt[0].cpu().numpy())
                print("Predicted: ", pred[0].cpu().numpy())
                print()

    print(f"Validation Loss: {total_loss / len(dataloader)}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequence_length = 8
    vocab_size = 10

    src_data, tgt_data = generate_data(10000, vocab_size, sequence_length)

    dataset = ToyDataset(src_data, tgt_data)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    model = Transformer(vocab_size, vocab_size).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    train(model, dataloader, optimizer, criterion, device)
    evaluate(model, dataloader, device)
