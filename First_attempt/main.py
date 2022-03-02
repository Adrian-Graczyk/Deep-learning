import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

# Train, Val split
train_data = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)

# model
model = nn.Sequential(
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(0.1), # if overfitting
    nn.Linear(64, 10)
)

# Define more flexibile model
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1)

    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h2 + h1)
        logits = self.l3(do)
        return logits
model = ResNet()

# Define my optimizer
params = model.parameters()
optimiser = optim.SGD(model.parameters(), lr=1e-2)

# Define loss
loss = nn.CrossEntropyLoss()

# Training and validation loops
nb_epochs = 5
for epoch in range(nb_epochs):
    losses = list()
    accuracies = list()
    model.train() # because of Dropout
    for batch in train_loader:
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)

        # 1 - forward
        l = model(x)  # l: logits

        # 2 compute the objective function
        J = loss(l, y)

        # 3 cleaning the gradients
        model.zero_grad()
        # optimizer.zero_grad()
        # params.grad._zero()

        # 4 accumulate the partial derivatives of J with respect to parameters
        J.backward()
        # params.grad.add_(dJ/dparams)

        # 5 step in the opposite direction of the gradient
        optimiser.step()
        # with torch.no_grad(): params = params - eta * params.grad

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch + 1}', end=', ')
    print(f'training loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'training accuracy: {torch.tensor(accuracies).mean():.2f}')

    losses = list()
    accuracies = list()
    model.eval()
    for batch in val_loader:
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)

        # 1 - forward
        with torch.no_grad():
            l = model(x)  # l: logits

        # 2 compute the objective function
        J = loss(l, y)
        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch + 1}', end=', ')
    print(f'validation loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'validation accuracy: {torch.tensor(accuracies).mean():.2f}')
