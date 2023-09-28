import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim
from tqdm.auto import tqdm
from ..lab4.lab4 import Linear
from ..lab3.lab3 import LinearFunction


class MyNet(nn.Module):
    def __init__(self, num_classes):
        super(MyNet, self).__init__()

        self.pipe = nn.Sequential(
            # 3 x 227 x 227
            nn.Conv2d(in_channels=3, out_channels=96,
                      kernel_size=11, stride=4),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),

            # 96 x 55 x 55
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 96 x 27 x 27
            nn.Conv2d(in_channels=96, out_channels=256,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            # 256 x 27 x 27
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 256 x 13 x 13
            nn.Conv2d(in_channels=256, out_channels=384,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),

            # 384 x 13 x 13
            nn.Conv2d(in_channels=384, out_channels=384,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),

            # 384 x 13 x 13
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            # 256 x 13 x 13
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 256 x 6 x 6
            nn.Flatten(),

            # 9216
            nn.Dropout(0.5),
            Linear(in_features=9216, out_features=4096),
            nn.ReLU(inplace=True),

            # 4096
            nn.Dropout(0.5),
            Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),

            # 4096
            Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        return self.pipe(x)


def prepare_cifar10(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(227, 227), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = datasets.CIFAR10

    training_data = ds(
        root='data',
        train=True,
        download=True,
        transform=transform
    )

    test_data = ds(
        root='data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        training_data, batch_size=batch_size,
        shuffle=True, num_workers=2,
        drop_last=True
    )

    test_loader = DataLoader(
        test_data, batch_size=batch_size,
        shuffle=False, num_workers=2,
        drop_last=True
    )

    return train_loader, test_loader


def test_accuracy(y_, y, num_batchs=1):
    with torch.no_grad():
        _, predicted = torch.max(y_, 1)
        accuracy = (predicted == y).sum() / (num_batchs * y.numel())
    return accuracy


def eval_valid_set_accuracy(net, test_data, device):
    accuracy = torch.zeros((1,), device=device)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_data):
            x, y = inputs.to(device), labels.to(device)
            y_ = net(x)
            accuracy += (
                test_accuracy(y_, y, len(test_data))
            )

    return accuracy


def check_loss_value(loss):
    with torch.no_grad():
        if loss.isinf() or loss.isnan():
            raise ValueError('Invalid loss.')


def train_iteration(
    train_data, test_data,
    net, epochs,
    lr=0.005, device='cuda:0',
    bar_label='', prev_epochs=0,
    best_valid_accuracy=0,
    learning_time_validation=False,
    accuracy_history=False,
    make_checkpoints=False
):
    ret = {}

    if accuracy_history:
        train_accuracy = torch.zeros((epochs,), device=device)
        ret['train_accuracy'] = train_accuracy

    if learning_time_validation:
        valid_accuracy = torch.zeros((epochs,), device=device)
        ret['valid_accuracy'] = valid_accuracy

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    total_epochs = prev_epochs + epochs

    if not bar_label:
        bar_label = 'Train'

    for e in range(0, epochs):
        with tqdm(train_data, unit='batch') as epoch:
            bar_epoch = prev_epochs + e + 1

            epoch.set_description(
                f'{bar_label}. Epoch {bar_epoch}/{total_epochs}')

            for i, (inputs, labels) in enumerate(epoch):
                x, y = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                y_ = net(x)
                loss = criterion(y_, y)

                check_loss_value(loss)

                loss.backward()
                optimizer.step()

                if accuracy_history:
                    train_accuracy[e] += (
                        test_accuracy(y_, y, len(train_data))
                    )

                epoch.set_postfix(loss=loss.item())

            if learning_time_validation or make_checkpoints:
                valid_accuracy[e] = \
                    eval_valid_set_accuracy(net, test_data, device).item()

            if make_checkpoints:
                if best_valid_accuracy < valid_accuracy[e]:
                    best_valid_accuracy = valid_accuracy[e]
                    torch.save(net.state_dict(), 'best_model.pt')

    if make_checkpoints:
        ret['best_valid_accuracy'] = best_valid_accuracy

    if ret:
        return ret


def test(net, test_data, num_classes,
         device='cuda:0', bar_label='', use_best_model=False):

    t = torch.zeros(num_classes, device=device)
    f = torch.zeros(num_classes, device=device)

    if use_best_model:
        torch.save(net.state_dict(), 'model.pt')
        net.load_state_dict(torch.load('best_model.pt'))

    net.to(device)

    training_mode = net.training
    if training_mode:
        net.eval()

    if not bar_label:
        bar_label = 'Testing'

    with tqdm(test_data, unit='batch', desc=bar_label) as test:
        for inputs, labels in test:
            x, y = inputs.to(device), labels.to(device)

            _, predicted = torch.max(net(x).data, 1)

            t[y] += (predicted == y)
            f[y] += (predicted != y)

    acc = t / (t + f) * 100

    if training_mode:
        net.train()

    if use_best_model:
        net.load_state_dict(torch.load('model.pt'))

    return acc


if __name__ == '__main__':
    train_data, test_data = prepare_cifar10()

    LinearFunction.up_backend('hs/lab3/lab3.cu')
    net = MyNet(num_classes=10)

    train_iteration(train_data, test_data, net, epochs=16, lr=0.04)

    valid_accuracy_cls = test(net, test_data, num_classes=10)

    print('\nAccuracy (valid)')
    print(f'  min: {valid_accuracy_cls.min().item():.0f}%')
    print(f'  avg: {valid_accuracy_cls.mean().item():.0f}%')
    print(f'  max: {valid_accuracy_cls.max().item():.0f}%\n')
