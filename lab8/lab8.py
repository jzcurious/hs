import torch
from torch import nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler
from ..lab3.lab3 import LinearFunction
from ..lab5.lab5 import (
    MyNet,
    prepare_cifar10,
    test,
    eval_valid_set_accuracy,
    test_accuracy
)


def train_iteration(
    train_data, test_data,
    net, epochs,
    lr=0.005, device='cuda:0',
    bar_label='', prev_epochs=0,
    best_valid_accuracy=-1,
    learning_time_validation=False,
    accuracy_history=False,
    make_checkpoints=False,
    use_mixed=True
):
    ret = {}

    if accuracy_history:
        train_accuracy = torch.zeros((epochs,), device=device)
        ret['train_accuracy'] = train_accuracy

    if learning_time_validation:
        valid_accuracy = torch.zeros((epochs,), device=device)
        ret['valid_accuracy'] = valid_accuracy

    net.to(device)

    grad_scaler = GradScaler()
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

                with torch.autocast(enabled=use_mixed,
                                    device_type='cuda', dtype=torch.float16):

                    y_ = net(x)
                    loss = criterion(y_, y)

                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

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


if __name__ == '__main__':
    LinearFunction.up_backend('hs/lab8/lab8_disp.cu')

    train_data, test_data = prepare_cifar10()

    net = MyNet(num_classes=10)

    train_iteration(train_data, test_data, net, epochs=1, lr=0.02)

    valid_accuracy_cls = test(net, test_data, num_classes=10)

    print('\nAccuracy (valid)')
    print(f'  min: {valid_accuracy_cls.min().item():.0f}%')
    print(f'  avg: {valid_accuracy_cls.mean().item():.0f}%')
    print(f'  max: {valid_accuracy_cls.max().item():.0f}%\n')
