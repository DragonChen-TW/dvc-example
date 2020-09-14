import time, os
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
#
from utils.dataset import get_dataset
from utils.models import LeNet

def train(model, data, epoch, criterion, optimizer, device):
    model.train()
    print('==========Train Epoch {}=========='.format(epoch))
    loss_list = []
    acc_count = 0

    for i, (image, label) in tqdm(enumerate(data), total=len(data)):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        score = model(image) # predict the label
        loss = criterion(score, label) # calculate error
        loss_list.append(loss.item())
        pred = torch.argmax(score, dim=1)
        correct = pred.eq(label)
        acc_count += correct.sum().item()
        
        loss.backward()  # back-propagation
        optimizer.step() # gradient descent

    acc = acc_count / (len(data) * data.batch_size) * 100
    return sum(loss_list) / len(loss_list), acc

def test(model, data, criterion, device):
    model.eval()
    loss_list = []

    acc_count = 0
    for i, (image, label) in tqdm(enumerate(data), total=len(data)):
        image = image.to(device)
        label = label.to(device)

        score = model(image)
        loss = criterion(score, label)
        loss_list.append(loss.item())

        pred = torch.argmax(score, dim=1)
        correct = pred.eq(label)
        acc_count += correct.sum().item()

    acc = acc_count / (len(data) * data.batch_size) * 100
    print('----------Acc: {}%----------'.format(acc))
    return sum(loss_list) / len(loss_list), acc

if __name__ == '__main__':
    # == Setting ==
    device = torch.device('cpu')
    print('Using', device)
    
    # == Data ==
    data_name = 'mnist'
    print('Data using: {}'.format(data_name))
    train_data, test_data = get_dataset(data_name)

    # == Model ==
    model = LeNet()
    model = model.to(device)

    # == optimizer ==
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # == Main Loop ==
    max_acc = 0
    max_epoch = 30
    scheduler = StepLR(optimizer=optimizer, step_size=10)

    # first epoch
    # test(model, test_data, device=device)
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(1, max_epoch + 1):
        t = time.time()
        train_loss, train_acc = train(model, train_data, epoch, criterion, optimizer, device=device)
        test_loss, test_acc = test(model, test_data, criterion, device=device)
        scheduler.step()

        print('train loss:', train_loss, 'test loss:', test_loss)
        print('train acc:', train_acc, 'test acc:', test_acc)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print('Epoch {} cost {} sec'.format(epoch, time.time() - t))
        t = time.time()
        
        # save model
        torch.save(model.state_dict(), 'ckpts/LeNet_e{:02}.pt'.format(epoch))

        plt.plot(range(1, epoch + 1), train_loss_list)
        plt.plot(range(1, epoch + 1), test_loss_list, color='r')
        plt.legend(['train_loss', 'test_loss'])
        plt.savefig('loss.png')
        plt.cla()

        plt.plot(range(1, epoch + 1), train_acc_list)
        plt.plot(range(1, epoch + 1), test_acc_list, color='r')
        plt.ylim(0, 100)
        plt.legend(['train_acc', 'test_acc'])
        plt.savefig('acc.png')
        plt.cla()
