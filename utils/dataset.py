from torch.utils.data import DataLoader
from torchvision import transforms, datasets

batch_size = 64
num_workers = 8
download = False

def get_mnist(data_dir='data/'):
    trans = transforms.ToTensor()
    
    dataset_train = datasets.MNIST(root=data_dir, train=True,
                            transform=trans,
                            download=download)
    dataset_test = datasets.MNIST(root=data_dir, train=False,
                            transform=trans,
                            download=download)

    train_data = DataLoader(dataset=dataset_train,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True)
    test_data = DataLoader(dataset=dataset_test,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)

    return train_data, test_data


def get_dataset(data_name):
    if data_name == 'mnist':
        train_data, test_data = get_mnist()

    return train_data, test_data

if __name__ == '__main__':
    print('mnist')
    data_dir = '../data'
    train_d, test_d = get_mnist(data_dir)
    print(next(iter(train_d))[0].shape)
    print(next(iter(test_d))[0].shape)
