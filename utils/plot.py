import torch
from models import LeNet
from dataset import get_dataset
# 
import matplotlib.pyplot as plt

def plot_with_label(images, labels, true_labels, batch_size=16, rows=4, cols=4):
    images = images.squeeze() # 16x28x28
    plt.figure(figsize=(10, 12))
#     plt.tight_layout()
    
    for i in range(batch_size):
        lbl = labels[i].item()
        true_lbl = true_labels[i].item()
        plt.subplot(rows, cols, i + 1) # 1 ~ 4
        plt.imshow(images[i,:,:].numpy(), cmap='gray')
        plt.title('{} -> {}'.format(true_lbl, lbl))
        plt.axis('off')
    plt.savefig('../plot.png')
    # plt.show()

if __name__ == '__main__':
    model = LeNet()
    
    epoch = 12
    ckpt = torch.load('../ckpts/LeNet_e{:02}.pt'.format(epoch))
    model.load_state_dict(ckpt)
    
    _, dataset = get_dataset('mnist')
    imgs, true_labels = next(iter(dataset))
    labels = model(imgs)
    labels = torch.argmax(labels, dim=1)
    batch_size = dataset.batch_size
    
    plot_with_label(imgs, labels, true_labels,
                    batch_size, batch_size ** 0.5, batch_size ** 0.5)