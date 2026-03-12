from  torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
from torchvision.transforms import Compose,ToTensor,Resize,RandomAffine,ColorJitter
from src import config

# because of this data structure, ImageFolder can be used to make it shorter
# data/raw/
#     Training/
#         glioma/
#         meningioma/
#         notumor/
#         pituitary/
#     Testing/
#         glioma/
#         meningioma/
#         notumor/
#         pituitary/

# custom dataset class
class BrainTumorMRIDataset(Dataset):
    def __init__(self,root,train=True,transform=None):
        if train:
            mode = "Training"
        else:
            mode = "Testing"

        root = os.path.join(root,mode)
        self.transform = transform
        self.categories = config.categories

        self.images_paths = []
        self.labels = []

        for idx, category in enumerate(self.categories):
            data_file_path = os.path.join(root,category)
            for file_name in os.listdir(data_file_path):
                file_path = os.path.join(data_file_path,file_name)
                self.images_paths.append(file_path)
                self.labels.append(idx)

    def __len__(self):
        return  len(self.labels)

    def __getitem__(self, item):
        image_path = self.images_paths[item]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self.labels[item]
        return image, label

def BrainTumorMRIDataloafers(data_dir=config.data_dir, batch_size=config.batch_size,image_size=config.image_size):
    # transfrom
    train_transform = Compose([
        RandomAffine(
            degrees=(-5, 5),
            translate=(0.05, 0.05),
            scale=(0.85, 1.15),
            shear=5
        ),
        Resize((image_size,image_size)),
        ToTensor()
    ])
    test_transform = Compose([
        Resize((image_size,image_size)),
        ToTensor()
    ])

    #dataset
    train_dataset = BrainTumorMRIDataset(root=data_dir,train=True,transform=train_transform)
    test_dataset = BrainTumorMRIDataset(root=data_dir,train=False,transform=test_transform)

    #dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    return train_dataloader,test_dataloader




if __name__ == '__main__':
    train_loader, test_loader= BrainTumorMRIDataloafers()

    for images, labels in train_loader:
        print(images.shape)
        print(labels)
        break
