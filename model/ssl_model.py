import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
import torch.nn.init as init
from loss_functions.info_nce import InfoNCE

# SimCLRDataset class - creates pairs of images (Anchor and Positive) with transformations
class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        """
               Initialize the dataset with the base dataset and transformations.

               :param dataset: The base dataset (e.g., STL10).
               :param transform: The transformations to be applied to the images.
        """
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        anchor = self.transform(img)
        positive = self.transform(img)
        return anchor, positive

    def __len__(self):
        return len(self.dataset)


def build_classifier_STL10(lr):
    """
       Builds the SimCLR model with ResNet50 backbone and a projection head, sets up
       the loss function and optimizer, and prepares the DataLoaders for STL10 dataset.

       :param lr: Learning rate for the optimizer.
       :return: Train loader, validation loader, model, loss function, optimizer, and class names.
       """
    # Define the ResNet50 model with pretrained weights
    resnet = models.resnet50(pretrained=True)

    num_classes = 10  # STL10 includes 10 classes

    # Get the number of input features for the final fully connected layer
    in_features = resnet.fc.in_features

    # Replace the fully connected layer with an Identity layer
    resnet.fc = nn.Identity()

    # Add a projection head (MLP)
    projection_head = nn.Sequential(
        nn.Linear(in_features, 128),  # Hidden dimension
        nn.ReLU(),
        nn.Linear(128, 128)  # Output dimension
    )

    # Combine ResNet and Projection Head
    model = nn.Sequential(resnet, projection_head)

    # Define the loss function and optimizer
    criterion = InfoNCE(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define the data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(96),  # STL10 images are 96x96
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the STL10 dataset
    # Replace train_dataset with SimCLRDataset
    train_dataset = SimCLRDataset(STL10(root='./data', split='unlabeled', download=True), data_transforms['train'])
    val_dataset = SimCLRDataset(STL10(root='./data', split='test', download=True), transform=data_transforms['train'])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)



    print(f'Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val')

    return train_loader, val_loader, model, criterion, optimizer







"""

def visualize_augmentations(dataloader,save_dir, num_images=5):
   
    os.makedirs(save_dir, exist_ok=True)

    # קבלת אצווה מה-DataLoader
    anchor_batch, positive_batch = next(iter(dataloader))
    print("Anchor shape:", anchor_batch.shape)
    print("Positive shape:", positive_batch.shape)


    for i in range(num_images):
        anchor = anchor_batch[i]
        positive = positive_batch[i]

        # המרת טנזור לתמונה
        anchor_img = F.to_pil_image(anchor)
        positive_img = F.to_pil_image(positive)

        # שמירת התמונות כקבצים
        anchor_path = os.path.join(save_dir, f"anchor_{i}.png")
        positive_path = os.path.join(save_dir, f"positive_{i}.png")
        anchor_img.save(anchor_path)
        positive_img.save(positive_path)

        print(f"Saved anchor image to: {anchor_path}")
        print(f"Saved positive image to: {positive_path}")


# Define the data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(96),  # STL10 images are 96x96
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
train_dataset = SimCLRDataset(STL10(root='./data', split='unlabeled', download=True), data_transforms['train'])
# יצירת DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
save_directory = "./output_images2"
# הצגת תמונות לאחר הטרנספורמציות
visualize_augmentations(train_loader,save_directory,num_images=5)
"""