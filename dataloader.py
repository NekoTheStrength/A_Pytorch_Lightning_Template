from torch.utils.data import Dataset, DataLoader
from utils import prepare_data


class TrainDataset(Dataset):
    def __init__(self, datalist):
        """The dataset for Neural Network.

        :return: A dataset object
        """
        self.img = datalist[0]
        self.label = datalist[1]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        label = self.label[idx]
        img = self.img[idx]
        return img, label


def loader(batch_size=32, num_workers=8):
    """Make dataloader for training, validation and testing."""
    if num_workers > 0:
        persistent_workers = True
    else:
        persistent_workers = False
    # transform = Compose([ToTensor()])
    train, val, test = prepare_data()
    train = TrainDataset(train)
    val = TrainDataset(val)
    test = TrainDataset(test)
    train = DataLoader(
        train, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers, shuffle=True
    )
    val = DataLoader(val, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers)
    test = DataLoader(test, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_workers)
    return train, val, test
