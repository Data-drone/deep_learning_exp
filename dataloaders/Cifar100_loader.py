from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
from torchvision.datasets import CIFAR100

class CIFAR100DataModule(LightningDataModule):
    def __init__(self, data_dir: str = "s3a://datasets/cifar100",
                batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
    
    def prepare_data(self):
        CIFAR100(root=self.data_dir, train=True, download=True)
        CIFAR100(root=self.data_dir, train=False, download=True)
    
    def setup(self):
        self.cifar100_test = CIFAR100(root=self.data_dir, 
                                       train=False, download=False)
        self.cifar100_train = CIFAR100(root=self.data_dir, 
                                       train=True, download=False)
        
        
    def train_dataloader(self):
        return DataLoader(
            self.cifar100_train,
            batch_size=self.batch_size
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.cifar100_test,
            batch_size=self.batch_size
        )
        
    