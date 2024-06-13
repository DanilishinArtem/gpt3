import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from binaryClassification.trainer import Trainer
from binaryClassification.dataloader import Dataset
from transformers import GPTNeoForSequenceClassification
from binaryClassification.config import Config
from faultInjector.hookSetter import HookSetter as Hook

config = Config()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    
    model = GPTNeoForSequenceClassification.from_pretrained(config.path, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id
    model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    Hook(model)
    
    dataset = Dataset()
    
    # Используем DistributedSampler для train_loader
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset.train_loader.dataset,
        num_replicas=world_size,
        rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        dataset.train_loader.dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )

    trainer = Trainer(model)
    trainer.train(train_loader)
    # trainer.evaluate(dataset.test_loader)
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size,
             join=True)
