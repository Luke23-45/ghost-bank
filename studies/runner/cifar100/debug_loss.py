import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import torch
import pytorch_lightning as pl
from src.methods.uniform_herding.method import UniformHerdingMethod
from src.training.pl_module import GhostBankLightningModule
from src.bank.strategies.herding import HerdingReplayBank
from src.models.resnet.model import ResNet

def test():
    model = ResNet(num_classes=10)
    bank = HerdingReplayBank(num_classes=10, total_budget=2000, seed=42)
    bank.start_task()
    method = UniformHerdingMethod(retrieval_budget=64, warmup_steps=0)
    
    pl_module = GhostBankLightningModule(model=model, method=method, bank=bank)
    
    # Mock trainer
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu")
    pl_module.trainer = trainer
    
    print(f"Global step: {pl_module.global_step}")
    
    # Add fake data to bank
    bank._selected = {0: [(torch.zeros(3, 32, 32, dtype=torch.uint8), 0)]}
    
    batch = (torch.zeros(2, 3, 32, 32), torch.tensor([0, 0]))
    try:
        loss = method.compute_loss(batch, pl_module, bank=bank)
        print(f"Loss computed! Loss: {loss}")
    except Exception as e:
        print(f"Error computing loss: {e}")

if __name__ == '__main__':
    test()
