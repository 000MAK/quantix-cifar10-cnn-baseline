"""Train a compact CNN baseline using torchvision's online CIFAR-10 loader."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def validate_limit(value: int, name="limit") -> int:
    if int(value)<1: raise ValueError(f"{name} must be at least 1")
    return int(value)

def main(output_dir="outputs", epochs=2, batch_size=128, seed=42, max_train=None, max_test=None):
    validate_limit(epochs,"epochs"); validate_limit(batch_size,"batch_size")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    torch.manual_seed(seed)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
    train_set=datasets.CIFAR10(root="data",train=True,download=True,transform=transform)
    test_set=datasets.CIFAR10(root="data",train=False,download=True,transform=transform)
    if max_train: train_set=Subset(train_set,range(min(validate_limit(max_train,"max_train"),len(train_set))))
    if max_test: test_set=Subset(test_set,range(min(validate_limit(max_test,"max_test"),len(test_set))))
    train=DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0)
    test=DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=0)
    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__(); self.layers=nn.Sequential(
              nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
              nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
              nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),
              nn.Flatten(),nn.Linear(128,10))
        def forward(self,x): return self.layers(x)
    model=SmallCNN(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3); loss_fn=nn.CrossEntropyLoss()
    for ep in range(validate_limit(epochs,"epochs")):
        model.train(); total=0
        for x,y in train:
            optimizer.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); optimizer.step(); total+=loss.item()*len(y)
        print(f"epoch={ep+1} train_loss={total/max(len(train_set),1):.4f}")
    model.eval(); correct=seen=0
    with torch.no_grad():
        for x,y in test:
            pred=model(x).argmax(1); correct+=int((pred==y).sum()); seen+=len(y)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),out/"cifar10_cnn.pt")
    report={"dataset":"CIFAR-10","train_rows":len(train_set),"test_rows":seen,"accuracy":correct/max(seen,1),"epochs":epochs,"seed":seed,
      "warning":"Short training runs are smoke-test baselines, not competitive benchmark results."}
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); return report

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir",default="outputs"); p.add_argument("--epochs",type=int,default=2); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--seed",type=int,default=42); p.add_argument("--max-train",type=int); p.add_argument("--max-test",type=int)
    a=p.parse_args(); print(json.dumps(main(a.output_dir,a.epochs,a.batch_size,a.seed,a.max_train,a.max_test),indent=2))
