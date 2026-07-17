import os, copy, torch, timm, torch.nn as nn, numpy as np, pandas as pd
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters(): p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            msd, esd = model.state_dict(), self.ema.state_dict()
            for k in esd: esd[k].copy_(esd[k] * self.decay + msd[k] * (1. - self.decay))

def main():
    SAVE_DIR = './exp2_enhanced'
    os.makedirs(SAVE_DIR, exist_ok=True)
    LOG_FILE = f"{SAVE_DIR}/train_log.csv"
    
    tf_t = transforms.Compose([transforms.RandomResizedCrop(224), transforms.TrivialAugmentWide(), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tf_v = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    train_set = datasets.ImageFolder('./data/train', tf_t)
    cw = compute_class_weight('balanced', classes=np.unique(train_set.targets), y=train_set.targets)
    train_loader = DataLoader(train_set, batch_size=64, sampler=WeightedRandomSampler([cw[t] for t in train_set.targets], len(train_set)), num_workers=8, pin_memory=True)
    val_loader = DataLoader(datasets.ImageFolder('./data/test', tf_v), batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=7).cuda()
    if os.path.exists('./weights/vit_base_patch16_224.pth'):
        sd = torch.load('./weights/vit_base_patch16_224.pth', map_location='cpu'); sd = sd['model'] if 'model' in sd else sd
        sd = {k: v for k, v in sd.items() if "head" not in k}; model.load_state_dict(sd, strict=False)

    ema = ModelEMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    best_acc = 0.0

    for epoch in range(80):
        model.train()
        train_loss, train_corrects = 0.0, 0
        for imgs, lbls in tqdm(train_loader, desc=f"Exp2 E{epoch+1}"):
            imgs, lbls = imgs.cuda(), lbls.cuda()
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(imgs); loss = criterion(out, lbls)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            ema.update(model); train_loss += loss.item() * imgs.size(0); train_corrects += torch.sum(out.argmax(1) == lbls.data)

        ema.ema.eval(); preds, targets, val_loss = [], [], 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.cuda(), lbls.cuda()
                with torch.amp.autocast('cuda'):
                    out = ema.ema(imgs); loss = criterion(out, lbls)
                val_loss += loss.item() * imgs.size(0); preds.extend(out.argmax(1).cpu().numpy()); targets.extend(lbls.cpu().numpy())

        t_loss, t_acc = train_loss/len(train_loader.dataset), train_corrects.double().item()/len(train_loader.dataset)
        v_loss, v_acc = val_loss/len(val_loader.dataset), accuracy_score(targets, preds)
        pre, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='macro')
        pd.DataFrame([{'epoch':epoch+1,'train_loss':t_loss,'train_acc':t_acc,'val_loss':v_loss,'val_acc':v_acc,'precision':pre,'recall':rec,'f1':f1}]).to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(ema.ema.state_dict(), f"{SAVE_DIR}/best.pth")
            with open(f"{SAVE_DIR}/best_report.txt", 'w') as f: f.write(classification_report(targets, preds, digits=4))
        print(f"E{epoch+1} | EMA-Val Acc: {v_acc:.4f} | F1: {f1:.4f}")

if __name__ == '__main__': main()
