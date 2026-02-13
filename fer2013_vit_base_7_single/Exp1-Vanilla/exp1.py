import os, torch, timm, torch.nn as nn, numpy as np, pandas as pd
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

def main():
    SAVE_DIR = './exp1_vanilla'
    os.makedirs(SAVE_DIR, exist_ok=True)
    LOG_FILE = f"{SAVE_DIR}/train_log.csv"
    
    tf = transforms.Compose([
        transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_loader = DataLoader(datasets.ImageFolder('./data/train', tf), batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(datasets.ImageFolder('./data/test', tf), batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=7).cuda()
    if os.path.exists('./weights/vit_base_patch16_224.pth'):
        sd = torch.load('./weights/vit_base_patch16_224.pth', map_location='cpu')
        sd = sd['model'] if 'model' in sd else sd
        sd = {k: v for k, v in sd.items() if "head" not in k}
        model.load_state_dict(sd, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    best_acc = 0.0

    for epoch in range(80):
        model.train()
        train_loss, train_corrects = 0.0, 0
        for imgs, lbls in tqdm(train_loader, desc=f"Exp1 E{epoch+1}"):
            imgs, lbls = imgs.cuda(), lbls.cuda()
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(imgs); loss = criterion(out, lbls)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            train_loss += loss.item() * imgs.size(0); train_corrects += torch.sum(out.argmax(1) == lbls.data)

        model.eval(); preds, targets, val_loss = [], [], 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.cuda(), lbls.cuda()
                with torch.amp.autocast('cuda'):
                    out = model(imgs); loss = criterion(out, lbls)
                val_loss += loss.item() * imgs.size(0)
                preds.extend(out.argmax(1).cpu().numpy()); targets.extend(lbls.cpu().numpy())

        t_loss, t_acc = train_loss/len(train_loader.dataset), train_corrects.double().item()/len(train_loader.dataset)
        v_loss, v_acc = val_loss/len(val_loader.dataset), accuracy_score(targets, preds)
        pre, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='macro')

        log_data = {'epoch':epoch+1,'train_loss':t_loss,'train_acc':t_acc,'val_loss':v_loss,'val_acc':v_acc,'precision':pre,'recall':rec,'f1':f1}
        pd.DataFrame([log_data]).to_csv(LOG_FILE, mode='a', index=False, header=not os.path.exists(LOG_FILE))
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), f"{SAVE_DIR}/best.pth")
            with open(f"{SAVE_DIR}/best_report.txt", 'w') as f: f.write(classification_report(targets, preds, digits=4))
        print(f"E{epoch+1} | Loss: {t_loss:.4f} | Acc: {v_acc:.4f} | F1: {f1:.4f}")

if __name__ == '__main__': main()