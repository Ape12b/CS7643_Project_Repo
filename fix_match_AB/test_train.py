import torch
from torch import nn
from tqdm import tqdm
from ema import EMA
import json

def consistency_loss(logits_weak, logits_strong):
    return nn.CrossEntropyLoss()(logits_weak, logits_strong)

def generate_pseudo_labels(model, images, threshold=0.95):    
    org_state = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }
    is_train = model.training
    with torch.no_grad():
        model.train()
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        max_probs, pseudo_labels = torch.max(probs, dim=1)
        mask = max_probs > threshold
    model.load_state_dict(org_state)
    if is_train:
        model.train()
    else:
        model.eval()
    return pseudo_labels[mask].detach(), mask


def train_fixmatch(model, labeled_loader, unlabeled_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs=150, threshold=0.95, lu_weight=1.0):
    results = []
    ema = EMA(model, alpha=0.99)
    
    best_acc = -1
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for (labeled_images, labels), (unlabeled_images, _) in zip(tqdm(labeled_loader), unlabeled_loader):
            labeled_images = labeled_images.to(device)
            labels = labels.to(device).long()
            weak_images, strong_images = unlabeled_images[0], unlabeled_images[1]
            weak_images = weak_images.to(device)
            strong_images = strong_images.to(device)

            optimizer.zero_grad()            
            
            weak_labels, mask = generate_pseudo_labels(model, weak_images, threshold=threshold)
            strong_images = strong_images[mask]
            n_l, n_u = labeled_images.size(0), strong_images.size(0)
            if n_u != 0:
                x = torch.cat([labeled_images, strong_images], dim=0).detach()
                y = torch.cat([labels, weak_labels], dim=0).detach()
                logits = model(x)
                logits_labelled, logits_unlabelled = logits[:n_l], logits[n_l:]
                supervised_loss = criterion(logits_labelled, labels.long())
                unsupervised_loss = criterion(logits_unlabelled, weak_labels.long())
                loss = supervised_loss + lu_weight * unsupervised_loss
            else:
                logits_labelled = model(labeled_images)
                supervised_loss = criterion(logits_labelled, labels.long())
                unsupervised_loss = torch.tensor(0)
                loss = supervised_loss

            loss.backward()
            optimizer.step()
            ema.update_params()
            running_loss += loss.item()

        running_loss /= len(labeled_loader)
        scheduler.step()
        ema.update_buffer()
        test_loss, acc = test(model, test_loader, criterion, device, ema)
        best_acc = max(best_acc, acc)
        
        log_msg = [
            f'epoch: {epoch + 1}',
            f'acc: {acc:.4f}',
            f'best_acc: {best_acc:.4f}'
        ]
        print(', '.join(log_msg))
        
        results.append({
            'epoch': epoch + 1,
            'accuracy': acc,
            'test_loss': test_loss,
            'running_loss': running_loss
        })
        
        with open('training_results_ver1.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}')
        torch.cuda.empty_cache()
    return model