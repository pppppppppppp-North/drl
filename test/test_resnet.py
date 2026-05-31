import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import time
import sys
import os

# ================= CONFIGURATION =================
RUNTIME_SECONDS = 3600       # 1 Hour
BATCH_SIZE = 128             # Efficient for RTX 4500
LEARNING_RATE = 0.001
DATA_PATH = './data'         # Local folder for dataset
# =================================================

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- HPC Benchmark: ResNet-50 Training ---")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # 2. Data Preparation (Resize to 224x224 to simulate heavy workload)
    print("Preparing Data (CIFAR-10 -> Upscaled to 224x224)...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Ensure download=True so it grabs data if missing
    trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                            download=True, transform=transform)
    
    # num_workers=4 matches the --cpus-per-task in the SLURM script
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=4, pin_memory=True)

    # 3. Model Setup
    model = resnet50(weights=None) 
    model.fc = nn.Linear(model.fc.in_features, 10) # Adjust for 10 classes
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop with Time Limit
    print(f"\nStarting training loop for {RUNTIME_SECONDS} seconds...")
    start_time = time.time()
    total_imgs = 0
    epoch = 0
    
    model.train()

    try:
        while (time.time() - start_time) < RUNTIME_SECONDS:
            epoch += 1
            running_loss = 0.0
            
            for i, data in enumerate(trainloader, 0):
                # Check time limit inside the batch loop
                if (time.time() - start_time) >= RUNTIME_SECONDS:
                    break

                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_imgs += labels.size(0)

                # Log progress every 20 batches
                if i % 20 == 19:
                    elapsed = time.time() - start_time
                    img_per_sec = total_imgs / elapsed
                    print(f"Epoch {epoch} [Batch {i+1}] | Loss: {running_loss/20:.3f} | Speed: {img_per_sec:.1f} img/sec")
                    running_loss = 0.0
                    sys.stdout.flush()

    except Exception as e:
        print(f"Error during training: {e}")

    # 5. Finish
    final_time = time.time() - start_time
    print(f"\n--- Time Limit Reached ---")
    print(f"Total Runtime: {final_time/60:.2f} minutes")
    print(f"Saving model to 'resnet50_finished.pth'...")
    torch.save(model.state_dict(), "resnet50_finished.pth")

if __name__ == "__main__":
    main()
