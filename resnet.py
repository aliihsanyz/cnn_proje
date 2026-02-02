# %% kütüphaneler 

import torch # tensor(matris) işlemleri

import torch.nn as nn # yapay sinir agi katmalarını tanılıyoruz

import torch.optim as optim  # optimizasyon algoritmalarını içeren modül

import torchvision # görüntü işleme ve önceden eğitilmiş modelleri içerir

import torchvision.transforms as transforms # görüntü donusumleri yapmak

from torchvision import models # resnet modeli için gerekli

from torchvision.datasets import ImageFolder #kendi yüklediğim verisetini kullanmak için

import matplotlib.pyplot as plt #görselleştirme

import numpy as np 
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Çalışılan Cihaz: {device}")

# %% veri seti yükleme 

def get_data_loaders(batch_size = 32): # her itarasyonda işlenecek veri mikatarı, batch size
 
    # ResNet için ImageNet standartları (Mean ve Std değerleri)
    # Bu değerler ResNet'in daha iyi performans vermesini sağlar
    mean_vals = (0.485, 0.456, 0.406)
    std_vals = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),   # ResNet standart giriş boyutu 224x224
        transforms.RandomHorizontalFlip(p=0.5),  # %50 ihtimalle aynala
        transforms.RandomRotation(15),           # 15 derece sağa/sola döndür
        transforms.ToTensor(), 
        transforms.Normalize(mean_vals, std_vals) 
    ])

    #test verisine data augmentation yapmaya gerek yok 
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet boyutu
        transforms.ToTensor(), #görüntüyü tensöre çevirir ve 0-255 -> 0-1 ölçeklenirir
        transforms.Normalize(mean_vals, std_vals) # rgb kanallarını normalize et
    ])

    #klasör kontrolü
    if not os.path.exists('./food9/train'):
        print("UYARI: Klasör bulunamadı! Lütfen dosya yolunu kontrol et.")

    #veri setini yükleme 
    train_set = ImageFolder(root = './food9/train', transform=train_transform)
    test_set = ImageFolder(root = './food9/test', transform=test_transform)

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader

# %% Model Mimarisi (ResNet-18)

class ResNetModel(nn.Module):

    def __init__(self):
        super(ResNetModel, self).__init__()
        
        # Önceden eğitilmiş ResNet-18 modelini indiriyoruz
        # weights=models.ResNet18_Weights.DEFAULT sayesinde ImageNet ağırlıkları gelir
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # ResNet'in son katmanı (fc) 1000 sınıf için ayarlıdır.
        # Bunu kendi 9 sınıfımıza göre değiştiriyoruz.
        num_ftrs = self.base_model.fc.in_features
        
        self.base_model.fc = nn.Linear(num_ftrs, 9) # 9 sınıfımız var

    def forward(self, x):
        # ResNet'in kendi forward yapısını kullanıyoruz
        x = self.base_model(x)
        return x

# %% loss fonksiyonu oluşturma

define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), #multi class classification yaptığımız için 
    optim.Adam(model.parameters(), lr = 0.001) #Adam optimizasyonu
)

# %% modelin eğitilmesi / training

def train_model(model, train_loader, criterion, optimizer, epochs = 15): # ResNet hızlı öğrenir, epoch 15 yeterli

    model.train() #modeli eğitim moduna alıyoruz
    train_losses = [] #loss degerlerini saklamak icin bir liste olustur

    for epoch in range(epochs): # for dongusu olusturup belirtilen epoch sayısı kadar 
        total_loss = 0 # toplam loss degerini saklamak icin total_loss
        
        print(f"Epoch {epoch+1}/{epochs} başlıyor...")

        for images, labels in train_loader: # for dongusu tum egitim veri setini taramak icin
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() #gradyanları sıfırlıyoruz

            outputs = model(images) #forward pro. (prediction)

            loss = criterion(outputs, labels) #loss degerini hesapla
            loss.backward() # geri yayılım (gradyan hesaplama)
            optimizer.step() # ogrenme = parametre yani ağırlık güncelleme

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        print(f"--> Epoch: {epoch+1}/{epochs}, Loss: {avg_loss: .5f}")

    # kayıp grafiğini çizdireceğiz
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker = "o", linestyle = "-", label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("ResNet-18 Training Loss")
    plt.legend()
    plt.savefig('resnet_result.png') # grafiği kaydet
    plt.show()
    
# %% modelin test edilmesi

def test_model(model, test_loader, dataset_type="test"):
    
    model.eval() # degerlendirme modu
    correct = 0 # dogru tahmin sayacı
    total = 0 # toplam veri sayaci
    
    with torch.no_grad(): #gradyan hesaplaması gereksiz olduğu için kapat
        for images, labels in test_loader: # test veri setini kullanarak degerlendirme
            images, labels = images.to(device), labels.to(device)# verileri cihaza tasi
            
            outputs = model(images) #prediction
            _, predicted = torch.max(outputs, 1) # en yuksek olasılıklı sınıfı seç
            
            total += labels.size(0) # toplam veri sayısı
            correct += (predicted == labels).sum().item() # dogru tahminleri say
            
    print(f"{dataset_type} accuracy: {100* correct / total:.2f}%")
            
# %% main 

if __name__ == "__main__":

    #veri seti yükleme
    train_loader, test_loader = get_data_loaders(batch_size=32)
    
    #training
    model = ResNetModel().to(device) 
    
    criterion, optimizer = define_loss_and_optimizer(model)
    
    # ResNet güçlü olduğu için 15 epoch genelde yeterlidir, isteğe göre 50 yapılabilir
    train_model(model, train_loader, criterion, optimizer, epochs=15) 

    # modeli kaydediyoruz
    torch.save(model.state_dict(), 'resnet18_food9.pth')
    print("Model kaydedildi: resnet18_food9.pth")

    #test
    test_model(model, test_loader, dataset_type = "Test Seti")
    # test_model(model, train_loader, dataset_type = "Training Seti")
