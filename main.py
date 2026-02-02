# %% kütüphaneler 
import torch # tensor(matris) işlemleri
import torch.nn as nn # yapay sinir agi katmalarını tanılıyoruz
import torch.optim as optim  # optimizasyon algoritmalarını içeren modül
import torchvision # görüntü işleme ve önceden eğitilmiş modelleri içerir
import torchvision.transforms as transforms # görüntü donusumleri yapmak
from torchvision.datasets import ImageFolder #kendi yüklediğim verisetini kullanmak için
import matplotlib.pyplot as plt #görselleştirme
import numpy as np 
import os

# Cihaz seçimi (GPU varsa kullan, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Çalışılan Cihaz: {device}")

# %% veri seti yükleme 

def get_data_loaders(batch_size = 32): 
 
    # Eğitim verisi için veri çoğaltma (Augmentation) ve ön işleme
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),           
        transforms.RandomHorizontalFlip(p=0.5),  # %50 ihtimalle aynala
        transforms.RandomRotation(15),           # 15 derece sağa/sola döndür
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Işık ve kontrastla oyna
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # Test verisi için sadece boyutlandırma ve normalizasyon
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), #görüntüyü tensöre çevirir ve 0-255 -> 0-1 ölçeklenirir
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # rgb kanallarını normalize et
    ])


    # Veri setini yükleme 
    train_set = ImageFolder(root = './food9/train', transform=train_transform)
    test_set = ImageFolder(root = './food9/test', transform=test_transform)

    # Dataloader (Veri yükleyiciler)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 1. BLOK (Giriş: 3x128x128)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) # RGB oldugu icin giris 3, Filtre sayisi 32, kernel size 3x3
        self.bn1 = nn.BatchNorm2d(32) # Dengeli sayilar uretir
        # Matris carpimlarindan gelen dengesiz sayilari duzeltir
        
        # 2. BLOK (Giriş: 32x64x64)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
        self.bn2 = nn.BatchNorm2d(64)
        
        # 3. BLOK (Giriş: 64x32x32)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 4. BLOK (Giriş: 128x16x16)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Ortak Katmanlar
        self.relu = nn.ReLU() # Aktivasyon fonksiyonu, negatifleri sifirlar
        self.pool = nn.MaxPool2d(2, 2) # Boyut azaltma (2x2 havuzlama)
        
        # Sınıflandırma Katmanları (Classifier)
        # 4 kere pool yapildi: 128 -> 64 -> 32 -> 16 -> 8
        # Son Kanal Sayisi: 256
        self.fc1 = nn.Linear(256 * 8 * 8, 512) # Flatten sonrasi giris nöronu
        self.dropout = nn.Dropout(0.4) # Ezberlemeyi onlemek icin nöron kapatma
        self.fc2 = nn.Linear(512, 9) # Cikis katmani (9 yemek sinifi)

    def forward(self, x):
        # 1. convolution Blok
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # 2. convolution Blok
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # 3. convolution Blok
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        # 4. convolution Blok
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        #flatten n boyutlu matrisi vektörü tek boyutlu vektöre çeviriyoruz
        x = x.view(-1, 256 * 8 * 8) #flatten
        
        x = self.dropout(self.relu(self.fc1(x))) # fully connected layer
        x = self.fc2(x) # output
        return x

# %% loss fonksiyonu ve optimizer

def define_loss_and_optimizer(model):(
    criterion = nn.CrossEntropyLoss() #multi class classification yaptığımız için 
    optimizer = optim.Adam(model.parameters(), lr = 0.001) 
)  

# %% modelin eğitilmesi / training

def train_model(model, train_loader, criterion, optimizer, epochs = 50):
    
    model.train() #modeli eğitim moduna alıyoruz
    train_losses = [] #loss degerlerini saklamak icin bir liste olustur
    
    for epoch in range(epochs):  # for dongusu olusturup belirtilen epoch sayısı kadar 
        total_loss = 0 # toplam loss degerini saklamak icin total_loss
        
        for images, labels in train_loader: # for dongusu tum egitim veri setini taramak icin
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad() # Gradyanları sıfırla
            outputs = model(images) #forward pro. (prediction), output = etiket, label, class
            loss = criterion(outputs, labels) # Hatayı hesapla
            loss.backward() # Geriye yayılım (Backpropagation) (gradyan hesaplama)
            optimizer.step() # ogrenme = parametre yani ağırlık güncelleme
            
            total_loss += loss.item()
          
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"--> Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")

    # kayıp grafiğini çizdircez
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, marker = "o", linestyle = "-", label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("CNN Training Loss")
    plt.legend()
    plt.savefig('cnn_training_result.png') 
    plt.show()

# %% modelin test edilmesi

def test_model(model, test_loader, dataset_type="Test"):
    
    model.eval() # degerlendirme modu
    correct = 0 # dogru tahmin sayacı
    total = 0 # toplam veri sayaci
    
    with torch.no_grad(): #gradyan hesaplaması gereksiz olduğu için kapat
        for images, labels in test_loader: # test veri setini kullanarak degerlendirme
            images, labels = images.to(device), labels.to(device) # verileri cihaza tasi
            
            outputs = model(images) #prediction
            _, predicted = torch.max(outputs, 1) # en yuksek olasılıklı sınıfı seç
            
            total += labels.size(0) # toplam veri sayısı
            correct += (predicted == labels).sum().item() # dogru tahminleri say
            
    print(f"{dataset_type} accuracy: {100* correct / total:.2f}%")

# %% main (Ana Çalıştırma Bloğu)

if __name__ == "__main__":

    # Veri setini yükle
    train_loader, test_loader = get_data_loaders()
    
    #training
    model = CNN().to(device) 
    
    criterion, optimizer = define_loss_and_optimizer(model)
    
    train_model(model, train_loader, criterion, optimizer, epochs=50) 

    # modeli kaydediyoruz ki sürekli eğitim yapmayalım
    torch.save(model.state_dict(), 'kullanilacak.pth')
    print("Model kaydedildi: kullanilacak.pth")

    #test
    test_model(model, test_loader, dataset_type = "Test Seti")
    test_model(model, train_loader, dataset_type = "Training Seti")
