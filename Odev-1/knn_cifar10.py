import numpy as np
import pickle
import os

# -------------------------------------------------------------------
# CIFAR-10 k-NN Sınıflandırma
# Gerekli veri klasörünü (cifar-10-batches-py) kodun olduğu dizine kopyaladığınızdan emin olun.
# -------------------------------------------------------------------

# CIFAR-10 veri setindeki sınıfların listesi
siniflar = ['Uçak', 'Otomobil', 'Kuş', 'Kedi', 'Geyik', 
            'Köpek', 'Kurbağa', 'At', 'Gemi', 'Kamyon']

# Veriyi internetten çekmiyoruz, lokalden okuyoruz
veri_klasoru = "cifar-10-batches-py"

if not os.path.exists(veri_klasoru):
    print(f"HATA: '{veri_klasoru}' klasörü bulunamadı!")
    print("Lütfen CIFAR-10 verisini zip'ten çıkarıp kodun bulunduğu dizine koyun.")
    exit()

print("Veriler lokalden yükleniyor, lütfen bekleyin...")

# Eğitim verilerini tutacağımız listeler
X_train_list = []
Y_train_list = []

# Toplam 5 adet batch dosyası var, bunları sırayla okuyoruz
for i in range(1, 6):
    dosya_adi = os.path.join(veri_klasoru, f"data_batch_{i}")
    with open(dosya_adi, 'rb') as f:
        sozluk = pickle.load(f, encoding='bytes')
        X_train_list.append(sozluk[b'data'])
        Y_train_list.append(sozluk[b'labels'])

# Okuduğumuz listeleri alt alta birleştirip (50.000 satır) Numpy dizisine çeviriyoruz.
# Taşmaları (overflow) önlemek adına int32 veri tipine çevirmek çok önemlidir. 
# Çünkü (küçük_sayı - büyük_sayı) gibi işlemlerde uint8 hata verecektir.
X_train = np.concatenate(X_train_list).astype('int32')
Y_train = np.concatenate(Y_train_list)

# Test verisini ayrı olan test batch dosyasından okuyoruz
test_dosyasi = os.path.join(veri_klasoru, "test_batch")
with open(test_dosyasi, 'rb') as f:
    test_sozluk = pickle.load(f, encoding='bytes')
    X_test = test_sozluk[b'data']
    Y_test = test_sozluk[b'labels']

X_test = np.array(X_test).astype('int32')
Y_test = np.array(Y_test)

print("Eğitim ve test verileri başarıyla yüklendi!\n")

# ---- Kullanıcıdan İşlem Seçimlerinin Alınması ----

# L1 (Manhattan) veya L2 (Öklid) Seçimi
print("Hangi mesafe metriğini kullanmak istersiniz?")
print("1: L1 (Manhattan)")
print("2: L2 (Öklid)")

secim = input("Seçiminiz (1/2, L1/L2): ").strip().upper()

if secim in ['1', 'L1', 'MANHATTAN']:
    metrik = 'L1'
elif secim in ['2', 'L2', 'ÖKLID', 'OKLID', 'EUCLIDEAN']:
    metrik = 'L2'
else:
    print("Geçersiz seçim yaptınız. Varsayılan olarak L1 (Manhattan) kullanılacak.")
    metrik = 'L1'

# 'k' Değerinin Ekstradan Alınması
k = 3 # varsayılan
while True:
    try:
        k_str = input("Lütfen 'k' değeri giriniz (Örn: 3): ").strip()
        k = int(k_str)
        if k <= 0:
            print("k değeri pozitif ve sıfırdan büyük bir sayı olmalıdır.")
            continue
        break
    except ValueError:
        print("Lütfen sadece sayı giriniz!")

# Cihazda hangi nesnenin sınıflandırılacağını belirleyelim
print(f"\nTest veri setinde toplam {len(X_test)} adet görüntü var.")
while True:
    try:
        test_indeksi_str = input(f"Sınıflandırılacak test nesnesinin indeksini seçin (0 - {len(X_test)-1}): ").strip()
        test_indeksi = int(test_indeksi_str)
        if test_indeksi < 0 or test_indeksi >= len(X_test):
            print(f"Lütfen 0 ile {len(X_test)-1} arasında bir rakam girin.")
            continue
        break
    except ValueError:
        print("Lütfen sadece sayı giriniz!")

# Seçilen test nesnesinin bilgileri
test_goruntusu = X_test[test_indeksi]
gercek_etiket = Y_test[test_indeksi]

print(f"\nSeçilen Test Objekt İndeksi: {test_indeksi}")
print(f"Nesnenin Gerçek Sınıfı: {siniflar[gercek_etiket]}")
print("Hesaplanıyor, Lütfen bekleyin... (Bu işlem bütün eğitim setini gezdiği için biraz vakit alabilir)")

# ---- k-NN Sınıflandırmasının Uygulanması ---- 

# Öğrenilen hiçbir şey kullanılmadan düz şekilde uzaklıklar hesaplanır
if metrik == 'L1':
    # Manhattan: sum(|A - B|)
    # X_train yapısındaki aynı formata göre uzaklık çıkartır
    mesafeler = np.sum(np.abs(X_train - test_goruntusu), axis=1)
else:
    # Öklid: sqrt(sum( (A - B)^2 ))
    mesafeler = np.sqrt(np.sum(np.square(X_train - test_goruntusu), axis=1))

# En küçük uzaklığa sahip k komşunun indekslerini bulma (argsort diziyi küçükten büyüğe sıralar)
k_yakin_indeksleri = np.argsort(mesafeler)[:k]

# Bulunan indekslere karşılık gelen sınıfları veri setinden çekiyoruz
k_yakin_etiketleri = Y_train[k_yakin_indeksleri]

# Komşulardan en çok tekrar eden (mod) etiket değerini bulma
# np.bincount dizideki her sayının kaç kez geçtiğini sayar, argmax en yüksek değeri döndürerek asıl sınıfı verir
tahmin_edilen_etiket = np.bincount(k_yakin_etiketleri).argmax()

# Sonucu yazdırma
print("\n--- SINIFLANDIRMA SONUCU ---")
print(f"Kullanılan Metrik: {metrik}")
print(f"Komşu Sayısı (k): {k}")
print(f"Algoritmanın Tahmin Ettiği Sınıf: {siniflar[tahmin_edilen_etiket]}")

if tahmin_edilen_etiket == gercek_etiket:
    print("Tebrikler, modelimiz nesneyi **DOĞRU** tahmin etti! ✓")
else:
    print("Maalesef, model nesneyi **YANLIŞ** tahmin etti. ✗")
