# Ödev 1: CIFAR-10 Üzerinde k-NN Uygulaması

Bu repoda, Derin Sinir Ağları dersi kapsamındaki ilk ödev olan "CIFAR-10 veri seti üzerinde k-NN (k-En Yakın Komşu) algoritması" uygulaması yer almaktadır. 

Ödev detaylarında istenildiği gibi kod; karmaşık object-oriented (sınıf) veya fonksiyonel yapılara bölünmeden, **düz ve lineer bir akışla** yukarıdan aşağıya çalışacak şekilde yazılmıştır. 

## Dosya İçeriği
- `knn_cifar10.py`: Ana Python kodu.
- Algoritma çalıştırılmadan önce `cifar-10-batches-py` isimli orijinal veri seti klasörünün bu kod ile aynı dizinde bulundurulması gereklidir. Veri internetten otomatik indirilmemekte, lokalden pickle ile okunması sağlanmaktadır.

## Nasıl Çalışır?

Kodu terminal üzerinden çalıştırdığınızda şu adımlar gerçekleşir:
1. Lokal klasördeki test ve eğitim dosyaları Numpy matrisleri olarak yüklenir. (Taşma hatalarını önlemek için `int32` formatı kullanılır).
2. Sizden mesafe metrik yöntemi istenilir: `1` diyerek L1 (Manhattan) veya `2` diyerek L2 (Öklid) seçebilirsiniz.
3. Kıyaslanacak komşu sayısı yani `k` değeri istenir.
4. Çıktıyı doğrudan görebilmeniz için, 10.000 test fotoğrafı arasından bir tanesi için index girmeniz beklenir (Örn: `15`, `8000` vb.)
5. Model, belirttiğiniz teste ait veriyi 50.000 fotoğraflık eğitim seti üzerinde formülle tek tek karşılaştırır.
6. Sonuç olarak seçtiğiniz nesnenin algoritma tarafından yapılan tahminini (doğruluğuyla birlikte) gösterir.

### Kullanım

Terminalde ilgili dizine gidip şu komutu çalıştırınız:
```bash
python knn_cifar10.py
```
