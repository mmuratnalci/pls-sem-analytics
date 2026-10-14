# PLS-SEM Analytics

Türkçe Streamlit uygulaması, CSV ve Excel dosyalarını yükleyip ön inceleme
istatistiklerini ve PLS-SEM yansıtıcı ölçüm modeli sonuçlarını üretir. Katılımcı
kimliği, ölçek madde öneki taşımadığından PLS-SEM analizine dahil edilmez.

## Analiz modeli

`PQ`, `BT`, `CS`, `BI`, `RI` ve `WOM` ile başlayan sütunlar ilgili yapının
maddeleri olarak otomatik tanınır. Başlangıç yapısında `PQ` ve `BT` dışsal
yapılardır; ikisi de `CS` ve `BI` yapılarına, `CS` ve `BI` de `RI` ve `WOM`
yapılarına bağlanır. Analiz ekranında kullanılabilir yapılar arasındaki yönlü
ilişkiler değiştirilebilir. Her yapıda en az iki sayısal ve değişken madde
bulunmalıdır. Eksik madde yanıtı olan satırlar analizden çıkarılır; kullanılan
katılımcı sayısı ve çıkarılan satır sayısı sonuçlarla gösterilir.

Hesaplamalar, `plspm==0.5.7` Python kütüphanesinin PLS Path Modeling algoritması
ile, yansıtıcı Mode A ölçüm modeli, standartlaştırılmış maddeler ve path iç
ağırlıklandırma şeması kullanılarak yapılır. Dış yükler kütüphanenin tahmin
edilmiş gösterge-yapı korelasyonlarıdır. Cronbach's Alpha standart kovaryans
formülüyle, rho_c standartlaştırılmış dış yüklerden
`(Σλ)² / ((Σλ)² + Σ(1−λ²))`, AVE ise `mean(λ²)` formülüyle hesaplanır. HTMT,
klasik HTMT'dir (HTMT+ değildir): mutlak Pearson madde korelasyonlarından,
heterotrait korelasyonlarının ortalamasının monotrait korelasyon ortalamalarının
geometrik ortalamasına oranı olarak hesaplanır. Sonuç tablosunda bu ayrım
"HTMT (Klasik)" olarak belirtilir. HTMT, her yapı için birden fazla madde
olmadığında tanımlı değildir; bu uygulama analiz öncesinde minimum madde
sayısını doğrular.

Her yapının dış yükleri toplamı negatifse, yüklerin tamamı ve ilgili gizil
değişken skoru aynı anda −1 ile çarpılarak yapı yönü belirlenir. Yol katsayıları,
kaynak ve hedef yapıların yön değişimleriyle uyumlu olacak şekilde dönüştürülür.
Böylece tabloda gösterilen yükler ile rho_c ve AVE aynı yönlendirilmiş yükleri
kullanır; CR formülünde negatif yükler varsa işaretleri korunur. `plspm` yakınsama
ölçütü (ağırlık farklarının kareler toplamı `< 1e-7`) sağlanmazsa en çok 1000
iterasyondan sonra hata üretir. Sonuçlardaki yakınsama durumu, hesaplayıcının
hata vermeden sonuç üretmesine dayanır; iterasyon sayısı kütüphanenin herkese
açık sonuç API'sinde bulunmadığı için raporlanmaz.

Eksik veya hesaplanamayan değerler için yapay sonuç üretilmez. Bu tahminler
SmartPLS ile sayısal olarak birebir eşleşmeyebilir: bu uygulama Python `plspm`
0.5.7'nin PLS Path Modeling uygulamasını, standartlaştırılmış maddeler ve
path-weighting şemasıyla kullanırken SmartPLS sürümü ve seçenekleri farklı
PLS-SEM algoritma/başlangıç, ağırlıklandırma, işaret değiştirme, durdurma
ölçütü, eksik veri ve ordinal veri işleme tercihleri kullanabilir. Bu modül
SmartPLS'e özgü PLSc/consistent PLS, HTMT+, bootstrap güven aralıkları veya
anlamlılık testlerini hesaplamaz. Dış yükler, güvenirlik ve HTMT sonuçları ayrı
tablolar halinde gösterilir ve Excel çalışma kitabına veya ayrı CSV dosyalarına
aktarılabilir.

## Yerel çalıştırma ve test

```bash
pip install -r requirements.txt
streamlit run app.py
python -m unittest discover -s tests
```
