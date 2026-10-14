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
mutlak maddeler arası korelasyonlardan heterotrait ortalamasının monotrait
ortalamalarının geometrik ortalamasına oranıdır. HTMT, her yapı için birden
fazla madde olmadığında tanımlı değildir; bu uygulama analiz öncesinde minimum
madde sayısını doğrular.

Eksik veya hesaplanamayan değerler için yapay sonuç üretilmez. Bu tahminler
SmartPLS'in algoritma, varsayılan ayarlar, veri işleme ve yuvarlama
tercihlerinden farklı olabileceği için SmartPLS ile sayısal olarak birebir
eşleşme taahhüt edilmez. Dış yükler, güvenirlik ve HTMT sonuçları ayrı
tablolar halinde gösterilir ve Excel çalışma kitabına veya ayrı CSV dosyalarına
aktarılabilir.

## Yerel çalıştırma ve test

```bash
pip install -r requirements.txt
streamlit run app.py
python -m unittest discover -s tests
```
