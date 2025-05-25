import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA



# Excel dosyasını oku
df = pd.read_excel("supplier_ranking_grades.xlsx", sheet_name="Sheet2")

# İlk 5 satırı görüntüle
print(df.head())

# Veri tiplerini kontrol et
print(df.dtypes)

# Eksik veri var mı?
print(df.isnull().sum())
# Tüm sütun adlarını sadeleştiriyoruz
df.columns = [
    "supplier", "quality", "quantity", "payment", "service", "reputation",
    "flexibility", "finance", "assets", "employees", "price", "delivery_time", "location"
]

# İlk 5 satırı yeniden görelim
print(df.head())
df.columns

# "supplier" dışındaki tüm sütunları sayısal değerlere dönüştür (hatalı olanları NaN yap)
for col in df.columns:
    if col != "supplier":
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Eksik veri sayımını tekrar kontrol edelim
print(df.isnull().sum())

# Eksik verili satırları çıkaralım
df_clean = df.dropna()

# Temiz veri boyutunu görelim
print("Temizlenmiş veri boyutu:", df_clean.shape)

# "supplier" kolonu dışındaki tüm sayısal verileri ölçeklendirelim
features = df_clean.drop(columns=["supplier"])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Gerekirse Pandas DataFrame'e çevir
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

# Dendrogram çizimi
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(scaled_df, method='ward'))
plt.title("Tedarikçi Segmentasyonu - Dendrogram")
plt.xlabel("Tedarikçiler")
plt.ylabel("Mesafe")
plt.tight_layout()
plt.show()

# Modeli oluştur
cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = cluster.fit_predict(scaled_df)

# Veriye etiketleri ekle
df_clean["cluster"] = labels

# Küme bazlı ortalamaları hesapla
cluster_summary = df_clean.groupby("cluster").mean(numeric_only=True)
print("Küme Ortalamaları:\n", cluster_summary)

# Küme dağılımı
print("\nHer kümede kaç tedarikçi var:")
print(df_clean["cluster"].value_counts())

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_df)

df_clean["pca1"] = pca_result[:, 0]
df_clean["pca2"] = pca_result[:, 1]


plt.figure(figsize=(8,6))
sns.scatterplot(data=df_clean, x="pca1", y="pca2", hue="cluster", palette="Set2")
plt.title("Tedarikçilerin Kümeleme Sonucu (2D PCA)")
plt.show()
