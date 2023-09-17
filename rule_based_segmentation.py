# Görev 1: Aşağıdaki soruları yanıtlayınız.


# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
df = pd.read_csv("python_for_data_science/data_analysis_with_python/datasets/persona.csv")
df.index
df.dtypes
df.size
df.ndim
df.values
type(df.values)
df.head(3)
df.tail(3)
df.shape
df.columns
df.info
df.describe()
df.isnull().values.any()
df.isnull().sum()

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?


df["SOURCE"].nunique()
df["SOURCE"].unique()
df["SOURCE"].describe()

# 2 UNIQUE değer (ANDROID,IOS) ve 2974 frekansa sahiptir.


# Soru 3: Kaç unique PRICE vardır?


df["PRICE"].nunique()
df["PRICE"].unique()
df["PRICE"].describe()


# 6 unique değer(39, 49, 29, 19, 59,  9) vardır


# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

df["PRICE"].value_counts()

# 19 **** 992
# 29 **** 1305
# 39 **** 1260
# 49 **** 1031
# 59 **** 212


# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

df["COUNTRY"].value_counts()

# usa *****  2065
# bra *****  1496
# deu *****  455
# tur *****  451
# fra *****  303
# can *****  230


# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df.groupby("COUNTRY").agg({"PRICE": "sum"})

# bra ****     51354
# can ****      7730
# deu ****     15485
# fra ****     10177
# tur ****     15689
# usa ****     70225


# Soru 7: SOURCE türlerine göre satış sayıları nedir?

df["SOURCE"].value_counts()

# android ****   2974
# ios     ****   2026



# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

df.groupby("COUNTRY").agg({"PRICE": "mean"})


# bra ****     34.327540
# can ****     33.608696
# deu ****     34.032967
# fra ****     33.587459
# tur ****     34.787140
# usa ****     34.007264



# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?

df.groupby("SOURCE").agg({"PRICE": "mean"})

# android ****  34.174849
# ios     ****  34.069102




# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?


df.groupby(["SOURCE","COUNTRY"]).agg({"PRICE": "mean"})

# android bra      34.387029
#         can      33.330709
#         deu      33.869888
#         fra      34.312500
#         tur      36.229437
#         usa      33.760357
# ios     bra      34.222222
#         can      33.951456
#         deu      34.268817
#         fra      32.776224
#         tur      33.272727
#         usa      34.371703


# Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).head()

#  PRICE
# COUNTRY SOURCE  SEX    AGE
# bra     android female 15   38.714286
#                        16   35.944444
#                        17   35.666667
#                        18   32.255814
#                        19   35.206897

# Görev 3: Çıktıyı PRICE’a göre sıralayınız.

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)

agg_df.head()

#  PRICE
# COUNTRY SOURCE  SEX    AGE
# bra     android male   46    59.0
# usa     android male   36    59.0
# fra     android female 24    59.0
# usa     ios     male   32    54.0
# deu     android female 36    49.0

# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.

agg_df = agg_df.reset_index()

agg_df.head()


# COUNTRY   SOURCE     SEX  AGE  PRICE
# 0     bra  android    male   46   59.0
# 1     usa  android    male   36   59.0
# 2     fra  android  female   24   59.0
# 3     usa      ios    male   32   54.0
# 4     deu  android  female   36   49.0

# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.

# AGE değişkeninin nerelereden bölünceğini belirledik.
bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]

# Bölünen noktalara karşılık isimlendirme yaptık.
mylabels = ["0_18", "19_23", "24_30", "31_40", "41_" + str(agg_df["AGE"].max())]

# AGE'yi böldük:
agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins=bins, labels=mylabels)

agg_df.head()


# COUNTRY   SOURCE     SEX  AGE  PRICE age_cat
# 0     bra  android    male   46   59.0   41_66
# 1     usa  android    male   36   59.0   31_40
# 2     fra  android  female   24   59.0   24_30
# 3     usa      ios    male   32   54.0   31_40
# 4     deu  android  female   36   49.0   31_40


# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
# • Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
# • Yeni eklenecek değişkenin adı: customers_level_based
# • Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based değişkenini oluşturmanız gerekmektedir.


# değişken isimleri:
agg_df.columns

# gözlem değerlerine erişelim:
for row in agg_df.values:
    print(row)

# COUNTRY, SOURCE, SEX ve age_cat değişkenlerinin DEĞERLERİNİ yan yana koymak ve alt tireyle birleştirmek istiyoruz.
# Bunu list comprehension ile yapabiliriz.
# Yukarıdaki döngüdeki gözlem değerlerinin bize lazım olanlarını seçecek şekilde işlemi gerçekletirelim:
[row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]

# Veri setine ekledik:
agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]
agg_df.head()

# COUNTRY   SOURCE     SEX  AGE  PRICE age_cat     customers_level_based
# 0     bra  android    male   46   59.0   41_66    BRA_ANDROID_MALE_41_66
# 1     usa  android    male   36   59.0   31_40    USA_ANDROID_MALE_31_40
# 2     fra  android  female   24   59.0   24_30  FRA_ANDROID_FEMALE_24_30
# 3     usa      ios    male   32   54.0   31_40        USA_IOS_MALE_31_40
# 4     deu  android  female   36   49.0   31_40  DEU_ANDROID_FEMALE_31_40

# Gereksiz değişkenleri çıkardık:
agg_df = agg_df[["customers_level_based", "PRICE"]]
agg_df.head()

# customers_level_based  PRICE
# 0    BRA_ANDROID_MALE_41_66   59.0
# 1    USA_ANDROID_MALE_31_40   59.0
# 2  FRA_ANDROID_FEMALE_24_30   59.0
# 3        USA_IOS_MALE_31_40   54.0
# 4  DEU_ANDROID_FEMALE_31_40   49.0


for i in agg_df["customers_level_based"].values:
    print(i.split("_"))



# Birçok aynı segment olduğu için groupby kullanıcaz.

agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})
agg_df.head()

#                              PRICE
# customers_level_based
# BRA_ANDROID_FEMALE_0_18   35.645303
# BRA_ANDROID_FEMALE_19_23  34.077340
# BRA_ANDROID_FEMALE_24_30  33.863946
# BRA_ANDROID_FEMALE_31_40  34.898326
# BRA_ANDROID_FEMALE_41_66  36.737179


# customers_level_based indexte yer alıyor değişkene çeviricez.

agg_df = agg_df.reset_index()
agg_df.head()

#  customers_level_based      PRICE
# 0   BRA_ANDROID_FEMALE_0_18  35.645303
# 1  BRA_ANDROID_FEMALE_19_23  34.077340
# 2  BRA_ANDROID_FEMALE_24_30  33.863946
# 3  BRA_ANDROID_FEMALE_31_40  34.898326
# 4  BRA_ANDROID_FEMALE_41_66  36.737179



# her personadan 1 tane mi var kontrol edelim.

agg_df["customers_level_based"].value_counts()
agg_df.head()

#  customers_level_based      PRICE
# 0   BRA_ANDROID_FEMALE_0_18  35.645303
# 1  BRA_ANDROID_FEMALE_19_23  34.077340
# 2  BRA_ANDROID_FEMALE_24_30  33.863946
# 3  BRA_ANDROID_FEMALE_31_40  34.898326
# 4  BRA_ANDROID_FEMALE_41_66  36.737179



# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.
# • Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
# • Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
# • Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).
agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})


#  PRICE
#               mean        max          sum
# SEGMENT
# D        29.206780  32.333333   817.789833
# C        33.509674  34.077340   904.761209
# B        34.999645  36.000000   944.990411
# A        38.691234  45.428571  1044.663328



# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
# • 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
# • 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?


new_user = "TUR_ANDROID_FEMALE_31_40"

agg_df[agg_df["customers_level_based"] == new_user]

# TUR_ANDROID_FEMALE_31_40 kullanıcı A segmentine aittir. Ortalama olarak 41.83 gelir getirmesi beklenir.

# customers_level_based      PRICE SEGMENT
# 72  TUR_ANDROID_FEMALE_31_40  41.833333       A



new_user_2 = "FRA_IOS_FEMALE_31_40"

agg_df[agg_df["customers_level_based"] == new_user_2]



# FRA_IOS_FEMALE_31_40 kullanıcı C segmentine aittir. Ortalama olarak 32.81 gelir getirmesi beklenir.

#  customers_level_based      PRICE SEGMENT
# 63  FRA_IOS_FEMALE_31_40  32.818182       C








