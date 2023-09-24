from typing import List, Any, Tuple

import nltk, re, string, collections
from nltk.util import ngrams
import pandas as pd
import numpy as np
from xlrd import open_workbook
import csv


def veriTemizleme_NoktalamaIsareti( text):
    text = re.sub(r'[^\w\s]', '', text)
    return text


def veriTemizleme_BoslukAltSatir( text):
    text = text.replace("\n", " ")
    text = re.sub(' +', ' ', text)
    return text


def veriTemizleme_Sayilar( text):
    text = re.sub(r'[0-9]+', '', text)
    return text

def bubble_sort(numbers, harf):
    for passesLeft in range(len(numbers) - 1, 0, -1):
        for index in range(passesLeft):
            if numbers[index] < numbers[index + 1]:
                numbers[index], numbers[index + 1] = numbers[index + 1], numbers[index]
                harf[index], harf[index + 1] = harf[index + 1], harf[index]


class NGram:
    def __init__(self):
        print("n -gram nesnesi oluşturuldu")

    def WordNgramArff(self,file_name,arffadres, n=2, mostCommon=100):
        # this corpus is pretty big, so let's look at just one of the files in it

        dfs = pd.read_excel(file_name, engine='openpyxl')
        # print(dfs)
        print("------------------------------------")
        # print(dfs.iloc[0:5, 1:3])  # ilk kısım satır ikinci kısım sutun bilgisini içerir
        text = " "

        for i in range(len(dfs.index)):
            text = text + dfs.iloc[i, 2]

        text = veriTemizleme_BoslukAltSatir(
            veriTemizleme_Sayilar(veriTemizleme_NoktalamaIsareti(text)))
        text = text.lower()


        # first get individual words
        tokenized = text.split()

        # and get a list of all the bi-grams
        esBigrams = ngrams(tokenized, n)



        # get the frequency of each bigram in our corpus
        esBigramFreq = collections.Counter(esBigrams)
        output = list(esBigramFreq.most_common(mostCommon))
        kelimelisteleri=[]
        for i in output:
            text=str(i)
            kelimelisteleri.append(text.split('(')[2].split(')')[0].replace("\'","").replace(",",""))

        with open(arffadres, "w", newline="") as f:
            yazici = csv.writer(f)
            yazici.writerow(['@relation cinsiyettespitikelimetabanli'])

            for satir in range(mostCommon):
                attr = kelimelisteleri[satir].replace(' ', '_').replace('i', 'II').replace('ç', 'cc').replace('ğ', 'gg').replace('ı','I').replace('ö', 'oo').replace('ş', 'ss').replace('ü', 'uu')
                # print(a)
                yazici.writerow(['@attribute ' + attr + ' REAL'])
            # yeni eklendi alttaki satır
            yazici.writerow(['@attribute class {0,1}'])
            yazici.writerow(['@data'])
            print("arff dosyasına yazmaya başlandı")
            k = 0
            for sayfa in range(len(dfs)):
                yazilacakSatir = []
                article_text = dfs.iloc[sayfa, 2]
                for i in range(mostCommon):
                    yazilacakSatir.append(article_text.count(kelimelisteleri[i]))
                yazilacakSatir.append(0 if str(dfs.iloc[sayfa, 1]) != "Females" else 1 )
                k += 1
                yazici.writerow(yazilacakSatir)
        f.close()



    def CharacterNgramArff(self, xlsxadres, arffadres, chars, rangegram=100):
        print(" işleme alındı")
        dfs = pd.read_excel(xlsxadres, engine='openpyxl')
        ngrams = {}
        metinler = []
        metinyazarcinsiyeti= []


        # sayfadaki satır sayısına nrows ile ulaşıyoruz.
        for satir in range(len(dfs)):

            # Sütün sayısına ncols üzerinden ulaşıyoruz
            sutun = 2
            article_text = str(dfs.iloc[satir, sutun])
            article_text = veriTemizleme_BoslukAltSatir(veriTemizleme_Sayilar(veriTemizleme_NoktalamaIsareti(article_text))).lower()
            metinler.append(article_text)
            metinyazarcinsiyeti.append(0 if str(dfs.iloc[satir, sutun - 1]) != "Females" else 1 )
            # print(article_text)

            for i in range(len(article_text) - chars):
                seq = article_text[i:i + chars]

                if seq not in ngrams.keys():
                    ngrams[seq] = []
                ngrams[seq].append(article_text[i + chars])

        del dfs
        grup = []
        sayi = []


        for ng in ngrams:
            # print(ng)
            grup.append(ng)
            i = i + 1
            # bv print(len(ngrams[ng]))
            sayi.append(len(ngrams[ng]))
        print("ngramlar çıkarıldı şimdi sıralanacak")
        bubble_sort(sayi, grup)
        print("ngramlar sıralandı")

        with open(arffadres, "w", newline="") as f:
            yazici = csv.writer(f)
            yazici.writerow(['@relation cinsiyettespiti'])

            for satir in range(rangegram):
                attr = grup[satir].replace(' ', '_').replace('i', 'II').replace('ç', 'cc').replace('ğ', 'gg').replace('ı','I').replace('ö', 'oo').replace('ş', 'ss').replace('ü', 'uu')
                # print(a)
                yazici.writerow(['@attribute ' + attr + ' REAL'])
            # yeni eklendi alttaki satır
            yazici.writerow(['@attribute class {0,1}'])
            yazici.writerow(['@data'])
            print("arff dosyasına yazmaya başlandı")
            k = 0
            for sayfa in metinler:
                yazilacakSatir = []
                article_text = sayfa
                for i in range(rangegram):
                    yazilacakSatir.append(article_text.count(grup[i]))
                yazilacakSatir.append(metinyazarcinsiyeti[k])
                k += 1
                yazici.writerow(yazilacakSatir)
        f.close()
        print("arff dosyası tamamlandı")
