import csv
import pandas as pd
from jpype import JClass, getDefaultJVMPath, startJVM, shutdownJVM

def zemberekKokBulmaTumMetin(metin,zemberek):
    # iism fiil bilgilerini tutan
    tip = {"isim": 0, "fiil": 0, "sifat": 0, "zamir": 0, "kisaltma": 0, "bilinmeyen": 0, "ozel": 0, "sayi": 0}
    i = 0
    for kelime in metin.split():

        if kelime.strip() > '':
            yanit = zemberek.kelimeCozumle(kelime)
            if yanit:
                a = str(yanit[0].kok())
                # print("{}".format(a))
                sayis = a.count('ISIM')
                sayfiil = a.count('FIIL')
                saysifat = a.count('SIFAT')
                sayzamir = a.count('ZAMIR')
                saykisaltma = a.count('KISALTMA')
                sayozel = a.count('OZEL')
                saysayi = a.count('SAYI')
                if sayis > 0:
                    tip["isim"] += 1
                if sayfiil > 0:
                    tip["fiil"] += 1
                if saysifat > 0:
                    tip["sifat"] += 1
                if sayzamir > 0:
                    tip["zamir"] += 1
                if saykisaltma > 0:
                    tip["kisaltma"] += 1
                if sayozel > 0:
                    tip["ozel"] += 1
                if saysayi > 0:
                    tip["sayi"] += 1
            else:
                tip["bilinmeyen"] += 1
        i += 1
    return tip

class SyntaticFeatures:
    def __init__(self):
        # shutdownJVM()
        ZEMBEREK_PATH = r'zemberek-tum.jar'
        startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))
        Tr = JClass("net.zemberek.tr.yapi.TurkiyeTurkcesi")
        # tr nesnesini oluştur
        tr = Tr()
        # Zemberek sınıfını yükle
        Zemberek = JClass("net.zemberek.erisim.Zemberek")
        # zemberek nesnesini oluştur
        zemberek = Zemberek(tr)
        print("Sözcüksel özellikler nesnesi oluşturuldu")

    def sozcukselOznitelikSetiOlustur(self, exceladres, arffadres):
        print(exceladres, " işleme alındı")
        dfs = pd.read_excel(exceladres, sheet_name='Çizelge1', engine='openpyxl')
        # print(dfs)
        print("------------------------------------")
        # print(dfs.iloc[0:5, 1:3])  # ilk kısım satır ikinci kısım sutun bilgisini içerir
        text = " "
        sutun=1
        with open(arffadres,"w",newline="") as f:
            yazici=csv.writer(f)
            yazici.writerow(['@relation cinsiyettespiti_sozcukselanaliz'])
            yazici.writerow(['@relation koseyazarlari'])
            yazici.writerow(['@attribute toplamkelimesayisi REAL'])
            yazici.writerow(['@attribute birkezkullanilankelimesayisi REAL'])
            yazici.writerow(['@attribute ikikezkullanilankelimesayisi REAL'])
            yazici.writerow(['@attribute v2oranv1 REAL'])
            yazici.writerow(['@attribute sesliharfsayisi REAL'])
            yazici.writerow(['@attribute soruisaret REAL'])
            yazici.writerow(['@attribute unlemisareti REAL'])
            yazici.writerow(['@attribute parantezisareti REAL'])
            yazici.writerow(['@attribute tireisareti REAL'])
            yazici.writerow(['@attribute noktalivirgulisareti REAL'])
            yazici.writerow(['@attribute noktaisareti REAL'])
            yazici.writerow(['@attribute virgulisareti REAL'])
            yazici.writerow(['@attribute tektirnakisareti REAL'])
            yazici.writerow(['@attribute cifttirnakisareti REAL'])
            yazici.writerow(['@attribute cumlesayisi REAL'])
            yazici.writerow(['@attribute harfsayisi REAL'])
            yazici.writerow(['@attribute toplamkaraktersayisi REAL'])
            yazici.writerow(['@attribute ortalamakelimeuzunlugu REAL'])
            yazici.writerow(['@attribute cumledeOrtalamaKelimeSayisi REAL'])
            yazici.writerow(['@attribute isimSayisi REAL'])
            yazici.writerow(['@attribute fiilSayisi REAL'])
            yazici.writerow(['@attribute sifatSayisi REAL'])
            yazici.writerow(['@attribute zamirSayisi REAL'])
            yazici.writerow(['@attribute kisaltmaSayisi REAL'])
            yazici.writerow(['@attribute turuBilinmeyenKelimeSayisi REAL'])
            yazici.writerow(['@attribute ozelIsimSayisi REAL'])
            yazici.writerow(['@attribute sayikelimeSayisi REAL'])
            yazici.writerow(['@attribute ssclass'])
            yazici.writerow(['@data'])

            for satir in range(len(dfs)):
                text = dfs.iloc[satir, 2]
                text=text.lower().replace("text:","")
                yanit = zemberekKokBulmaTumMetin(text)
                yazici.writerow([yanit])
        f.close()
        print(arffadres, "dosyasi yazılımı tamamlandı")



