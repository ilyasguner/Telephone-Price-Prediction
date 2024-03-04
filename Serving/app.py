import streamlit as st
import pandas as pd 
import numpy as np
import joblib
from PIL import Image,ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns


def main():       
    predict()
           

def predict():

    # Markalar ve Modellerin yüklenmesi
    markalar = load_data()
    
    # Kullanıcı arayüzü ve değer alma
    st.title('Merhaba, *Streamlit! İle Makine Öğrenmesi Telefon Tahmin Uygulamasına Hoş Geldiniz!* 👨‍💻')

    selected_brand = marka_index(markalar,st.selectbox('Marka Seçiniz..',markalar))
    
    selected_os=isletim_sistemi(st.radio("İşletim Sistemi",["Android","IOS"]))

    selected_cpu = st.slider('CPU',min_value=1.0,max_value=3.5,step=0.2)
    st.write("CPU :"+str(selected_cpu)+" GHz")

    selected_dahili_hafiza = st.number_input('Dahili Hafıza',min_value=4,max_value=1024)
    st.write("Dahili Hafıza :"+str(selected_dahili_hafiza)+" GB")

    selected_ekran_boyutu = st.slider("Ekran Boyutu",min_value=4.5,max_value=14.0)
    st.write("Ekran Boyutu :"+str(selected_ekran_boyutu)+" inç")

    selected_kamera_cozunurlugu = st.slider("Arka Kamera Çözünürlüğü",min_value=5,max_value=210)
    st.write("Arka Kamera Çözünürlüğü :"+str(selected_kamera_cozunurlugu)+" MP")

    selected_mobil_baglanti_hizi = st.slider("Mobil Bağlantı Hızı",min_value=3.5,max_value=5.5,step=0.5)
    st.write("Mobil Bağlantı Hızı :"+str(selected_mobil_baglanti_hizi)+" G")

    selected_pil_gucu = st.slider("Pil Gücü",min_value=1500,max_value=20000)
    st.write("Pil Gücü :"+str(selected_pil_gucu)+" mAh")

    selected_ram_kapasitesi= st.number_input('Ram Kapasitesi',min_value=1,max_value=32)
    st.write("Ram Kapasitesi :"+str(selected_ram_kapasitesi)+" GB")

    selected_on_kamera_cozunurluk = st.slider("Ön Kamera",min_value=5,max_value=120)
    st.write("Ön Kamera :"+str(selected_on_kamera_cozunurluk)+" MP")

    selected_model = st.selectbox('Tahmin Modeli Seçiniz..',["Random Forest","Gradient Boosting"])


    prediction_value = create_prediction_value(selected_cpu,selected_dahili_hafiza,selected_ekran_boyutu,
                                               selected_kamera_cozunurlugu,selected_mobil_baglanti_hizi,selected_pil_gucu,selected_ram_kapasitesi,selected_on_kamera_cozunurluk,selected_os,selected_brand)
    prediction_model = load_models(selected_model)


    if st.button("Tahmin Yap"):
            result = predict_models(prediction_model,prediction_value)
            if result != None:
                st.success('Tahmin Başarılı')
                st.balloons()
                st.write("Tahmin Edilen Fiyat: "+ result + "TL")
            else:
                st.error('Tahmin yaparken hata meydana geldi..!')
    

#markalar için csv dosyası
def load_data():
    markalar = pd.read_csv("brands.csv")
    return markalar

#model yükleme
def load_models(modelName):
    if modelName == "Random Forest":  
        rf_model = joblib.load("phones_random_forest.pkl")
        return rf_model
    if modelName=="Gradient Boosting":
        rf_model=joblib.load("phones_gradient_boosting.pkl")
        return rf_model
    else:
        st.write("Model yüklenirken hata meydana geldi..!")
        return 0

#marka indexi bulma
def marka_index(markalar,marka):
    index = int(markalar[markalar["brands"]==marka].index.values)
    return index

#isletim sistemi için sayısal değer atama
def isletim_sistemi(isletim_sistemi):
    if isletim_sistemi == "Android":
        return 0
    else:
        return 1


def create_prediction_value(cpu,dahili_hafiza,ekran_boyutu,kamera_cozunurlugu,mobil_baglanti_hizi,pil_gucu,ram_kapasitesi,on_kamera_cozunurluk,isletim_sistemi,brands):
    res = pd.DataFrame(data = 
            {"cpu":[cpu],"dahili_hafiza":[dahili_hafiza],"ekran_boyutu":[ekran_boyutu],
                    "kamera_cozunurlugu":[kamera_cozunurlugu],"mobil_baglanti_hizi":[mobil_baglanti_hizi],"pil_gucu":[pil_gucu],"ram_kapasitesi":[ram_kapasitesi],
                    "on_kamera_cozunurluk":[on_kamera_cozunurluk],"isletim_sistemi":[isletim_sistemi],"brands":[brands]})
    return res

#modeli çalıştırma
def predict_models(model,res):
    result = str(int(model.predict(res))).strip('[]')
    return result

if __name__ == "__main__":
    main()