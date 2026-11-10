import streamlit as st
import requests
from PIL import Image


st_title = "paper rock scissors"

api_url = "http://127.0.0.1:8000/predict001"

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Image.', width=200)

    if st.button("Proverit"):
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        try:
            response = requests.post(api_url, files=files)

            if response.status_code == 200:
                pred = response.json()
                st.success(f'Prediction: {pred["result"]}')
            else:
                st.error(f'Oshibka API {response.text}')

        except requests.exceptions.ConnectionError:
            st.error(f'Hе удалось подключиться к серверу API')

        except Exception as e:
            st.error(f'Произошла ошибка: {e}')