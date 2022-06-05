import numpy as np
import streamlit as st
import pickle



pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))
st.title("Price prediction of laptop")

company = st.selectbox('Brand', df['Company'].unique())

type = st.selectbox('Type', df['TypeName'].unique())

ram = st.selectbox('Ram', [2, 4, 6, 8, 12, 16, 32])

weight = st.number_input('Laptop weight')

touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'])

ips = st.selectbox('IPS', ['No', 'Yes'])

screen_size = st.number_input('Screen Size')

resolution = st.selectbox('Screen Resolution', ['1280 x 720', '1920 x 1080', '2560 x 1440', '3840 x 2160'])

ppi = st.selectbox('PPI', df['PPI'].unique())

cpu = st.selectbox('CPUBrand', df['CPU Brand'].unique())

hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 156, 512, 1024])

gpubrand = st.selectbox('GPUBrand', df['GpuBrand'].unique())

os = st.selectbox('os', df['os'].unique())

if st.button('Predict price'):
    X_res= resolution.split('X')[0]
    Y_res = resolution.split('Y')[1]
    ppi = ((X_res**2) + (Y_res **2))**0.5/screen_size
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    query = np.array([company, type, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpubrand, os])

    query = query.reshape(1, 12)
    st.title(pipe.predict(query))