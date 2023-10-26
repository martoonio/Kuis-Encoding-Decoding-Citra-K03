import streamlit as st
import plotly.graph_objs as go
from skimage import io
import plotly.express as px
import numpy as np
import pandas as pd
import scipy.fft as fft
import matplotlib.image as mpimg
import utils

st.set_page_config(layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.title("Kuis Encoding Decoding Citra K03")
st.markdown("Nama : Farchan Martha Adji Chandra\n\nNIM : 18221011")

img = io.imread('KUIS2.jpg')
# Tabel Kuantisasi didapat dari PPT
tk = np.matrix(
'16 11 10 16 24 40 51 61 ;\
12 12 14 19 26 58 60 55 ;\
14 13 16 24 40 57 69 56 ;\
14 17 22 29 51 87 80 62 ;\
18 22 37 56 68 109 103 77 ;\
24 35 55 64 81 104 103 92 ;\
49 64 78 77 103 121 120 101;\
72 92 95 98 112 100 103 99'
).astype('float')

tk = np.array(tk.tolist())
df = pd.DataFrame(tk)

img_array = np.array(img)
Y = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

arrayImg = pd.DataFrame(img_array[:, :, 0])

macroblock = Y[216:224, 197:205]

# Buat DataFrame untuk mencetak makroblok
dfMacro = pd.DataFrame(macroblock)


fig = px.imshow(img, color_continuous_scale='gray')
fig.update_layout(coloraxis_showscale=False)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.update_layout(title="FOTO SAMPLE", title_x=0.46)  # Judul grafik
st.plotly_chart(fig, use_container_width=True)

st.header("Pilih Proses :")
st.markdown("nb : Hasil kompresi terdapat di proses decoding step terakhir")
cb1,cb2 = st.columns(2)
with cb1.container():
    encode = st.button("Encoding")

with cb2.container():
    decode = st.button("Decoding")


if encode:
    st.header("Proses Encoding")
    with st.expander("Step 1 : Environment Setup"):
        c1,c2 = st.columns(2)
        with c1.container():
            st.subheader("Source Code")
            code = st.code("""
                           # Import Library
                            import numpy as np
                            import pandas as pd
                            import matplotlib.pyplot as plt
                            import matplotlib.image as mpimg
                            import scipy.fft as fft
                            from IPython.display import Image

                            # Tabel Kuantisasi didapat dari PPT
                            tk = np.matrix(
                            '16 11 10 16 24 40 51 61 ;\
                            12 12 14 19 26 58 60 55 ;\
                            14 13 16 24 40 57 69 56 ;\
                            14 17 22 29 51 87 80 62 ;\
                            18 22 37 56 68 109 103 77 ;\
                            24 35 55 64 81 104 103 92 ;\
                            49 64 78 77 103 121 120 101;\
                            72 92 95 98 112 100 103 99'
                            ).astype('float')

                            tk = np.array(tk.tolist())
                            print("Tabel 1.1. Tabel Kuantisasi")
                            pd.DataFrame(tk)

                            """)
        with c2.container():
            st.subheader("Tabel Kuantisasi")
            st.dataframe(df)

    with st.expander("Step 2 : Ambil Matrix Y Gambar"):
        c1,c2 = st.columns(2)
        with c1.container():
            st.subheader("Source Code")
            code = st.code("""
                            # Ambil Matrix Y Gambar
                            img = mpimg.imread('KUIS2.jpg')
                            img_array = np.array(img)
                            Y = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

                            arrayImg = pd.DataFrame(img_array[:, :, 0])
                            print("Tabel 1.2. Tabel Matrix Y Gambar")
                            arrayImg
                            """)
        with c2.container():
            st.subheader("Tabel Matrix Y Gambar")
            st.dataframe(arrayImg)

    with st.expander("Step 3 : Ambil Microblock 8x8"):
        c1,c2 = st.columns(2)
        with c1.container():
            st.subheader("Source Code")
            code = st.code("""
                            # Ambil Microblock 8x8
                            macroblock = Y[216:224, 197:205]

                            # Buat DataFrame untuk mencetak makroblok
                            dfMacro = pd.DataFrame(macroblock)
                            print("Tabel 1.3. Macrobloc 8x8")
                            dfMacro
                            """)
        with c2.container():
            st.subheader("Tabel Macrobloc 8x8")
            st.dataframe(dfMacro)

    with st.expander("Step 4 : Konversi Nilai Macroblock"):
        c1,c2 = st.columns(2)
        with c1.container():
            st.subheader("Source Code")
            code = st.code("""
                            # Konversi Macrobloc
                            for i in range(0,8):
                                for j in range(0,8):
                                    dfMacro[i][j] -= 128

                            print("Tabel 1.4. Macrobloc Konversi Nilai")
                            dfMacro
                            """)
        with c2.container():
            for i in range(0,8):
                for j in range(0,8):
                    dfMacro[i][j] -= 128
            st.subheader("Tabel Macrobloc Konversi Nilai")
            st.dataframe(dfMacro)

    with st.expander("Step 5 : Transform Coding dengan DCT"):
        c1,c2 = st.columns(2)
        with c1.container():
            st.subheader("Source Code")
            code = st.code("""
                            # Transform Coding dengan DCT
                            mb_compress = fft.dct(dfMacro)
                            print("Tabel 1.5. Hasil DCT")
                            pd.DataFrame(mb_compress)
                            """)
        with c2.container():
            mb_compress = fft.dct(dfMacro)
            st.subheader("Tabel Hasil DCT")
            st.dataframe(mb_compress)

    with st.expander("Step 6 : Kuantisasi dengan Tabel Kuantisasi"):
        c1,c2 = st.columns(2)
        with c1.container():
            st.subheader("Source Code")
            code = st.code("""
                            # Kuantisasi dengan Tabel Kuantisasi
                            for i in range(0,8):
                                for j in range(0,8) :
                                    mb_compress[i][j] = round(mb_compress[i][j]/tk[i][j])

                            print("Tabel 1.6. Hasil Kuantisasi")
                            pd.DataFrame(mb_compress)
                            """)
        with c2.container():
            for i in range(0,8):
                for j in range(0,8) :
                    mb_compress[i][j] = round(mb_compress[i][j]/tk[i][j])
            st.subheader("Tabel Hasil Kuantisasi")
            st.dataframe(mb_compress)

        fig1 = px.imshow(mb_compress, color_continuous_scale='gray')
        st.plotly_chart(fig1, use_container_width=True)

if decode:

    st.header("Proses Decoding")

    tk = np.matrix(
    '16 11 10 16 24 40 51 61 ;\
    12 12 14 19 26 58 60 55 ;\
    14 13 16 24 40 57 69 56 ;\
    14 17 22 29 51 87 80 62 ;\
    18 22 37 56 68 109 103 77 ;\
    24 35 55 64 81 104 103 92 ;\
    49 64 78 77 103 121 120 101;\
    72 92 95 98 112 100 103 99'
    ).astype('float')

    tk = np.array(tk.tolist())
    df = pd.DataFrame(tk)
    img_array = np.array(img)
    Y = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

    arrayImg = pd.DataFrame(img_array[:, :, 0])

    macroblock = Y[216:224, 197:205]

    # Pengubahan format dari 0-255 menjadi -128-127
    for i in range(0,8):
        for j in range(0,8):
            macroblock[i][j] -= 128

    mb_compress = fft.dct(macroblock)
    mb_compress = pd.DataFrame(mb_compress)

    for i in range(0,8):
        for j in range(0,8) :
            mb_compress[i][j] = round(mb_compress[i][j]/tk[i][j])

    mb_decomp = mb_compress
    for i in range(0,8):
        for j in range(0,8):
            mb_decomp[i][j] = round(mb_decomp[i][j] * tk[i][j])


    with st.expander("Step 1 : Lakukan Dekuantisasi"):
        c1,c2 = st.columns(2)
        with c1.container():
            st.subheader("Source Code")
            code = st.code("""
                            # Lakukan Dekuantisasi
                            mb_decomp = mb_compress
                            for i in range(0,8):
                                for j in range(0,8):
                                    mb_decomp[i][j] = round(mb_decomp[i][j] * tk[i][j])

                            print("Tabel 2.1. Hasil Dekuantisasi")
                            pd.DataFrame(mb_decomp)
                            """)
        with c2.container():
            st.subheader("Tabel Hasil Dekuantisasi")
            st.dataframe(mb_decomp)

    with st.expander("Step 2 : Transform Inverse DCT"):
        c1,c2 = st.columns(2)
        with c1.container():
            st.subheader("Source Code")
            code = st.code("""
                            # Transform Inverse DCT
                            mb_decomp = fft.idct(mb_decomp)
                            # Pembulatan hasil transform invers
                            for i in range(0,8):
                                for j in range(0,8):
                                    mb_decomp[i][j] = round(mb_decomp[i][j])

                            print("Tabel 2.2. Hasil Transform Inverse DCT")
                            pd.DataFrame(mb_decomp)
                            """)
        with c2.container():
            mb_decomp = fft.idct(mb_decomp)
            # Pembulatan hasil transform invers
            for i in range(0,8):
                for j in range(0,8):
                    mb_decomp[i][j] = round(mb_decomp[i][j])
            st.subheader("Tabel Hasil Transform Inverse DCT")
            st.dataframe(mb_decomp)

    with st.expander("Step 3 : Konversi ke Range 0-255"):
        c1,c2 = st.columns(2)
        with c1.container():
            st.subheader("Source Code")
            code = st.code("""
                            # Konversi ke Range 0-255
                            for i in range(0,8):
                                for j in range(0,8):
                                    mb_decomp[i][j] += 128

                            print("Tabel 2.3. Hasil Konversi ke Range 0-255")
                            pd.DataFrame(mb_decomp)
                            """)
        with c2.container():
            st.subheader("Tabel Hasil Konversi ke Range 0-255")
            for i in range(0,8):
                for j in range(0,8):
                    mb_decomp[i][j] += 128
            st.dataframe(mb_decomp)

        st.header("Perbandingan  nilai warna sebelum dan sesudah kompresi")
        tcol1, tcol2 = st.columns(2)
        tcol1.markdown("Hasil Sebelum Kompressi")
        tcol1.dataframe(round(dfMacro))
        tcol2.markdown("Hasil Kompressi")
        tcol2.dataframe(mb_decomp)

        st.header("Perbandingan Warna per Pixel")
        col1,col2 = st.columns(2)
        with col1.container():
            st.markdown("Hasil Sebelum Kompressi")
            fig3 = px.imshow(round(dfMacro), color_continuous_scale='gray')
            st.plotly_chart(fig3, use_container_width=True)
        with col2.container():
            st.markdown("Hasil Kompressi")
            fig2 = px.imshow(mb_decomp, color_continuous_scale='gray')
            st.plotly_chart(fig2, use_container_width=True)