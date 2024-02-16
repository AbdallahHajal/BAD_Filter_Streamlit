
#Abdallah Abou Hajal

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published

# This program is distributed in the hope that it will be useful.


#load packages
import streamlit as st
import requests
from streamlit_option_menu import option_menu
from rdkit import Chem
from rdkit.Chem import Draw
from mordred import Calculator, descriptors
import pandas as pd
import numpy as np
import time
import hashlib
import json
import html
import gc
import os
from streamlit.components.v1 import html
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from PIL import Image
from streamlit_ketcher import st_ketcher
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
import pickle
import urllib.request
import requests
import joblib
from io import BytesIO

url2 = 'https://github.com/AbdallahHajal/BAD_Filter_Streamlit/releases/download/Model/scaler_MM.pkl' 
url = 'https://github.com/AbdallahHajal/BAD_Filter_Streamlit/releases/download/Model/MM_model.pkl'

# Load the compressed model file
try:
    model = pickle.load(urllib.request.urlopen(url))
except Exception as e:
    print("Error loading model:", e)
    model = None

# Load the scalar file
try:
    scaler = pickle.load(urllib.request.urlopen(url2))
except Exception as e:
    print("Error loading scaler:", e)
    scaler = None

# Check if both model and scaler are loaded successfully
if model is not None and scaler is not None:
    print("Model and scaler loaded successfully.")
    # You can now use 'model' and 'scaler' for prediction.
else:
    print("Failed to load model and/or scaler. Check file paths and file integrity.")

st.set_page_config(page_title="BAD_Molecule_Filter", layout="wide")

#background of webpage
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] > .main {
    padding: 0 5%;
    margin: 0;
    max-width: 100%;
    background: linear-gradient(to right, #ffffff, #fff7e6); /* White to very light orange gradient */
    background-size: cover;
    background-attachment: local;
}

/* Transparent header to match with the new background */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


def preprocess_and_scale_MM(X,scaler):
    binary_cols = [col for col in X.columns if 'Col' in col]
    continuous_cols = [col for col in X.columns if col not in binary_cols]
    
    # Use the global scaler object
    X_continuous = X[continuous_cols].values
    X_continuous_scaled = scaler.transform(X_continuous)  # We'll use transform instead of fit_transform
    
    X_scaled = np.concatenate((X_continuous_scaled, X[binary_cols].values), axis=1)
    return X_scaled

def morgan_fpts(data):
    Morgan_fpts = []
    for i in data:
        mol = Chem.MolFromSmiles(i) 
        fpts =  AllChem.GetMorganFingerprintAsBitVect(mol,2,2048)
        mfpts = np.array(fpts)
        Morgan_fpts.append(mfpts)  
    return np.array(Morgan_fpts)

def All_Mordred_descriptors(data):
    calc = Calculator(descriptors, ignore_3D=False)
    mols = [Chem.MolFromSmiles(smi) for smi in data]
    
    # pandas df
    df = calc.pandas(mols)
    return df

descriptor_columns_MM=['ATS0p', 'SLogP', 'ATSC1Z', 'Col_694', 'TopoPSA', 'nAcid', 'piPC10', 'SlogP_VSA2', 'nHBDon', 'AATSC1are', 'BalabanJ', 'SsssNH', 'ZMIC0', 'Col_2009', 'ATSC0v', 'SlogP_VSA1', 'n6aHRing', 'MPC9', 'SdO', 'ECIndex', 'SMR_VSA1', 'NdsN', 'Col_1602', 'VSA_EState8', 'AATSC1v', 'SssS', 'EState_VSA9', 'SMR_VSA6', 'FilterItLogS', 'SaaN', 'Col_222', 'ATSC2p', 'WPath', 'Col_516', 'C3SP2', 'AATS1i', 'Col_1696', 'ATS2Z', 'Col_229', 'SMR_VSA3', 'Lipinski', 'Col_314', 'SlogP_VSA8', 'Col_1549', 'ATSC2are', 'Col_1089', 'Diameter', 'VSA_EState2', 'GATS1p', 'FCSP3', 'ATSC0i', 'AATSC1i', 'SMR_VSA2', 'piPC5', 'nS', 'Col_561', 'Xc-3dv', 'MATS1p', 'Col_1417', 'AATS1p', 'Col_216', 'Col_798', 'SaasC', 'ATSC3dv', 'PEOE_VSA4', 'PEOE_VSA10', 'NaasC', 'ATSC1pe', 'SsssN', 'NddsN', 'ATS0m', 'ATS1dv', 'PEOE_VSA2', 'Col_739', 'Mi', 'PEOE_VSA1', 'Col_464', 'piPC1', 'EState_VSA8', 'GGI10', 'Col_875', 'Col_950', 'Col_1970', 'Col_385', 'SsCl', 'Col_341', 'ATS6v', 'SssO', 'SlogP_VSA10', 'TopoPSA(NO)', 'Col_608', 'ATSC4d', 'ATSC3i', 'Col_1449', 'Xch-7d', 'Col_714', 'Col_1959', 'Col_495', 'PEOE_VSA13', 'JGI10', 'ATSC2dv', 'SdssC', 'ATSC7i', 'ATSC5dv', 'Col_1905', 'Col_105', 'Col_1034', 'GATS1m', 'Xch-5d', 'Col_1867', 'ZMIC1', 'SlogP_VSA3', 'SMR_VSA5', 'Col_1142', 'Col_2044', 'SMR_VSA4', 'ATS8m', 'Col_162', 'Col_231', 'PEOE_VSA6', 'Col_725', 'Col_248', 'CIC2', 'Mv', 'AATSC0d', 'Col_1911', 'NsssNH', 'GATS1i', 'Xpc-5d', 'Col_716', 'SIC0', 'Col_451', 'Col_1858', 'Col_949', 'nX', 'Col_1655', 'Xc-4d', 'nAromBond', 'C1SP2', 'IC5', 'ATSC2d', 'ATSC2Z', 'Xch-6dv', 'Col_1261', 'ATSC4m', 'Xc-4dv', 'SaasN', 'ATS6dv', 'VSA_EState6', 'ATSC6i', 'Col_611', 'GATS1dv', 'Col_251', 'Col_362', 'VSA_EState3', 'ATSC0p', 'EState_VSA10', 'Col_1917', 'Col_946', 'AATSC0v', 'Col_695', 'Col_797', 'Col_1784', 'Col_980', 'VSA_EState7', 'SlogP_VSA5', 'NaasN', 'ATSC6m', 'Col_381', 'Col_598', 'ATSC3v', 'PEOE_VSA12', 'nBase', 'Col_242', 'Col_423', 'Col_792', 'Col_505', 'JGI2', 'Col_1967', 'Col_494', 'IC2', 'PEOE_VSA5', 'GGI5', 'Col_823', 'Col_638', 'Col_1130', 'Col_184', 'ATSC8are', 'Col_273', 'ATSC3Z', 'Col_1632', 'Col_1771', 'JGI4', 'VSA_EState4', 'AATS1are', 'TSRW10', 'SlogP_VSA11', 'ZMIC4', 'Col_502', 'Col_1639', 'JGI8', 'Col_55', 'VSA_EState9', 'Col_280', 'Col_496', 'ATSC6v', 'Col_456', 'Col_927', 'Col_1347', 'ATSC5v', 'Xch-7dv', 'Col_355', 'ATSC4p', 'Col_1775', 'Col_1635', 'Col_421', 'Col_301', 'Xpc-5dv', 'Col_1693', 'AATS0d', 'ATSC4are', 'Col_1589', 'Col_147', 'Col_1790', 'Col_140', 'MWC07', 'Col_197', 'Col_670', 'MIC0', 'Col_1032', 'RotRatio', 'AATSC1dv', 'ATSC0pe', 'CIC1', 'Col_1070', 'Col_578', 'nBondsD', 'AATSC0dv', 'ATSC2i', 'fMF', 'Col_1827', 'ATSC3d', 'Col_1145', 'ATSC8d', 'JGI7', 'Xch-6d', 'Col_232', 'Col_2029', 'Col_896', 'ATSC8v', 'Col_275', 'JGI5', 'PEOE_VSA11', 'TopoShapeIndex', 'GATS1pe', 'SaaaC', 'Col_563', 'ATSC5d', 'nHRing', 'Col_1171', 'Col_626', 'Col_523', 'C2SP2', 'ATSC1p', 'VSA_EState5', 'Col_1485', 'Col_1734', 'ATSC8Z', 'Col_1412', 'Col_165', 'Mp', 'Col_1984', 'Col_1553', 'SMR_VSA9', 'Col_879', 'ATSC3p', 'StsC', 'AATSC0m', 'Col_557', 'Col_1672', 'Col_278', 'Col_1210', 'Col_94', 'ATSC1d', 'Col_1739', 'Col_1508', 'ATS8dv', 'PEOE_VSA7', 'Xc-6dv', 'Xc-3d', 'Col_559', 'ATSC5Z', 'Col_983', 'Col_224', 'Col_210', 'PEOE_VSA9', 'CIC5', 'Col_1657', 'MIC3', 'JGI6', 'Col_133', 'Col_948', 'Col_214', 'ATSC8i', 'Xch-5dv', 'ATSC1dv', 'Col_1403', 'Col_277', 'Col_318', 'nG12FaHRing', 'EState_VSA3', 'Col_393', 'Col_1656', 'ATSC7pe', 'nSpiro', 'n6HRing', 'Col_1000', 'Col_893', 'ATSC7p', 'n12FARing', 'Col_1845', 'nBondsS', 'ATSC4i', 'Col_265', 'Col_1966', 'Col_1770', 'Col_1537', 'ATSC5i', 'Col_1934', 'GGI2', 'Col_865', 'Col_730', 'Col_268', 'Col_990', 'Col_71', 'Col_776', 'Col_1203', 'EState_VSA2', 'Col_1745', 'ATSC7d', 'Col_1057', 'n4aRing', 'Col_883', 'Col_1573', 'Col_207', 'Col_432', 'Xc-6d', 'Col_1114', 'Col_1385', 'Col_1027', 'ATSC7dv', 'Col_1831', 'Col_503', 'SRW09', 'Col_575', 'AATSC1d', 'ATSC8dv', 'nBondsO', 'Col_1861', 'Col_1388', 'AATSC0are', 'Xc-5d', 'Col_99', 'ATSC5p', 'Col_1169', 'Col_760', 'Col_943', 'n8Ring', 'JGI9', 'SlogP_VSA7', 'Xch-4d', 'Col_1992', 'PEOE_VSA8', 'Col_822', 'Col_1859', 'Col_586', 'Col_1011', 'Col_1601', 'SMR_VSA7', 'Col_1855', 'Col_1713', 'Col_1728', 'Col_2017', 'Col_1814', 'Col_1058', 'Col_127', 'ZMIC2', 'nAromAtom', 'Col_1585', 'ATS5m', 'nHetero', 'Col_1122', 'Col_1355', 'Col_83', 'Col_1168', 'ATSC2v', 'Col_1050', 'C1SP1', 'Col_1188', 'Col_441', 'Col_633', 'Col_1996', 'Col_527', 'Col_1005', 'Col_349', 'Col_173', 'fragCpx', 'Col_962', 'MIC1', 'Col_1157', 'Col_2012', 'Col_978', 'Col_1885', 'ATSC5are', 'nO', 'Col_1759', 'Xc-5dv', 'ATSC6are', 'ATSC7Z', 'Col_84', 'ATS4m', 'Col_842', 'ATSC6p', 'Col_855', 'Col_835', 'ATSC6d', 'Col_960', 'ATSC6dv', 'ATSC4dv', 'n6ARing', 'Col_294', 'AATS1dv', 'Xp-4dv', 'SssssC', 'Col_1152', 'Col_504', 'ATSC3pe', 'Col_1160', 'nFaHRing', 'Col_1866', 'Col_1504', 'Col_809', 'Col_590', 'Col_802', 'Col_935', 'Col_334', 'Col_389', 'Col_1737', 'Col_1164', 'Col_1166', 'Col_350', 'Col_771', 'AATS1Z', 'Col_1364', 'Col_779', 'Col_1061', 'Col_1937', 'Col_651', 'Col_917', 'Col_1304', 'Col_1400', 'Col_738', 'Col_486', 'Col_1580', 'Col_1221', 'Col_1804', 'Col_995', 'Col_388', 'Col_811', 'Col_706', 'Col_1542', 'Col_863', 'Col_2041', 'Col_1311', 'Col_2021', 'Col_1299', 'Col_573', 'Col_580', 'nFRing', 'Col_255', 'AATSC0i', 'Col_996', 'Col_473', 'Col_1240', 'Col_777', 'Col_1738', 'Col_702', 'Col_918', 'Col_839', 'Col_477', 'Col_521', 'Col_1066', 'Col_366', 'nFAHRing', 'Col_506', 'Col_1922', 'Col_1958', 'Col_591', 'Col_614', 'Col_1444', 'Col_476', 'Col_1495', 'Col_1565', 'Col_675', 'Col_1982', 'Col_446', 'Col_1779', 'Col_645', 'Col_1243', 'Col_1281', 'Col_1604', 'Col_544', 'Col_1381', 'Col_1264', 'Col_816', 'Col_1056', 'Col_1852', 'Col_1990', 'Col_1019', 'Col_1295', 'Col_1398', 'Col_1084', 'Col_1613', 'Col_866', 'MZ', 'Col_840', 'NaaO', 'Col_718', 'Col_661', 'Col_1199', 'Col_619', 'Col_2034', 'Col_424', 'Col_1705', 'Col_1547', 'Col_1704', 'Col_1865', 'Col_1889', 'Col_97', 'Col_1266', 'Col_1540', 'Col_376', 'Col_1039', 'Col_1109', 'Col_1920', 'Col_984', 'Col_878', 'Col_1803', 'Col_758', 'Col_1954', 'Col_1554', 'Col_1119', 'Col_852', 'Col_1276', 'Col_459', 'Col_2040', 'Col_750', 'n9FHRing', 'Col_1818', 'Col_815', 'Col_853', 'Col_2026', 'Col_475', 'n10FaRing', 'Col_1481', 'Col_819', 'Col_1624', 'Col_961', 'Col_938', 'Col_1357', 'Col_1012', 'Col_909', 'Col_1459', 'Col_511', 'Col_245', 'Col_1821', 'Col_1570', 'Col_1666', 'Col_1766', 'Col_680', 'Col_1038', 'ATSC0d', 'Col_618', 'Col_1339', 'Col_1471', 'Col_1644', 'Col_1785', 'Col_1810', 'Col_783', 'Col_1721', 'Col_624', 'Col_1891', 'Col_354', 'Col_971', 'Col_1729', 'Col_252', 'Col_1088', 'Col_1174', 'Col_532', 'Col_837', 'Col_1205', 'Col_534', 'Col_1030', 'Col_1177', 'Col_1856', 'Col_1273', 'Col_2032', 'Col_260', 'Col_1329', 'Col_1309', 'Col_1452', 'Col_1136', 'Col_1308', 'Col_1287', 'Col_1087', 'Col_1194', 'Col_1232', 'Col_1929', 'Col_1217', 'Col_1685', 'Col_342', 'Col_327', 'Col_1227', 'Col_378', 'Col_2013', 'Col_1028', 'n5aRing', 'Col_2033', 'Col_807', 'n10FHRing', 'nF', 'Col_1665', 'Col_1753', 'n5ARing', 'Col_1095', 'AATS0dv', 'Col_1230', 'Col_1386', 'Col_1427', 'Col_1825', 'Col_1091', 'Col_1646', 'Col_747', 'Col_1235', 'Col_1252', 'Col_1694', 'Col_1480', 'Col_844', 'Col_1220', 'Col_1106', 'Col_1137', 'Col_920', 'Col_1204', 'Col_1212', 'Col_2028', 'Col_1462', 'Col_2015', 'Col_1489', 'Col_1522', 'Col_1869', 'Col_1453', 'Col_1228', 'Col_1047', 'Col_1008', 'Col_1915', 'Col_1033', 'Col_1393', 'Col_1979', 'Col_841', 'Col_1384', 'Col_833', 'Col_1409', 'Col_1794', 'Col_1455', 'Col_1490', 'Col_1138', 'Col_1768', 'Col_1608', 'Col_1105', 'Col_1463', 'Col_2023', 'Col_1419', 'Col_1121', 'Col_1269', 'Col_941', 'Col_1300', 'Col_1928', 'SdsssP', 'Col_1523', 'Col_1035', 'Col_1260', 'Col_1482', 'Col_474', 'Col_1411', 'n7Ring', 'Col_1413', 'Col_1351', 'Col_1305', 'Col_1706', 'Col_864', 'Col_1757', 'Col_2035', 'Col_1429', 'Col_1762', 'Col_1792', 'Col_1246', 'Col_1637', 'Col_856', 'Col_1795', 'Col_1238', 'Col_1989', 'Col_1956', 'Col_1249', 'Col_1997', 'Col_1839', 'n5HRing', 'Col_1535', 'Col_1620', 'Col_1262', 'Col_1919', 'Col_923', 'Col_1801', 'Col_1185', 'Col_1363', 'Col_1819', 'Col_1747', 'Col_1702', 'Col_1007', 'Col_804', 'Col_1285', 'Col_1173', 'Col_1428', 'Col_1887', 'Col_1835', 'Col_1727', 'Col_1677', 'Col_1510', 'Col_1163', 'Col_836', 'Col_1972', 'Col_1206', 'Col_1090', 'Col_1981', 'Col_736', 'Col_1983', 'Col_1284', 'Col_1591', 'Col_1654', 'Col_1430', 'Col_857', 'Col_1755', 'Col_1722', 'Col_1312', 'Col_1816', 'Col_1291', 'Col_1245', 'Col_1707', 'Col_1914', 'Col_1275', 'Col_1279', 'Col_1296', 'Col_1617', 'Col_1673', 'Col_1606', 'Col_1330', 'Col_1756', 'Col_1310', 'Col_1214', 'Col_1454', 'Col_1501', 'Col_1998', 'Col_1360', 'Col_1496', 'Col_1348', 'Col_1921', 'Col_1631', 'Col_1410', 'Col_1244', 'Col_1720', 'Col_1517', 'Col_1687', 'Col_1754', 'Col_1362', 'Col_1674', 'Col_1551', 'Col_1892', 'Col_1980', 'Col_1382', 'Col_1603', 'Col_1669', 'Col_1248', 'Col_1924', 'Col_1971', 'Col_1609', 'Col_1725', 'Col_2004', 'Col_1733', 'Col_1800', 'Col_1829', 'nG12FRing', 'Col_1947', 'Col_1599', 'Col_1750', 'Col_1499', 'Col_1864', 'Col_1483', 'Col_1986', 'Col_1799', 'Col_1909', 'Col_1479', 'Col_1475', 'Col_1897', 'Col_1520', 'Col_1527', 'Col_1614', 'Col_1995', 'Col_1752', 'Col_1679', 'Col_1650', 'Col_1274', 'Col_1539', 'Col_1850', 'Col_1465', 'Col_1791', 'Col_1683', 'Col_1724', 'Col_1849', 'Col_1991', 'Col_1908', 'Col_1918', 'Col_1847', 'Col_1746', 'Col_1808', 'Col_2003', 'Col_2014', 'Col_2039', 'Col_1815', 'Col_1953', 'Col_2000', 'Col_1773', 'Col_1940', 'Col_1903', 'Col_1778', 'Col_1977', 'Col_1783', 'Col_2042', 'Col_1726', 'Col_1823', 'Col_1708', 'Col_1946', 'Col_2001', 'Col_1964', 'Col_2022', 'Col_2038', 'Col_2007', 'Col_1760', 'Col_1697']
start_column_MM= 255
end_column_MM = 255

#clearning SMILES input button
def clear_text():
    st.session_state["text"] = ""

custom_css = """
<style>
    :root {
        font-size: 16px;
        text-align: left;
    }
    .text-second-title {
        font-size: 2rem;
        text-align: left;
        color: #f97316;
    }

    @keyframes text-gradient-title {
    0%, 100% { color: #ffedd5; }
    50% { color: #fb923c; }
}

    .text-gradient-title {
        position: relative;
        top: -100px;
        animation: text-gradient-title 10s ease-in-out infinite; /* Slowed down the animation */
        font-size: 70px;
        text-align: center;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5); /* Softer shadow for better legibility */
        font-style: italic;
    }
    .centered-image {
        display: flex;
        justify-content: center;
    }
    </style>

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)
#main title
st.markdown('<h1 class="text-gradient-title">BAD Molecule filter</h1>', unsafe_allow_html=True)


success_style = """
    background-color: #ffa500; /* Bright orange for more vibrancy */
    color: #ffffff; /* White text for contrast */
    border-radius: 5px; /* Rounded corners for a modern look */
    padding: 10px;
    width: 100%; /* Full width to stand out */
    font-size: 1rem; /* Use rem for scalability */
    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2); /* Subtle shadow for depth */
    animation-name: fadeOut;
    animation-duration: 5s;
"""

#footer
footer_style = """
    position: fixed;
    left: 0;
    z-index: 3;
    bottom: 0;
    width: 100%; /* Full width footer */
    color: #525354;
    font-style: italic;
    text-align: center; /* Centered text for a balanced look */
    padding: 10px;
    background: rgba(255, 255, 255, 0.8); /* Slightly transparent white background */
    font-size: 16px;
    border-top: 1px solid #ddd; /* Subtle border for separation */
"""
st.markdown(
    f'<div style="{footer_style}">Copyright (C) 2024 MolMod Lab</div>',
    unsafe_allow_html=True
)
menu_styles = {
    "container": {
        "padding": "10px 15px",  # Increase padding for more space
        "margin-bottom": "-50px",  # Pull the menu up closer to the title
        "background-color": "#f6f6f6",  # Neutral background for the container
        "border": "none",
        "box-shadow": "0 1px 3px rgba(0, 0, 0, 0.1)",
    },
    "icon": {
        "color": "#ff8c00",  # Softer orange color for the icons
        "font-size": "20px",  # Slightly reduced icon size
    },
    "nav-link": {
        "font-size": "18px",  # Reduced font size for a more refined look
        "padding": "10px 15px",  # Increase padding for a larger click area
        "--hover-color": "#ffaf66",  # Lighter orange for hover
        "color": "#333333",  # Dark text color for better readability
    },
    "nav-link-selected": {
        "background-color": "#ff8c00",  # Softer orange for selected item
        "color": "#000",  # White text for selected item
        "border-radius": "4px",  # Rounded corners for a modern touch
    }
}


if st.session_state.get('switch_button', False):
    st.session_state['menu_option'] = (st.session_state.get('menu_option', 0) + 1) % 2
    manual_select = st.session_state['menu_option']
else:
    manual_select = None
  
selected = option_menu(None, ["Home", "Single Molecule Prediction", "Batch calculation",  "Contact"],
                       icons=['house', "layers", "layers-fill",'cloud-upload', "capsule-pill","cloud-check", "question-diamond",'envelope-at'],
                       orientation="horizontal",
                       manual_select=manual_select,
                       key='menu_8',
                       default_index=0,
                       styles=menu_styles)
if selected == "Home":
    st.markdown(
    "<h1 style='text-align: center; font-size: 30px; font-style: italic; color: #333;'>"
    "Boosted Aggregation Detector: Decipher Molecule's aggregation status for more successful discoveries!</h1>", 
    unsafe_allow_html=True
)
    st.write('')
    st.markdown("<h1 style='text-align: center; fontSize: 23px; color: grey; font-weight: normal;'>Discover Molecules Aggregation predictions with our cutting-edge app.</h1>", unsafe_allow_html=True)
    st.write("---")
    st.subheader("How it Works")
    st.write("The BAD Molecule Filter leverages sophisticated Decision Tree Ensemble Methods to accurately classify Small Colloidally Aggregating Molecules (SCAMs). By enhancing the precision of SCAM detection, our tool significantly boosts hit rate optimization, reducing time and resources in early drug discovery stages.")
    st.write("In the menu above you can find the main sections:")
    st.write("""
        <ul>
            <li><strong>Single Molecule Prediction:</strong> Assess individual molecules with our robust model, which draws from comprehensive aggregation data.</li>
            <li><strong>Batch Calculation:</strong> Increase your efficiency with our batch processing feature, enabling aggregation status predictions for multiple molecules simultaneously.</li>
            <li><strong>Contact Us:</strong> Connect with the MolMod Lab team for support, feedback, or inquiries about the BAD Molecule Filter.</li>
        </ul>
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.write("""
        <ul>
            <li><strong>The deployed model achieved performance metrics that highlight its precision in classifying molecular aggregation. The following table display the Sensitivity and Specificity values that demonstrate the model's accuracy.</strong></li>
        </ul>
        """, unsafe_allow_html=True)

    custom_table_css = """
        <style>
            table {
                width: 60% !important; /* Set the width of the tables */
                margin-left: auto !important;
                margin-right: auto !important;
            }
            th {
                font-size: 1.1rem !important; /* Larger font size for headers */
                text-align: left !important;
            }
            td {
                font-size: 1rem !important; /* Larger font size for data */
            }
        </style>
    """
    st.markdown(custom_table_css, unsafe_allow_html=True)

    # Model details in a larger table without index
    model_details = {
        "Metric": ["Sensitivity", "Specificity"],
        "Value": ["95%", "76%"]
    }
    model_details_df = pd.DataFrame(model_details)
    st.write(model_details_df.to_html(index=False), unsafe_allow_html=True)

    # Space between tables, adjust as necessary
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("""
        <ul>
            <li><strong>The following tables display the model's outcomes and their definitions.</strong></li>
        </ul>
        """, unsafe_allow_html=True)
    # Outcome definitions in a larger table without index
    outcomes = [
        {"Outcome": "Aggregator", "Definition": "Identifies molecules with a high probability of aggregation."},
        {"Outcome": "Non-Aggregator", "Definition": "Identifies molecules with a low probability of aggregation."},
        {"Outcome": "Ambiguous", "Definition": "Indicates molecules with uncertain aggregation properties; further investigation is advised."}
    ]
    outcomes_df = pd.DataFrame(outcomes)
    st.write(outcomes_df.to_html(index=False), unsafe_allow_html=True)

   
# Single Molecule Prediction Section
elif selected == "Single Molecule Prediction":
    # Load the necessary model and scaler

    # Allow users to input or draw SMILES
    st.subheader("Input or Draw a Molecule")
    agree_draw_smiles = st.checkbox("Draw chemical structure")

    if agree_draw_smiles:
        # Drawing functionality
        smile_code = st_ketcher()
        smiles_input = smile_code  # Use the drawn molecule's SMILES
        st.markdown(f"SMILES: {smiles_input}")
    else:
        # Text input for SMILES
        smiles_input = st.text_input("Input SMILES", key="text")

    # Process the SMILES input or drawn molecule
    if smiles_input:
        col1, col2 = st.columns(2)
        try:
            molecule = Chem.MolFromSmiles(smiles_input)
            df = pd.DataFrame({'SMILES': [smiles_input]})
            if molecule:
                # Show the molecule's structure
                img = Draw.MolToImage(molecule)
                with col1:
                    st.image(img, caption='Chemical structure', use_column_width=True)

                with col2:
                    with st.spinner('Calculation in progress...'):
                        # Prepare the molecule for prediction
                        morgan_fingerprints = morgan_fpts(df['SMILES'])
                        mordred_descriptors = All_Mordred_descriptors(df['SMILES'])
                        Morgan_fingerprints_df = pd.DataFrame(morgan_fingerprints, columns=['Col_{}'.format(i) for i in range(morgan_fingerprints.shape[1])])
                        descriptors_combined = pd.concat([Morgan_fingerprints_df, mordred_descriptors], axis=1)

                        # descriptors_combined = np.concatenate([morgan_fingerprints, mordred_descriptors], axis=1)
                        
                        # Scale and predict
                        X_test = descriptors_combined[descriptor_columns_MM]

                        X_test_scaled = preprocess_and_scale_MM(X_test, scaler)
                        
                        prediction_proba = model.predict_proba(X_test_scaled)[:, 1]  # Assuming the second column is the probability of being an aggregator
                        
                        # Determine the outcome based on prediction
                        if prediction_proba < 0.3:
                            outcome = "Non-Aggregator"
                        elif prediction_proba > 0.7:
                            outcome = "Aggregator"
                        else:
                            outcome = "Ambiguous"

                        st.write(f"Prediction Outcome: **{outcome}**")

        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.write("---")
    st.subheader("Aggregators in Drug Discovery")
    st.markdown("""
    <h2 style='font-size: 1.5rem; color: #333;'>Aggregators in Drug Discovery</h2>
    <p>
        Aggregators represent a significant challenge in drug discovery, often manifesting as false positives that can obscure true findings. These compounds can pollute the chemical space, leading to considerable loss of time, effort, and resources. Understanding the nature of aggregators is crucial; our research indicates that their propensity to aggregate is closely linked to their complex chemical structures and amphiphilic properties, which drive their behavior in biological systems.
    </p>
""", unsafe_allow_html=True)

elif selected == "Batch calculation":
    st.markdown('<h1 class="text-second-title">Batch Mode Calculation</h1>', unsafe_allow_html=True)
    st.write('Upload a CSV file containing SMILES strings for batch prediction of aggregation status.')
    st.markdown("""
    - The file must be in CSV format.
    - The first row should contain column headers.
    - There must be a column named `SMILES` which contains the SMILES strings of the molecules for which you want to predict the aggregation status.
    - Each row after the header should contain one molecule's SMILES string in the `SMILES` column.
    - Ensure there are no empty rows or columns in your file.
    """)
    st.markdown('**Example of the CSV file format:**')
    st.markdown("""
    ```plaintext
    SMILES
    CCO
    c1ccccc1
    CCN(CC)CC
    ```
    """)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data_file = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(data_file.head())
        if 'SMILES' in data_file.columns:
            predictions = []
            for smi in data_file["SMILES"]:
                try:
                    morgan_fingerprints = morgan_fpts([smi])
                    mordred_descriptors = All_Mordred_descriptors([smi])
                    morgan_fingerprints_df = pd.DataFrame(morgan_fingerprints, columns=[f'Col_{i}' for i in range(morgan_fingerprints.shape[1])])
                    descriptors_df = pd.concat([morgan_fingerprints_df, mordred_descriptors], axis=1).fillna(0)
                    X_test = descriptors_df[descriptor_columns_MM]
                    X_test_scaled = preprocess_and_scale_MM(X_test, scaler)
                    prediction_proba = model.predict_proba(X_test_scaled)[:, 1]
                    for proba in prediction_proba:
                        if proba < 0.3:
                            outcome = "Non-Aggregator"
                        elif proba > 0.7:
                            outcome = "Aggregator"
                        else:
                            outcome = "Ambiguous"
                    predictions.append(outcome)                    
                except Exception as e:
                    predictions.append(f"Error: {str(e)}")
            data_file['Prediction'] = predictions
            st.write("Predictions:")
            st.dataframe(data_file)
            
            # Convert DataFrame to CSV for download
            csv = data_file.to_csv(index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name='batch_predictions.csv',
                mime='text/csv',
            )
        else:
            st.error("The uploaded file does not contain a 'SMILES' column.")
    

elif selected == "Contact":
    st.markdown('<h1 class="text-second-title">Contact</h1>', unsafe_allow_html=True)
    st.write("My name is **Abdallah Abou Hajal** and I am the author of ***BAD Molecule Filter***. I hope this application will help in your drug discovery campaigns.")
    st.write("I am a member of the Molecular Modeling Lab team at Al Ain University. We are a group of passionate researchers. If you're interested in developing AI-based drug discovery tools or collaborating on research projects, feel free to contact me!")

    # Email and clickable link to the lab's website
    st.markdown("Email: [Abdallah.Abouhajal@gmail.com](mailto:Abdallah.Abouhajal@gmail.com)")
    st.markdown('Visit our lab\'s website to meet the rest of the team and check out our latest works: [MolModLab-AAU](http://MolModLab-aau.com)', unsafe_allow_html=True)


