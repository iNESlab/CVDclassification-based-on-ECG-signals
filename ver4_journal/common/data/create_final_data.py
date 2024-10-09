# final_data 생성 (이진 분류시와 범주형 분류시 데이터 전처리)
import os

import pandas as pd
import seaborn as sns
import re

from dotenv import load_dotenv

# 현재 파일의 위치를 기준으로 경로를 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '../../.env')

# .env 파일 로드
load_dotenv(env_path)

ecg_data=pd.read_csv(f'{os.getenv("DATA_PATH")}/ptbxl_database.csv')
diagnostics={
    "NORM":['NORM','CSD'],
    "STTC":['NDT', 'NST_', 'DIG', 'LNGQT', 'ISC_', 'ISCAL', 'ISCIN', 'ISCIL', 'ISCAS', 'ISCLA', 'ANEUR', 'EL', 'ISCAN' ],
    "MI":['IMI', 'ASMI', 'ILMI', 'AMI', 'ALMI', 'INJAS', 'LMI', 'INJAL', 'IPLMI', 'IPMI', 'INJIN', 'INJLA', 'PMI', 'INJIL'],
    "HYP":['LVH', 'LAO/LAE', 'RVH', 'RAO/RAE', 'SEHYP'],
    "CD":['LAFB', 'IRBBB', '1AVB', 'IVCD', 'CRBBB', 'CLBBB', 'LPFB', 'WPW', 'ILBBB', '3AVB', '2AVB'],
    "OTHER":['AFLT', 'AFIB', 'PSVT', 'STACH', 'PVC', 'PACE', 'PAC']
}
def create_final_data(temp=None, dd=None):
    # Merging the diagnosis into a superclass:
    labels = []
    discard = []
    for index in range(ecg_data.shape[0]):
        counter = 0
        temp_diag = ecg_data['scp_codes'][index]
        temp_diag = re.sub('{', "", str(temp_diag))
        temp_diag = re.sub('}', "", temp_diag)
        temp_diag = temp_diag.split(',')
        len_diag = len(temp_diag)
        for idx in range(len_diag):
            temp_d = temp_diag[idx]
            temp_d = temp_d.split(':')[0]
            temp_d = re.sub(r'[^\w\s]', "", temp_d)
            if temp_d in diagnostics['NORM']:
                label = 0 if dd is None else temp[dd][0]
                counter = 1
            elif temp_d in diagnostics['STTC']:
                label = 1 if dd is None else temp[dd][1]
                counter = 1
            elif temp_d in diagnostics['MI']:
                label = 2 if temp is None else temp[dd][2]
                counter = 1
            elif temp_d in diagnostics['HYP']:
                label = 3 if temp is None else temp[dd][3]
                counter = 1
            elif temp_d in diagnostics['CD']:
                label = 4 if temp is None else temp[dd][4]
                counter = 1
            elif temp_d in diagnostics['OTHER']:
                label = 100
            else:
                label = 100
            labels.append(label)
        if counter == 0:
            discard.append(index)

    final_labels = []
    for index in range(len(labels)):
        if labels[index] != 100:
            final_labels.append(labels[index])

    final_data = ecg_data.drop(axis=0, index=discard)
    final_data['Labels'] = final_labels
    sns.countplot(data=final_data, x='Labels')

    return final_data