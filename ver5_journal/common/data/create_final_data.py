import os
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

# 현재 파일의 위치를 기준으로 경로를 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '../../.env')

# .env 파일 로드
load_dotenv(env_path)

# 데이터 로드
ecg_data = pd.read_csv(f'{os.getenv("DATA_PATH")}/ptbxl_database.csv')

# 진단 코드를 레이블로 매핑
diagnostics = {
    "NORM": ['NORM', 'CSD'],
    "MI": ['IMI', 'ASMI', 'ILMI', 'AMI', 'ALMI', 'INJAS', 'LMI', 'INJAL', 'IPLMI', 'IPMI', 'INJIN', 'INJLA', 'PMI',
           'INJIL'],
    "STTC": ['NDT', 'NST_', 'DIG', 'LNGQT', 'ISC_', 'ISCAL', 'ISCIN', 'ISCIL', 'ISCAS', 'ISCLA', 'ANEUR', 'EL',
             'ISCAN'],
    "CD": ['LAFB', 'IRBBB', '1AVB', 'IVCD', 'CRBBB', 'CLBBB', 'LPFB', 'WPW', 'ILBBB', '3AVB', '2AVB'],
    "HYP": ['LVH', 'LAO/LAE', 'RVH', 'RAO/RAE', 'SEHYP'],
    "OTHER": ['AFLT', 'AFIB', 'PSVT', 'STACH', 'PVC', 'PACE', 'PAC']
}

# 진단 코드를 레이블로 매핑하는 구조 생성
diagnostic_mapping = {}
for label, codes in diagnostics.items():
    for code in codes:
        diagnostic_mapping[code] = label

# 범주형 레이블 목록 정의
label_columns = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def create_final_data(data=ecg_data):
    labels = []
    # discard = []

    # 각 레코드에 대해 레이블 매핑
    for index in range(data.shape[0]):
        temp_labels = {label: 0 for label in label_columns}  # 레이블 초기화 (0으로)
        temp_diag = data['scp_codes'][index]

        # 데이터가 dict 형태이므로, 키와 값을 추출
        try:
            temp_diag = eval(temp_diag)  # string으로 저장된 dict를 실제 dict로 변환
        except:
            print(f'{temp_labels}')
            ValueError(f'데이터{index}가 dict 형태가 아닙니다. ')
            continue

        for diag_code, value in temp_diag.items():
            # 값이 0인 항목은 제외
            if value == 0.0:
                continue

            diag_code = diag_code.strip()  # 공백 제거

            # 매핑된 레이블에 따라 multi-label로 설정
            if diag_code in diagnostic_mapping:
                category = diagnostic_mapping[diag_code]
                if category in temp_labels:
                    temp_labels[category] = 1  # 해당 레이블에 1 할당

        labels.append(temp_labels) # OTHER 데이터들을 삭제하지 않음

    # 레이블 데이터프레임으로 변환하여 병합
    labels_df = pd.DataFrame(labels)
    final_data = pd.concat([data, labels_df], axis=1)

    # 레이블 분포 시각화
    sns.set(style="whitegrid")
    sns.countplot(data=final_data.melt(id_vars=["ecg_id"], value_vars=label_columns),
                  x="variable", hue="value")

    return final_data
