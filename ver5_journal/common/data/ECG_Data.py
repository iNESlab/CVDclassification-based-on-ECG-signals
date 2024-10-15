import os
import numpy as np
import torch
import torchaudio
import wfdb
from torch.utils.data import Dataset
from dotenv import load_dotenv

# 현재 파일의 위치를 기준으로 경로를 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '../../.env')

# .env 파일 로드
load_dotenv(env_path)


class ECG_Data(Dataset):
    def __init__(self, dataframe, base_path=os.getenv("DATA_PATH")):
        if base_path is None:
            raise ValueError("환경 변수 DATA_PATH가 설정되지 않았거나 올바르지 않습니다.")

        self.data = dataframe
        self.base_path = base_path  # 경로를 주입받도록 변경
        self.n_mels = 20
        self.hop_length = 21  # 임의의 값, 조정 가능
        self.win_length = 39  # 임의의 값, 조정 가능
        self.mel_specgram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=100,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            normalized=True,
            power=1
        )
        self.stft_spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.win_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            normalized=True,
            power=1
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # 경로 처리
        path = self.base_path + "/" + self.data['filename_lr'][idx]
        file_audio = wfdb.rdsamp(path)
        data = file_audio
        data_new = np.array(data[0], dtype=np.float32)  # float32로 변환
        data_new = np.transpose(data_new, (1, 0))
        data_final = torch.Tensor(data_new)
        print(f'data_new : {data_new.shape}')
        # 레이블 처리: NORM, STTC, MI, HYP, CD 열을 하나의 벡터로 결합
        label_columns = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        label = self.data.loc[idx, label_columns].values.astype(np.float32)
        print(f'label : {label.shape}')

        # 레이블 값 검증
        if not np.all(np.isfinite(label)):  # 숫자가 아닌 값이 있으면 에러 처리
            raise ValueError(f"Invalid label value found at index {idx}: {label}")

        if not np.issubdtype(label.dtype, np.number):  # 숫자가 아닌 타입이 있을 경우
            raise ValueError(f"Label contains non-numeric data at index {idx}: {label}")

        label = torch.Tensor(label).float()  # numpy array를 torch.Tensor로 변환

        return data_final, label
