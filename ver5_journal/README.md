# 🙌 파일 순서
1. 전처리 : pre_processing.ipynb
2. 학습 : `./process/*`
3. 테스트 : `./test/*`

    학습시 나눈 두 가지(BCE/weight_BCE Loss)케이스를 각각 다른 Lossfunction으로 테스트

### ♻️ 전처리
- 데이터셋의 scp_codes 클래스의 multi-label들을 super-label (`'NORM', 'MI', 'STTC', 'CD', 'HYP'`)로 묶음
- super-label로 묶을 때, 기존 레이블의 수치가 0인 경우 제외 (ex. `{'NORM': 100.0, 'SR': 0.0}` 의 경우, SR 삭제)
- super-label을 binary-class로 변경하여 기존 데이터셋의 열에 concatenate (ex. NORM, STTC, MI 클래스를 데이터셋에 col=1로 추가)
![img](https://github.com/user-attachments/assets/fffe6e07-7faa-4769-8b4b-f841fe6c23a6)

### 📖 학습
- LossFunction을 단순 BCE로 학습(에폭 100) : 기존 50으로 학습 후, 정확도를 높이기 위해 100으로 변경
- LossFunction을 weight BCE로 학습(에폭 50) : 새로 추가
- BCResNet을 분류기 하나로 학습 : one-stage-BCResNet

### 🧪 테스트
- 정확도는 weight_BCE_test가 BCE_test보다 0.08%p 더 낮음  
- AUC는 weight_BCE가 전체적으로 높음.
- 에폭을 감안했을 때, weight_BCE가 더 개선 가능성이 있음.
- 다중 분류기 파라미터가 단일 분류기 파라미터보다 약 5배 더 많음.
- 대신, 다단계 분류기가 AUC(약 1.9%p)와 HammingLoss(약 3.2%p)가 더 좋음.
