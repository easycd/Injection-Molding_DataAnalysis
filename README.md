# 앞유리 사이드 몰딩 데이터 분석 프로그램

사출성형 공정 CSV를 불러와 데이터 품질을 확인하고, 그래프로 탐색한 뒤 불량 여부를 예측하는 Tkinter 프로그램입니다.

## 실행 준비

PyCharm의 Terminal에서 다음 명령을 실행합니다.

```bash
pip install -r requirements.txt
python main.py
```

## 사용 순서

1. 왼쪽의 **열기** 버튼으로 `project_data.csv`를 선택합니다.
2. `데이터 입력`, `전처리`, `시각화(EDA)` 탭에서 데이터 상태를 확인합니다.
3. 모델과 테스트 비율을 선택하고 **모델 학습**을 누릅니다.
4. `평가` 탭에서 Accuracy, Precision, Recall, F1-score, ROC-AUC와 혼동행렬을 확인합니다.
5. 평가 화면의 **불량 검출 데이터 저장 (.txt)** 버튼으로 불량 예측 행을 메모장 파일로 저장할 수 있습니다.
6. `예측 결과` 탭에서 공정값을 입력하고 **예측 실행**을 누릅니다.

저장되는 텍스트에는 원본 행 번호, 실제 라벨, 예측 라벨, 불량 확률과 모든 공정변수가 탭으로 구분되어 포함됩니다.

화면에는 처음 보는 사람도 이해할 수 있도록 영문 변수명과 한국어 의미를 함께 표시합니다. 하단 상태 안내와 각 탭의 설명을 순서대로 따라가면 됩니다.

## 코드 구성 설명

역할별 클래스를 불러와 사용할 수 있도록 네 파일로 분리했습니다.

- `main.py`: `MoldingAnalysisApp` UI 클래스를 불러와 프로그램을 실행합니다.
- `ui.py`: `MoldingAnalysisApp` 클래스와 Tkinter 화면 표시를 담당합니다.
- `data_analyzer.py`: `DataAnalyzer` 클래스가 CSV 정리, 저장 모델 평가 및 예측을 연결합니다.
- `pretrained_models.py`: SMOTE를 제거한 두 PKL 모델을 불러오고 입력 열을 변환합니다.
- `models/`: Random Forest와 Gradient Boosting 저장 모델이 있습니다.
- `TrainingResult`: 학습 결과를 UI에 전달하는 데이터 클래스입니다.

선택할 수 있는 분류 모델은 다음 2개입니다.

- Gradient Boosting
- Random Forest

클래스를 다른 파일에서 사용하는 기본 형태는 다음과 같습니다.

```python
from ui import MoldingAnalysisApp

app = MoldingAnalysisApp()
app.mainloop()
```

화면 없이 분석 기능만 사용할 수도 있습니다.

```python
from data_analyzer import DataAnalyzer

analyzer = DataAnalyzer()
analyzer.load_csv("project_data.csv")
result = analyzer.train("Random Forest", test_size=0.2, cv_count=5)
print(result.metrics)
```

## 핵심 분석 개념

### 독립변수와 종속변수

- 종속변수 `y`: `PassOrFail`
- 독립변수 `X`: 시간, 압력, 속도, 위치, 온도에 해당하는 공정 변수
- `Unnamed: 0`과 같은 행 번호 열이 있으면 모델에서 제외
- `_id`, `ID`, `TimeStamp`, 날짜, 시리얼 번호처럼 품질 예측에 필요 없는 관리용 열은 자동 제외
- `Injection_Time`, `Cycle_Time` 같은 실제 사출 공정시간 변수는 유지
- 모든 값이 같은 상수 열이 있으면 자동 제외

### 학습 데이터와 평가 데이터

`train_test_split()`은 데이터를 모델이 공부할 부분과 성능을 확인할 부분으로 나눕니다. `stratify=y`는 양품과 불량의 비율이 양쪽에 최대한 유지되게 합니다.

프로그램에서는 이미 학습이 끝난 PKL 모델을 불러와 사용합니다. 저장 모델에서 SMOTE 단계는 제거했으며 예측 파이프라인에는 전처리기와 분류기만 들어 있습니다. 기본 판정 임계값은 0.5입니다.

### 클래스 분포

데이터의 양품·불량 분포를 확인하고 다음 지표를 함께 평가합니다. Random Forest에는 `class_weight="balanced"`를 사용하고, Gradient Boosting에는 같은 목적의 표본 가중치를 적용합니다.

- Precision: 불량이라고 예측한 것 중 실제 불량의 비율
- Recall: 실제 불량 중 모델이 찾아낸 비율
- F1-score: Precision과 Recall을 함께 고려한 점수
- ROC-AUC: 분류 기준을 바꿨을 때 두 클래스를 구분하는 능력

학습·평가 데이터의 클래스 비율이 원본과 비슷하게 유지되도록 `stratify=y`를 적용합니다.

## 주의사항

- 원본 `PassOrFail`의 `Y=양품`, `N=불량`을 내부에서 `0=양품`, `1=불량`으로 자동 변환합니다. 이미 0/1인 CSV도 지원합니다.
- 숫자형 변수는 모델 파이프라인 내부에서 `StandardScaler`로 표준화하고, 문자형 변수는 One-Hot Encoding합니다.
- CSV 원본과 모델 학습에는 전체 정밀도를 사용하고, 화면에는 읽기 쉽도록 소수점 셋째 자리까지 표시합니다.
- 중복 데이터가 있으므로 모델 평가 결과가 과대평가되지 않는지 함께 확인해야 합니다.
- 예측 결과는 실제 현장의 품질검사를 대체하지 않습니다.
