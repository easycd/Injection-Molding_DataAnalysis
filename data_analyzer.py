"""
=====================================================================
data_analyzer.py
---------------------------------------------------------------------
'데이터'와 관련된 모든 로직을 담당하는 파일 (UI 코드 없음).
    - CSV 로드
    - 전처리 (결측치 처리 / 라벨 인코딩 / 스케일링)
    - 시각화용 그래프(matplotlib Figure) 생성
    - 모델 학습 / 평가 / 예측

ui.py는 이 파일의 DataAnalyzer 클래스를 가져다 쓰기만 하고,
화면(위젯)을 그리는 역할만 담당한다.
=====================================================================
"""

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")  # matplotlib 그래프를 tkinter 창 위에 그리기 위한 백엔드 설정
# 그래프 안에 한글(제목/축이름)이 깨져서(네모로) 보이지 않도록 한글 폰트로 지정
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False  # 한글 폰트 사용 시 마이너스 기호 깨짐 방지
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 회귀 모델들
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# 평가 지표들
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


# =====================================================================
# 회귀(연속값 예측)에 사용할 모델 목록
# key   : 화면 콤보박스에 표시될 이름
# value : 실제 sklearn 모델 객체를 만들어주는 함수(람다)
# =====================================================================
REGRESSION_MODELS = {
    "선형 회귀 (Linear Regression)": lambda: LinearRegression(),
    "의사결정나무 회귀 (Decision Tree)": lambda: DecisionTreeRegressor(random_state=42),
    "랜덤포레스트 회귀 (Random Forest)": lambda: RandomForestRegressor(random_state=42),
    "서포트벡터 회귀 (SVR)": lambda: SVR(),
    "K최근접이웃 회귀 (KNN)": lambda: KNeighborsRegressor(),
}

# =====================================================================
# 분류(범주 예측)에 사용할 모델 목록
# =====================================================================
CLASSIFICATION_MODELS = {
    "로지스틱 회귀 (Logistic Regression)": lambda: LogisticRegression(max_iter=1000),
    "의사결정나무 분류 (Decision Tree)": lambda: DecisionTreeClassifier(random_state=42),
    "랜덤포레스트 분류 (Random Forest)": lambda: RandomForestClassifier(random_state=42),
    "서포트벡터 분류 (SVC)": lambda: SVC(probability=True),
    "K최근접이웃 분류 (KNN)": lambda: KNeighborsClassifier(),
}


class DataAnalyzer:
    """
    CSV 로드부터 전처리, 시각화, 모델 학습/평가/예측까지
    '데이터 처리'와 관련된 모든 상태와 로직을 가지고 있는 클래스.
    tkinter 관련 코드는 전혀 포함하지 않는다 (UI 독립적).
    """

    def __init__(self):
        # -----------------------------------------------------------
        # 분석 과정 전반에서 공유되는 상태(데이터) 변수들
        # -----------------------------------------------------------
        self.raw_df = None          # CSV에서 읽은 원본 데이터
        self.df = None              # 전처리가 적용된 데이터

        self.encoders = {}          # 컬럼별 라벨인코더 저장 (예측 시 재사용)
        self.scaler = None          # 스케일러 객체 (예측 시 재사용)
        self.scaled_columns = []    # 스케일링이 적용된 컬럼 목록

        self.feature_columns = []   # 모델 입력으로 사용할 컬럼(X)
        self.target_column = None   # 예측 대상 컬럼(y)
        self.problem_type = "회귀"  # "회귀" 또는 "분류"

        self.model = None           # 학습이 끝난 모델 객체
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_pred_test = None     # 테스트셋 예측 결과 (평가용)

    # =================================================================
    # 1) 파일 로드
    # =================================================================
    def load_csv(self, file_path):
        """CSV 파일을 읽어 raw_df / df에 저장한다. 인코딩은 utf-8 -> cp949 순으로 시도."""
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="cp949")

        self.raw_df = df
        self.df = df.copy()  # 전처리는 원본을 보존한 채 복사본에 적용
        return self.df

    def get_summary_text(self):
        """행/열 개수, 컬럼별 타입, 결측치 개수를 문자열로 요약해서 반환한다."""
        df = self.df
        lines = [f"행 개수: {df.shape[0]}   열 개수: {df.shape[1]}", ""]
        lines.append(f"{'컬럼명':<20}{'타입':<12}{'결측치개수':<10}")
        for col in df.columns:
            lines.append(f"{col:<20}{str(df[col].dtype):<12}{df[col].isnull().sum():<10}")
        return "\n".join(lines)

    # =================================================================
    # 2) 전처리
    # =================================================================
    def apply_missing_handling(self, method):
        """
        결측치를 처리한다.
        method: "평균값으로 채우기" / "중앙값으로 채우기" / "최빈값으로 채우기" / "결측치 행 삭제"
        """
        df = self.df

        if method == "결측치 행 삭제":
            df = df.dropna()
        else:
            strategy_map = {
                "평균값으로 채우기": "mean",
                "중앙값으로 채우기": "median",
                "최빈값으로 채우기": "most_frequent",
            }
            strategy = strategy_map[method]

            # 수치형 컬럼은 선택한 방법, 문자형 컬럼은 항상 최빈값으로 채운다.
            numeric_cols = df.select_dtypes(include=np.number).columns
            object_cols = df.select_dtypes(exclude=np.number).columns

            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy=strategy)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

            if len(object_cols) > 0:
                imputer = SimpleImputer(strategy="most_frequent")
                df[object_cols] = imputer.fit_transform(df[object_cols])

        self.df = df
        return self.df

    def apply_label_encoding(self, columns):
        """선택된 문자형 컬럼들을 숫자로 변환(라벨 인코딩)한다."""
        for col in columns:
            encoder = LabelEncoder()
            self.df[col] = encoder.fit_transform(self.df[col].astype(str))
            self.encoders[col] = encoder  # 예측 단계에서 동일한 규칙으로 변환하기 위해 저장
        return self.df

    def apply_scaling(self, columns, method):
        """
        사용자가 선택한 수치형 컬럼(columns)에 표준화 또는 정규화를 적용한다.
        method: "표준화 (StandardScaler)" / "정규화 (MinMaxScaler)"
        분류 문제의 타겟 컬럼처럼 스케일링하면 안 되는 컬럼은 선택하지 않아야 한다.
        """
        self.scaler = StandardScaler() if "표준화" in method else MinMaxScaler()
        self.df[columns] = self.scaler.fit_transform(self.df[columns])
        self.scaled_columns = list(columns)
        return self.df

    def get_object_columns(self):
        """문자형(인코딩 대상 후보) 컬럼 목록."""
        return list(self.df.select_dtypes(exclude=np.number).columns)

    def get_numeric_columns(self):
        """수치형(스케일링 대상 후보) 컬럼 목록."""
        return list(self.df.select_dtypes(include=np.number).columns)

    # =================================================================
    # 3) 시각화 (matplotlib Figure 객체를 만들어서 반환)
    #    실제로 화면에 그리는 것은 ui.py의 몫이고, 여기서는 Figure만 생성한다.
    # =================================================================
    def make_histogram_figure(self, col):
        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.hist(self.df[col].dropna(), bins=30, color="#4C72B0", edgecolor="white")
        ax.set_title(f"'{col}' 히스토그램")
        ax.set_xlabel(col)
        ax.set_ylabel("빈도수")
        return fig

    def make_boxplot_figure(self, col):
        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.boxplot(self.df[col].dropna())
        ax.set_title(f"'{col}' 박스플롯")
        return fig

    def make_scatter_figure(self, x_col, y_col):
        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.scatter(self.df[x_col], self.df[y_col], alpha=0.6, color="#DD8452")
        ax.set_title(f"{x_col} vs {y_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        return fig

    def make_corr_heatmap_figure(self):
        numeric_df = self.df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
            raise ValueError("상관관계를 그리려면 수치형 컬럼이 2개 이상 필요합니다.")

        corr = numeric_df.corr()

        fig = Figure(figsize=(7, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)

        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)

        fig.colorbar(im, ax=ax)
        ax.set_title("상관관계 히트맵")
        fig.tight_layout()
        return fig

    # =================================================================
    # 4) 모델링
    # =================================================================
    def train_model(self, target, features, problem_type, model_name, test_size):
        """
        target/features 컬럼으로 학습 데이터를 구성하고, 선택된 알고리즘으로 모델을 학습한다.
        학습 결과(모델, train/test 데이터, 예측값)는 인스턴스 상태로 저장된다.
        """
        if target in features:
            raise ValueError("Target 컬럼은 Feature 목록에서 제외해야 합니다.")

        # 학습에 사용할 데이터는 결측치가 없어야 하므로, 남아있는 결측 행은 제거하고 진행
        model_df = self.df[features + [target]].dropna()
        if model_df.empty:
            raise ValueError("학습에 사용할 데이터가 없습니다. 전처리를 확인해주세요.")

        # 문자형 컬럼이 인코딩 없이 섞여 있으면 학습이 불가능하므로 사전에 확인
        non_numeric = model_df[features].select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric:
            raise ValueError(f"다음 입력 변수는 문자형입니다. 먼저 인코딩해주세요.\n{non_numeric}")

        X = model_df[features]
        y = model_df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model_dict = REGRESSION_MODELS if problem_type == "회귀" else CLASSIFICATION_MODELS
        model = model_dict[model_name]()
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)

        # 다른 단계(평가, 예측)에서 재사용할 수 있도록 상태 저장
        self.model = model
        self.feature_columns = features
        self.target_column = target
        self.problem_type = problem_type
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.y_pred_test = y_pred_test

        return {
            "train_count": len(X_train),
            "test_count": len(X_test),
        }

    # =================================================================
    # 5) 평가
    # =================================================================
    def evaluate_regression(self):
        """회귀 모델 평가: RMSE / MAE / R2 지표와 실제값-예측값 Figure를 반환."""
        y_test, y_pred = self.y_test, self.y_pred_test

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        text = (
            f"[회귀 모델 평가 결과]\n"
            f"RMSE (평균제곱근오차): {rmse:.4f}\n"
            f"MAE  (평균절대오차)  : {mae:.4f}\n"
            f"R2   (결정계수)      : {r2:.4f}\n"
        )

        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.scatter(y_test, y_pred, alpha=0.6, color="#55A868")
        min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--", label="이상적 예측선")
        ax.set_xlabel("실제값")
        ax.set_ylabel("예측값")
        ax.set_title("실제값 vs 예측값")
        ax.legend()

        return text, fig

    def evaluate_classification(self):
        """분류 모델 평가: Accuracy/Precision/Recall/F1 지표와 혼동행렬 Figure를 반환."""
        y_test, y_pred = self.y_test, self.y_pred_test

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        text = (
            f"[분류 모델 평가 결과]\n"
            f"정확도 (Accuracy) : {acc:.4f}\n"
            f"정밀도 (Precision): {prec:.4f}\n"
            f"재현율 (Recall)   : {rec:.4f}\n"
            f"F1 Score          : {f1:.4f}\n"
        )

        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(pd.unique(y_test))

        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("예측값")
        ax.set_ylabel("실제값")
        ax.set_title("혼동행렬 (Confusion Matrix)")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        return text, fig

    # =================================================================
    # 6) 예측
    # =================================================================
    def predict(self, input_values):
        """
        input_values: {컬럼명: 문자열입력값} 형태의 dict.
        학습 때와 동일한 인코딩/스케일링 규칙을 적용한 뒤 예측값을 반환한다.
        """
        if self.model is None:
            raise ValueError("먼저 모델을 학습해주세요.")

        row = {}
        for col in self.feature_columns:
            raw_value = input_values.get(col, "")
            if raw_value == "":
                raise ValueError(f"'{col}' 값이 비어있습니다.")

            # 학습 단계에서 라벨 인코딩된 컬럼이면 동일한 인코더로 문자->숫자 변환
            if col in self.encoders:
                row[col] = self.encoders[col].transform([raw_value])[0]
            else:
                row[col] = float(raw_value)

        input_df = pd.DataFrame([row], columns=self.feature_columns)

        # 학습 때 스케일링을 적용한 컬럼이 입력 변수에 포함되어 있다면 동일하게 변환
        common_scaled = [c for c in self.feature_columns if c in self.scaled_columns]
        if self.scaler is not None and common_scaled:
            input_df[common_scaled] = self.scaler.transform(input_df[common_scaled])

        prediction = self.model.predict(input_df)[0]
        return prediction
