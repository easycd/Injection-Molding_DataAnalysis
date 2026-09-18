"""CSV 정리와 custom_models.py 모델을 UI에 연결하는 기능 클래스."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from pretrained_models import PretrainedModelManager


TARGET_COLUMN = "PassOrFail"
INDEX_COLUMNS = ["Unnamed: 0"]

## 제거할 칼럼
UNUSED_COLUMNS = [
    "_id", "TimeStamp", "PART_FACT_PLAN_DATE", "PART_FACT_SERIAL",
    "PART_NAME", "EQUIP_CD", "Reason", "ID", "INDEX", "DateTime",
    "DATE", "SERIAL", "SERIAL_NO", "PART_ID", "PRODUCT_ID",
]

## 데이터 입출력, UI용 분석 결과 관리
class DataAnalyzer:
    def __init__(self) -> None:
        self.raw_df: pd.DataFrame | None = None
        self.analysis_df: pd.DataFrame | None = None
        self.feature_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.model = None
        self.model_manager: PretrainedModelManager | None = None
        self.threshold = 0.5

    ## 원본의 Y/N 또는 가공된 0/1 라벨을 0 = 양품, 1 = 불량으로 변환
    @staticmethod
    def _convert_target(series: pd.Series) -> pd.Series:
        non_null = set(series.dropna().unique())
        if non_null.issubset({0, 1}):
            return pd.to_numeric(series, errors="coerce")
        normalized = series.astype("string").str.strip().str.upper()
        converted = normalized.map({"Y": 0, "N": 1, "0": 0, "1": 1})
        invalid = series.notna() & converted.isna()
        if invalid.any():
            bad_values = sorted(series[invalid].astype(str).unique().tolist())
            raise ValueError(f"PassOrFail에서 지원하지 않는 값이 발견됐습니다: {bad_values}")
        return converted.astype("Int64")

    @staticmethod
    def _metadata_columns(columns) -> list[str]:
        ## ID, 날짜, 시리얼넘버 같은 관리용 열을 찾고, 공정시간 열을 유지
        exact_names = {
            name.lower() for name in INDEX_COLUMNS + UNUSED_COLUMNS
        }
        result = []
        for column in columns:
            normalized = str(column).strip().lower()
            is_unnamed_index = normalized.startswith("unnamed:")
            is_identifier = normalized.endswith(("_id", "_serial", "_serial_no"))
            is_record_time = normalized in {
                "time", "timestamp", "time_stamp", "datetime", "date",
                "created_at", "updated_at", "recorded_at", "measurement_time",
            }
            if normalized in exact_names or is_unnamed_index or is_identifier or is_record_time:
                result.append(column)
        return result

    ## CSV를 읽고 PassOrFail의 Y/N 값을 _convert_target 함수로 내부 0/1값으로 변환
    def load_csv(self, path: str) -> pd.DataFrame:
        """CSV를 읽고 PassOrFail의 Y/N 값을 내부 0/1 값으로 변환합니다."""
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp949")
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"'{TARGET_COLUMN}' 열이 없습니다.")
        df[TARGET_COLUMN] = self._convert_target(df[TARGET_COLUMN])
        self.raw_df = df.copy()
        self._prepare_data()
        return self.raw_df

    ## 모델과 분석에 사용할 수 있도록 데이터를 정리하는 전처리
    def _prepare_data(self) -> None:
        if self.raw_df is None:
            raise ValueError("먼저 CSV 파일을 불러오세요.")

        ## 불필요 상수 열을 제거하고 숫자형과 범주형 변수를 구분
        excluded = self._metadata_columns(self.raw_df.columns)
        df = self.raw_df.drop(columns=excluded).copy()
        df = df.dropna(subset=[TARGET_COLUMN])
        schema = PretrainedModelManager("Random Forest").expected_features
        if set(schema).issubset(df.columns):
            candidates = list(schema)
        else:
            raw_schema = [
                feature.removeprefix("num__")
                for feature in schema if feature.startswith("num__")
            ]
            if any(feature.startswith("cat__EQUIP_NAME_") for feature in schema):
                raw_schema.append("EQUIP_NAME")
            candidates = [column for column in raw_schema if column in df.columns]
        constant_columns = [column for column in candidates if df[column].nunique(dropna=True) <= 1]
        df = df.drop(columns=constant_columns)
        candidates = [column for column in candidates if column not in constant_columns]

        self.numeric_columns = df[candidates].select_dtypes(include=np.number).columns.tolist()
        self.categorical_columns = [column for column in candidates if column not in self.numeric_columns]

        for column in self.numeric_columns:
            df[column] = df[column].fillna(df[column].median())
        for column in self.categorical_columns:
            mode = df[column].mode(dropna=True)
            fill_value = str(mode.iloc[0]) if not mode.empty else "Unknown"
            df[column] = df[column].fillna(fill_value).astype(str)

        self.feature_columns = self.numeric_columns + self.categorical_columns
        self.analysis_df = df[[TARGET_COLUMN] + self.feature_columns]

    ## _prepare_data에서 전처리한 데이터 요약
    def preprocessing_summary(self) -> dict:
        if self.raw_df is None or self.analysis_df is None:
            raise ValueError("먼저 CSV 파일을 불러오세요.")
        excluded = self._metadata_columns(self.raw_df.columns)
        candidate = self.raw_df.drop(columns=excluded + [TARGET_COLUMN], errors="ignore")
        constants = [column for column in candidate.columns if candidate[column].nunique(dropna=True) <= 1]
        return {
            "excluded": excluded,
            "constants": constants,
            ## 전체 결측값 개수 계산
            "missing": int(self.raw_df.isna().sum().sum()),
            ## 양품과 불량 개수 계산
            "counts": self.analysis_df[TARGET_COLUMN].value_counts().sort_index(),
        }

    ## 각 공정 변수와 품질 판정(PassOrFail) 사이의 상관계수를 계산
    def correlations(self) -> pd.Series:
        if self.analysis_df is None:
            raise ValueError("먼저 CSV 파일을 불러오세요.")
        return self.analysis_df.corr(numeric_only=True)[TARGET_COLUMN].drop(TARGET_COLUMN)

    ## 학습된 PKL 모델을 불러오고 예측 준비
    def load_model(self, model_name: str) -> None:
        manager = PretrainedModelManager(model_name)
        self.model_manager = manager
        self.model = manager.model
        self.threshold = 0.5

    ## 불러온 CSV파일 전체로 Gradient Boosting, Random Forest 모델 성능 계산
    def evaluate_saved_model(self, model_name: str) -> dict[str, float]:
        if self.analysis_df is None:
            raise ValueError("먼저 CSV 파일을 불러오세요.")
        manager = PretrainedModelManager(model_name)
        X = self.analysis_df[self.feature_columns]
        y = self.analysis_df[TARGET_COLUMN].astype(int)
        probability = manager.predict_proba(X)
        prediction = (probability >= 0.5).astype(int)
        return {
            "Accuracy": accuracy_score(y, prediction),
            "Precision": precision_score(y, prediction, zero_division=0),
            "Recall": recall_score(y, prediction, zero_division=0),
            "F1-Score": f1_score(y, prediction, zero_division=0),
            "ROC-AUC": roc_auc_score(y, probability) if y.nunique() == 2 else float("nan"),
        }

    ## 사용자가 입력한 공정값 한 건을 저장된 AI 모델에 전달해서 양품인지 불량인지 예측
    def predict(self, values: dict[str, object]) -> tuple[int, float]:
        if self.model_manager is None:
            raise ValueError("먼저 저장 모델을 불러오세요.")
        input_df = pd.DataFrame([values], columns=self.feature_columns)
        probability = float(self.model_manager.predict_proba(input_df)[0])
        prediction = int(probability >= self.threshold)
        return prediction, probability
