"""SMOTE를 제거한 사전 학습 PKL 모델을 불러오는 클래스."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATHS = {
    "Random Forest": "random_forest_no_smote.pkl",
    "Gradient Boosting": "gradient_boosting_no_smote.pkl",
}


class PretrainedModelManager:
    """저장 모델 로드, 입력 열 변환, 확률 예측을 담당합니다."""

    def __init__(self, model_name: str) -> None:
        if model_name not in MODEL_PATHS:
            raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
        model_path = Path(__file__).resolve().parent / "models" / MODEL_PATHS[model_name]
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        self.model_name = model_name
        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.expected_features = list(self.model.feature_names_in_)

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """원본 공정 열을 저장 모델이 학습한 num__/cat__ 열 구조로 변환합니다."""
        # 이미 모델용 열로 변환된 CSV라면 열 순서만 맞춥니다.
        if set(self.expected_features).issubset(data.columns):
            prepared = data[self.expected_features].copy()
            return prepared.apply(pd.to_numeric, errors="raise")

        prepared = pd.DataFrame(index=data.index)
        missing = []
        for feature in self.expected_features:
            if feature.startswith("num__"):
                source = feature.removeprefix("num__")
                if source not in data.columns:
                    missing.append(source)
                else:
                    prepared[feature] = pd.to_numeric(data[source], errors="coerce")
            elif feature.startswith("cat__EQUIP_NAME_"):
                if "EQUIP_NAME" not in data.columns:
                    missing.append("EQUIP_NAME")
                else:
                    category = feature.removeprefix("cat__EQUIP_NAME_")
                    prepared[feature] = (data["EQUIP_NAME"].astype(str) == category).astype(float)
            else:
                missing.append(feature)

        if missing:
            missing_text = ", ".join(dict.fromkeys(missing))
            raise ValueError(f"저장 모델에 필요한 열이 CSV에 없습니다: {missing_text}")
        if prepared.isna().any().any():
            bad = prepared.columns[prepared.isna().any()].tolist()
            raise ValueError(f"숫자로 변환할 수 없거나 비어 있는 모델 입력값이 있습니다: {bad}")
        return prepared[self.expected_features]

    def predict_proba(self, data: pd.DataFrame):
        prepared = self.prepare_features(data)
        return self.model.predict_proba(prepared)[:, 1]

    def feature_importance(self) -> pd.Series:
        classifier = self.model.named_steps["classifier"]
        return pd.Series(
            classifier.feature_importances_,
            index=self.expected_features,
        ).sort_values(ascending=False)

    def model_parameters(self) -> dict:
        classifier = self.model.named_steps["classifier"]
        params = classifier.get_params()
        if self.model_name == "Random Forest":
            keys = ["n_estimators", "max_depth", "min_samples_leaf", "random_state"]
        else:
            keys = ["n_estimators", "learning_rate", "max_depth", "random_state"]
        return {key: params.get(key) for key in keys}
