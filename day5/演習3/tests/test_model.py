import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
    assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()

    inference_time = end_time - start_time

    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(predictions1, predictions2), (
        "モデルの予測結果に再現性がありません"
    )


def test_feature_importance(train_model):
    """特徴量重要度を検証"""
    model, X_test, _ = train_model

    # RandomForestClassifierから特徴量重要度を取得
    feature_importances = model.named_steps["classifier"].feature_importances_

    # 重要度が0より大きいことを確認（少なくとも一部の特徴量が予測に貢献している）
    assert np.any(feature_importances > 0), "すべての特徴量の重要度が0です"

    # 特徴量重要度の合計が約1になることを確認（RandomForestの仕様）
    assert np.isclose(np.sum(feature_importances), 1.0), (
        f"特徴量重要度の合計が1でありません: {np.sum(feature_importances)}"
    )

    # 上位の特徴量が一定の重要度を持つことを確認
    top_importance = np.max(feature_importances)
    assert top_importance >= 0.1, (
        f"最も重要な特徴量の重要度が低すぎます: {top_importance}"
    )


def test_model_serialization(train_model):
    """モデルのシリアライズとデシリアライズの整合性を検証"""
    model, X_test, _ = train_model

    # 予測結果を取得
    original_predictions = model.predict(X_test)
    original_proba = model.predict_proba(X_test)

    # モデルを一度シリアライズしてから再度読み込む
    with open("temp_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("temp_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    # 再読込したモデルで予測
    loaded_predictions = loaded_model.predict(X_test)
    loaded_proba = loaded_model.predict_proba(X_test)

    # 予測結果が同じであることを確認
    assert np.array_equal(original_predictions, loaded_predictions), (
        "シリアライズ前後で予測結果が異なります"
    )
    assert np.allclose(original_proba, loaded_proba), (
        "シリアライズ前後で予測確率が異なります"
    )

    # テスト用の一時ファイルを削除
    if os.path.exists("temp_model.pkl"):
        os.remove("temp_model.pkl")


def test_model_robustness(train_model):
    """異常値に対する予測安定性を検証"""
    model, X_test, _ = train_model

    # テスト用のデータフレームをコピー
    X_anomaly = X_test.copy()

    # 年齢に極端な値を設定
    if "Age" in X_anomaly.columns:
        X_anomaly.loc[0, "Age"] = 999  # 極端に高い年齢

    # 料金に極端な値を設定
    if "Fare" in X_anomaly.columns:
        X_anomaly.loc[1, "Fare"] = 99999  # 極端に高い料金

    # 負の値も試す
    if "Fare" in X_anomaly.columns:
        X_anomaly.loc[2, "Fare"] = -100  # 負の料金

    try:
        # 異常値を含むデータでも予測が実行できることを確認
        predictions = model.predict(X_anomaly)
        assert len(predictions) == len(X_anomaly), (
            "異常値を含むデータの予測数が不正です"
        )

        # 確率値が0-1の範囲内であることを確認
        probabilities = model.predict_proba(X_anomaly)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1), (
            "予測確率が0-1の範囲外です"
        )

        # 合計が1になることを確認
        assert np.allclose(np.sum(probabilities, axis=1), 1.0), (
            "予測確率の合計が1になりません"
        )

    except Exception as e:
        pytest.fail(f"異常値を含むデータの予測に失敗しました: {str(e)}")
