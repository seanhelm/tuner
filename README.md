# tuner

Predict song genre from audio features using [XGBoost](https://github.com/dmlc/xgboost) gradient boosting classification.

Audio features and metadata were taken from [FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma). The audio
features were extracted using [LibROSA](https://librosa.github.io/librosa/).

To preprocess, use `python preprocess.py`. To perform cross-validation, use `python classify.py`.