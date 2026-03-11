import torch
import joblib
import numpy as np
import pandas as pd
import json
from flowpredictor import FlowPredictor
from seq2seq_predictor import LSTMSeq2SeqAttnModel  # ← 추가
from scipy.signal import savgol_filter

# ===== Seq2Seq+Attention 대상 배수지 =====
SEQ2SEQ_RESERVOIRS = {4, 7, 13}


#=========Resv Service========
class ReservoirInferenceService:
    def __init__(self, reservoir_configs, input_dim, window_size=180):
        self.window_size = window_size
        self.input_dim = input_dim

        self.models = {}
        self.scalers_x = {}
        self.scalers_y = {}

        for name, paths in reservoir_configs.items():
            try:
                # ===== 배수지 4, 7, 13: Seq2Seq+Attention =====
                if name in SEQ2SEQ_RESERVOIRS:
                    checkpoint = torch.load(
                        paths['weights'], map_location='cpu', weights_only=False
                    )
                    model = LSTMSeq2SeqAttnModel(
                        input_size=len(checkpoint['feature_cols']),
                        hidden_size=128,
                        num_layers=2,
                        output_size=checkpoint['output_time'],
                        embed_dim=16,
                        dropout=0.2,  # encoder inter-layer dropout (decoder dropout은 별도 없음)
                    )
                    state_dict = checkpoint['model_state_dict']
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.models[name] = model
                    # 체크포인트 내장 scaler 사용 (joblib 불필요)
                    self.scalers_x[name] = checkpoint['scalers']
                    self.scalers_y[name] = checkpoint['scalers']['value']
                    print(f"[resv {name}] Seq2Seq+Attention 모델 로드 완료")
                    continue

                # ===== 기존 배수지 (8, 10, 15): FlowPredictor =====
                with open(paths['config'], 'r') as f:
                    config = json.load(f)

                self.scalers_x[name] = joblib.load(paths['scaler_x'])
                self.scalers_y[name] = joblib.load(paths['scaler_y'])

                model = FlowPredictor(
                    input_dim=4,
                    hidden_dim=config['units'],
                    output_dim=config['forecast_size'],
                    dropout=config['dropout']
                )
                model.load_state_dict(torch.load(paths['weights'], map_location=torch.device('cpu')))
                model.eval()
                self.models[name] = model

            except FileNotFoundError as e:
                print(f"[resv {name}] 파일 없음, 스킵: {e}")
                continue

    def predict(self, reservoir_name, raw_data):
        model = self.models[reservoir_name]

        # ===== 배수지 4, 7, 13: Seq2Seq+Attention (자체 정규화) =====
        if reservoir_name in SEQ2SEQ_RESERVOIRS:
            scalers = self.scalers_x[reservoir_name]
            feature_cols = ['resv_flow', 'temperature', 'precipitate', 'humidity',
                           'time_sin', 'time_cos', 'dow_sin', 'dow_cos',
                           'season_sin', 'season_cos']

            features = raw_data[feature_cols].values.astype(np.float32).copy()

            # 정규화 (scale_cols만, sin/cos는 이미 [0,1])
            col_to_scaler = {
                'resv_flow': 'value',
                'temperature': 'temperature',
                'precipitate': 'rainfall',
                'humidity': 'humidity',
            }
            for col_name, scaler_key in col_to_scaler.items():
                idx = feature_cols.index(col_name)
                d_min = float(scalers[scaler_key]['min'])
                d_max = float(scalers[scaler_key]['max'])
                if d_max - d_min > 0:
                    features[:, idx] = (features[:, idx] - d_min) / (d_max - d_min)

            # 모델 입력은 72분(72 steps)만 사용하므로 window_size 초과분은 앞부분을 truncate
            input_tensor = torch.FloatTensor(features[-72:]).unsqueeze(0)

            with torch.no_grad():
                prediction = model(input_tensor).cpu().numpy().flatten()

            # 역정규화
            val_min = float(scalers['value']['min'])
            val_max = float(scalers['value']['max'])
            return (prediction * (val_max - val_min) + val_min).reshape(1, -1)

        # ===== 기존 배수지 (8, 10, 15): FlowPredictor =====
        scaler_x = self.scalers_x[reservoir_name]
        scaler_y = self.scalers_y[reservoir_name]

        n_min = self.window_size
        raw_data.loc[:, 'resv_flow'] = savgol_filter(raw_data['resv_flow'], window_length=31, polyorder=1)
        scaled_data = scaler_x.transform(raw_data)
        input_tensor = torch.FloatTensor(scaled_data).view(1, n_min, 4)

        with torch.no_grad():
            prediction = model(input_tensor)

        out = scaler_y.inverse_transform(prediction.cpu().numpy())
        return out
