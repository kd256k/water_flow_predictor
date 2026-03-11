# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

import asyncio
import os
import socket
import json
import sys
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from generator import run_generator

app = FastAPI()

RESULT_EXPIRE_SECONDS = 3600

configs={
    4:'a',
    7:'d',
    8:'e',
    10:'g',
    13:'j',
    15:'l',
}

origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Check Redis Connection
def is_redis_available(host: str, port: int, timeout: int = 1) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

#Initialize Redis Client
redis_client = None
_redis_host = os.environ.get("REDIS_HOST", "127.0.0.1")
_redis_port = 6379

if is_redis_available(_redis_host, _redis_port):
    redis_client = Redis(host=_redis_host, port=_redis_port, decode_responses=True)
else:
    print(f"[WARNING] Redis 서버에 연결할 수 없습니다: {_redis_host}:{_redis_port}")
    sys.exit(1)


#Saving Logic
async def _save_result(
    task_id: str,
    status: str,
    prediction_data: str | None = None,
    predict_time: str | None = None,
    accuracy: str | None = None,
    error: str | None = None,
):
    if redis_client is None:
        print(f"[{task_id}] Redis 미연결로 결과 저장 불가 (status={status})")
        return
    mapping = {"status": status}
    if prediction_data is not None:
        mapping["prediction_data"] = prediction_data
    if predict_time is not None:
        mapping["predict_from"] = predict_time
    if accuracy is not None:
        mapping["accuracy"] = accuracy
    if error is not None:
        mapping["error"] = error
    key = f"result:{task_id}"
    await redis_client.hset(key, mapping=mapping)
    await redis_client.expire(key, RESULT_EXPIRE_SECONDS)


#Call Prediction Logic
async def resv_pred(task_id: str, resv_id: int, date: str | None = None):
    print(f"[{task_id}] 예측 시작... ")
    print(f"[배수지: {resv_id}] [date: {date}]")

    if redis_client is None:
        print(f"[{task_id}] Redis 미연결로 작업 중단")
        await _save_result(task_id, "error", error="Redis 서버 연결 불가")
        return

    if resv_id not in configs.keys():
        print(f"[{task_id}] 알 수 없는 배수지 ID: {resv_id}")
        await _save_result(task_id, "failed", error=f"알 수 없는 배수지 ID: {resv_id}")
        return

    try:
        data, pred_time, approx_mape = await asyncio.to_thread(run_generator, task_id, resv_id, date)
        await _save_result(
            task_id, "completed", prediction_data=data, predict_time=pred_time, accuracy=approx_mape
        )
        print(f"[{task_id}] Hash 데이터 저장 완료!")
    except Exception as e:
        print(f"[{task_id}] 오류: {e}")
        await _save_result(task_id, "error", error=str(e))

#Predict API
@app.post("/predict/{resv_id}/{task_id}")
async def start_predict(resv_id: int, task_id: str, date: str, background_tasks: BackgroundTasks):
    if resv_id not in configs:
        raise HTTPException(status_code=400, detail=f"알 수 없는 배수지 ID: {resv_id}")
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis 서버 연결 불가")
    background_tasks.add_task(resv_pred, task_id, resv_id, date)
    return {"status": "started", "task_id": task_id, "resv_id": resv_id, "date": date}

#Result View API
@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """태스크 결과 조회 (Redis에서 조회)."""
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis 서버 연결 불가")
    key = f"result:{task_id}"
    raw = await redis_client.hgetall(key)
    if not raw:
        raise HTTPException(
            status_code=404, detail=f"task_id '{task_id}' 결과 없음 또는 만료됨"
        )
    status = raw.get("status", "unknown")
    out = {"task_id": task_id, "status": status}
    if "prediction_data" in raw:
        out["prediction_data"] = json.loads(raw["prediction_data"])
    if "predict_from" in raw:
        out["predict_from"] = raw["predict_from"]
    if "accuracy" in raw:
        out["accuracy"] = raw["accuracy"]
    if "error" in raw:
        out["error"] = raw["error"]
    return out
