#!/usr/bin/env python
"""패치 비율 조정 기능 확인 스크립트

BYUMotorDataset에서 계산된 pos/neg 패치 수가 1:1에 가까운지 출력한다.
"""
from __future__ import annotations
import pandas as pd
from data.ds_byu import BYUMotorDataset, LABEL_CSV, NEG_PER_TOMO

if __name__ == "__main__":
    df = pd.read_csv(LABEL_CSV).drop_duplicates("tomo_id")
    ids = df["tomo_id"].tolist()

    ds = BYUMotorDataset(ids, mode="train", split="train")
    info = ds.df["Number of motors"].groupby("tomo_id").first()
    n_pos = int((info > 0).sum())
    n_neg = int((info == 0).sum())

    pos_total = n_pos * ds.pos_per_tomo
    neg_total = (n_pos + n_neg) * NEG_PER_TOMO
    ratio = pos_total / neg_total if neg_total else 0.0

    print(f"양성 패치 수: {pos_total}")
    print(f"음성 패치 수: {neg_total}")
    print(f"pos:neg 비율 ≈ {ratio:.2f}:1")
