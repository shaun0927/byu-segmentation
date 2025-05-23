import numpy as np
from postprocess.pp_byu import post_process_volume


def test_score_mode_linear_vs_exp():
    vol = np.zeros((50, 1, 1), dtype=float)
    vol[1, 0, 0] = 0.4  # near peak
    vol[40, 0, 0] = 0.9  # far peak

    df_lin = post_process_volume(
        vol,
        spacing=1.0,
        tomo_id="t",
        topk=2,
        gt_coord=(0.0, 0.0, 0.0),
        dist_weight=0.02,
        n_keep=1,
        score_mode="linear",
    )

    df_exp = post_process_volume(
        vol,
        spacing=1.0,
        tomo_id="t",
        topk=2,
        gt_coord=(0.0, 0.0, 0.0),
        dist_weight=0.02,
        n_keep=1,
        score_mode="exp",
    )

    coord_lin = df_lin.iloc[0, 1:4].tolist()
    coord_exp = df_exp.iloc[0, 1:4].tolist()

    assert coord_lin == [1.0, 0.0, 0.0]
    assert coord_exp == [40.0, 0.0, 0.0]
    print("\u2713 score_mode test passed")
