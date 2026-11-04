"""Vessel segmentation on synthetic frames with a known ellipse."""
import numpy as np

from usintent.metrics.vessel import centring_error_px, iou, segment_vessel
from usintent.sim.holdworld import HoldWorldConfig, generate


def test_segments_synthetic_vessel() -> None:
    cfg = HoldWorldConfig(seed=3, n_stations=(2, 2), p_lift=0.0)
    ep = generate(cfg, 1)[0]
    rows, cols = cfg.image_shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    (vr, vc), (ar, ac) = cfg.vessel_center_rc, cfg.vessel_radii_rc
    gt = ((rr - vr * rows) / (ar * rows)) ** 2 + ((cc - vc * cols) / (ac * cols)) ** 2 <= 1
    contact = np.flatnonzero(ep.extras["gt.in_contact"] == 1)
    ious = [iou(segment_vessel(ep.bmode[k]).mask, gt) for k in contact[::20]]
    assert np.median(ious) > 0.75
    res = segment_vessel(ep.bmode[contact[10]])
    assert res.in_view and centring_error_px(res, cols) < 2.0


def test_no_vessel_in_uncoupled_frame() -> None:
    frame = np.full((32, 32), 5, np.uint8)
    res = segment_vessel(frame)
    assert not res.in_view and np.isnan(centring_error_px(res, 32))
