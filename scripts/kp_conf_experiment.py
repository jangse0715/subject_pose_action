"""
2일차 과업: keypoint_confidence 기준(threshold)을 찾기 위한 실험

흐름 ((6) '해석·결정'은 사람이 결과 파일을 보고 진행)
  (1) 추론·기록      모든 Frame → YOLO Pose → 관절별 좌표·Confidence 원본 기록 (기준 적용 X)
  (2) 대표 Frame 표시 앞뒤 3% 제외 → 남은 구간 5등분 → 구간 번호, 구간 중앙 Frame = 대표 Frame
  (3) 분포 묘사      전체 / 관절별 / 조건별 / 구간별 / 동작별 Confidence 분포 + 히스토그램
  (4) 후보별 재계산  후보 기준마다 VALID·WARNING·LOW 비율 (추론 없이 (1)의 표로 반복)
  (5) 화면 대조      대표 Frame에 Person BBox + 그림용 기준별 Skeleton → Clip별 파노라마
  [보류] Frame별 요약표(frame_quality.csv) — 기준 결정 후 main()의 주석 해제

입력  : data/videos/S/<사람 폴더 S01~S03>/ 아래 mp4 전부
출력  : reports/conf_experiment/
실행  : python scripts/kp_conf_experiment.py   (프로젝트 최상위 폴더에서)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config, resolve_device
from src.pose_utils import COCO_KEYPOINT_NAMES, COCO_SKELETON, confidence_state

# Skeleton 색: 교재 10번 result.plot()과 같은 Ultralytics 색 (BGR 값을 그대로 cv2에 사용)
#   출처: .venv/Lib/site-packages/ultralytics/utils/plotting.py (ultralytics 8.4.203)
#         Colors.pose_palette, Annotator.limb_color·kpt_color
#   COCO_SKELETON(pose_utils)의 연결 순서가 Ultralytics skeleton 순서와 같아 순서대로 대응
#   Ultralytics는 왼쪽·오른쪽을 같은 색으로 그림
FACE, ARM, TORSO, LEG = (0, 255, 0), (255, 128, 0), (255, 51, 255), (51, 153, 255)
LIMB_COLORS = [LEG] * 4 + [TORSO] * 3 + [ARM] * 5 + [FACE] * 7  # 선 19개
KPT_COLORS = [FACE] * 5 + [ARM] * 6 + [LEG] * 6  # 관절 17개
MAIN_BOX, OTHER_BOX, BELOW = (
    (255, 0, 0),
    (0, 255, 255),
    (0, 0, 255),
)  # 파랑·노랑·빨강


# ==================================================
# 후보 기준 (파일 안에서 직접 고쳐 쓰는 함수)
# ==================================================
# [역할] (4)에서 하나씩 시험해 볼 '관절 신뢰도 기준값' 후보를 만든다 (기본 0.30, 0.35, … 0.90 → 13개)
# [입력] start·stop·step: 시작값·끝값·간격. 범위를 바꾸려면 여기 기본값만 고치면 된다
# [출력] 후보 기준값 목록(list)
# [쓰는 곳] main() → evaluate_thresholds()
def candidate_thresholds(
    start: float = 0.30, stop: float = 0.90, step: float = 0.05
) -> list[float]:
    """(4)에서 비교할 keypoint_confidence 후보 목록"""
    count = int(round((stop - start) / step)) + 1
    return [round(start + step * i, 2) for i in range(count)]


# [역할] (5)에서 Skeleton 그림을 그려 볼 기준값만 따로 고른다 (그림이 너무 많아지지 않게 3개만)
# [출력] 그림용 기준값 목록(list)
# [쓰는 곳] save_representatives()
def draw_thresholds() -> list[float]:
    """(5)에서 Skeleton을 다시 그릴 기준 목록 (그림 수를 줄이려고 짧게 둠)"""
    return [0.30, 0.50, 0.70]


# ==================================================
# 실행 옵션
# ==================================================
# [역할] 터미널에서 실행할 때 덧붙일 수 있는 옵션(입력 폴더, 앞뒤 제외 비율, 구간 수 등)을 정한다
# [출력] 옵션 값 묶음(args). 아무 옵션도 안 붙이면 기본값을 쓴다
# [쓰는 곳] main()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", default=str(ROOT / "data" / "videos"))
    parser.add_argument(
        "--pattern", default="S/*/*.mp4", help="video-root 기준 mp4 검색 패턴"
    )
    parser.add_argument(
        "--trim", type=float, default=0.03, help="앞뒤에서 각각 제외할 비율"
    )
    parser.add_argument("--segments", type=int, default=5, help="남은 구간을 나눌 개수")
    parser.add_argument(
        "--warning", type=float, default=0.30, help="confidence_state의 WARNING 하한"
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "reports" / "conf_experiment")
    )
    return parser.parse_args()


# ==================================================
# 입력: 파일명에서 촬영 조건 읽기
# ==================================================
# 파일 이름 규칙(예: S02_A003_near_front, 방향은 없어도 됨)을 읽어내는 틀(정규표현식)
CLIP_PATTERN = re.compile(
    r"^(S\d+)_(A\d+)_(near|far)(?:_(front|left|right))?$", re.IGNORECASE
)


# [역할] 영상 파일 이름에서 누가(S02)·무슨 동작(A003)·거리(near/far)·방향(front/left/right)인지 뽑아낸다
# [입력] 확장자를 뺀 파일 이름
# [출력] {subject, action_id, distance, direction} — 규칙에 안 맞으면 전부 'unknown'
# [쓰는 곳] main() → (3)·(4)에서 촬영 조건별로 묶을 때 쓰는 컬럼이 된다
def parse_clip_name(stem: str) -> dict:
    """S02_A003_near_front → 사람·동작·거리·방향 (방향이 없는 이름은 'none')"""
    match = CLIP_PATTERN.match(stem)
    if match is None:
        return {
            "subject": "unknown",
            "action_id": "unknown",
            "distance": "unknown",
            "direction": "unknown",
        }
    subject, action_id, distance, direction = match.groups()
    return {
        "subject": subject.upper(),
        "action_id": action_id.upper(),
        "distance": distance.lower(),
        "direction": (direction or "none").lower(),
    }


# ==================================================
# (1) 추론·기록 (교재 2일차 10번 기반)
# ==================================================
# [역할] (1) 영상을 처음부터 끝까지 한 장씩 읽어 YOLO Pose로 관절을 찾고,
#        관절마다 좌표와 신뢰도를 그대로 적어 둔다 (여기서는 기준값으로 거르지 않음)
# [입력] 영상 경로, 불러온 모델, project.json 설정, 장치(GPU/CPU)
# [출력] 'Frame 1장 × 관절 17개'를 한 줄씩 펼친 표(DataFrame). 영상을 못 열면 None
# [기록] 사람 수, 주 인물 박스 신뢰도, 가장 낮은 박스 신뢰도, 주 인물 박스 좌표, 관절 좌표·신뢰도
# [쓰는 곳] main()
def extract_clip(video_path: Path, model, config: dict, device) -> pd.DataFrame | None:
    """모든 Frame 추론 → 'Frame × 관절' 한 행씩 기록. 기준은 적용하지 않음"""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    rows = []
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        result = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )[0]

        people = len(result.boxes) if result.boxes is not None else 0
        base = {
            "frame_index": frame_index,
            "time_sec": round(frame_index / fps, 3),
            "people": people,
        }

        if (
            people > 0
            and result.keypoints is not None
            and result.keypoints.conf is not None
        ):
            box_conf = result.boxes.conf.cpu().numpy()
            main = int(
                np.argmax(box_conf)
            )  # 박스 Confidence가 가장 높은 사람 = 주 인물
            base["main_box_conf"] = round(float(box_conf[main]), 4)
            base["lowest_box_conf"] = round(
                float(np.min(box_conf)), 4
            )  # 가장 낮은 person_confidence 확인
            x1, y1, x2, y2 = result.boxes.xyxy.cpu().numpy()[
                main
            ]  # 주 인물 Person BBox 좌표
            base.update(
                box_x1=round(float(x1), 1),
                box_y1=round(float(y1), 1),
                box_x2=round(float(x2), 1),
                box_y2=round(float(y2), 1),
            )
            xy = result.keypoints.xy.cpu().numpy()[main]
            conf = result.keypoints.conf.cpu().numpy()[main]
        else:  # 사람을 못 찾은 Frame: 값은 비워 두고 people=0으로 표시
            xy = np.full((len(COCO_KEYPOINT_NAMES), 2), np.nan)
            conf = np.full(len(COCO_KEYPOINT_NAMES), np.nan)

        for keypoint_id, name in enumerate(COCO_KEYPOINT_NAMES):
            rows.append(
                {
                    **base,
                    "keypoint_id": keypoint_id,
                    "keypoint_name": name,
                    "pixel_x": round(float(xy[keypoint_id][0]), 2),
                    "pixel_y": round(float(xy[keypoint_id][1]), 2),
                    "confidence": round(float(conf[keypoint_id]), 4),
                }
            )
        frame_index += 1

    capture.release()
    return pd.DataFrame(rows) if rows else None


# ==================================================
# (2) 대표 Frame 표시
# ==================================================
# [역할] (2) 영상 앞뒤를 조금(기본 3%) 잘라내고, 남은 부분을 똑같이 5구간으로 나눠
#        구간마다 가운데 Frame을 '대표 Frame'으로 고른다
# [입력] 전체 Frame 수, 앞뒤 제외 비율, 구간 수
# [출력] Frame마다 붙일 구간 번호(잘려 나간 Frame은 0), {구간 번호: 대표 Frame 번호}
# [쓰는 곳] main() → 구간 번호는 (3)·(4)에서 범위를 고를 때, 대표 Frame은 (5) 그림에 쓰인다
def segment_frames(
    total_frames: int, trim: float, n_segments: int
) -> tuple[np.ndarray, dict[int, int]]:
    """
    앞뒤 trim 비율을 뺀 구간을 n_segments 등분
    반환: Frame별 구간 번호(0 = 제외 구간), {구간 번호: 대표 Frame 번호(구간 중앙)}
    """
    segments = np.zeros(total_frames, dtype=int)
    cut = int(round(total_frames * trim))
    kept = np.arange(cut, total_frames - cut)
    if len(kept) < n_segments:
        return segments, {}

    representatives = {}
    for number, part in enumerate(np.array_split(kept, n_segments), start=1):
        segments[part] = number
        representatives[number] = int(part[len(part) // 2])
    return segments, representatives


# ==================================================
# (3) 분포 묘사 (교재 2일차 18·24절의 읽는 순서)
# ==================================================
# (3)에서 분포를 어떤 기준으로 묶어 볼지(전체/관절별/조건별/구간별/동작별)와
# 요약에 넣을 백분위(5·25·50·75·95%)
GROUPINGS = {
    "overall": [],
    "by_keypoint": ["keypoint_name"],
    "by_condition": ["distance", "direction"],
    "by_segment": ["segment"],
    "by_action": ["action_id"],
}
PERCENTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


# [역할] (3) 관절 신뢰도 값들이 어디에 몰려 있는지 숫자로 요약한다(평균, 최솟값, 백분위 등)
#        + 0.05 간격 히스토그램(구간별 개수)으로 '값이 갈리는 골짜기'를 찾을 수 있게 한다
# [입력] 사람이 잡히고 앞뒤 제외 구간 밖인 관절 값만 남긴 표, 저장 폴더
# [출력] distribution_*.csv 6개 (overall, by_keypoint, by_condition, by_segment, by_action, histogram)
# [쓰는 곳] main()
def describe_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """묶음별 요약 통계 + 0.05 간격 히스토그램(골짜기 찾기용)"""
    for name, keys in GROUPINGS.items():
        grouped = df.assign(all="all").groupby(keys or ["all"])["confidence"]
        grouped.describe(percentiles=PERCENTILES).round(4).to_csv(
            out_dir / f"distribution_{name}.csv", encoding="utf-8-sig"
        )

    bins = np.linspace(0.0, 1.0, 21)
    binned = pd.cut(df["confidence"], bins=bins, include_lowest=True)
    histogram = pd.crosstab(binned, df["keypoint_name"]).reindex(
        index=binned.cat.categories, columns=COCO_KEYPOINT_NAMES, fill_value=0
    )  # 빈 구간도 0으로 남김
    histogram.insert(0, "all_keypoints", histogram.sum(axis=1))
    histogram.to_csv(out_dir / "distribution_histogram.csv", encoding="utf-8-sig")


# ==================================================
# (4) 후보별 재계산 (교재 09·10번 계산식 + confidence_state)
# ==================================================
# [역할] (4) 후보 기준값을 하나씩 적용해 보며 '이 기준이면 관절이 얼마나 걸러지나'를 계산한다
#        (모델을 다시 돌리지 않고 (1)의 표만 다시 읽음)
# [입력] (3)과 같은 표, 후보 기준 목록, WARNING 하한, 저장 폴더
# [출력] threshold_comparison.csv(기준별 전체 비율·Frame당 걸러지는 관절 수),
#        threshold_by_keypoint.csv(관절별), threshold_by_condition.csv(촬영 조건별)
# [쓰는 곳] main()
def evaluate_thresholds(
    df: pd.DataFrame, thresholds: list[float], warning: float, out_dir: Path
) -> None:
    """후보 기준마다 상태 비율과 'Frame별 기준 미만 관절 수'를 다시 계산"""
    summary, by_keypoint, by_condition = [], [], []
    frame_keys = [df["clip"], df["frame_index"]]

    for threshold in thresholds:
        states = df["confidence"].map(
            lambda c: confidence_state(
                c, valid_threshold=threshold, warning_threshold=warning
            )
        )
        below = states.ne("VALID")  # 기준 미만 = 교재의 low_keypoint_count 대상
        below_per_frame = below.groupby(frame_keys).sum()

        summary.append(
            {
                "threshold": threshold,
                "valid_ratio": states.eq("VALID").mean(),
                "warning_ratio": states.eq("WARNING").mean(),
                "low_ratio": states.eq("LOW").mean(),
                "below_threshold_ratio": below.mean(),
                "mean_below_per_frame": below_per_frame.mean(),
                "frames_with_below_ratio": (below_per_frame > 0).mean(),
            }
        )
        by_keypoint.append(below.groupby(df["keypoint_name"]).mean().rename(threshold))
        by_condition.append(
            below.groupby([df["distance"], df["direction"]]).mean().rename(threshold)
        )

    pd.DataFrame(summary).round(4).to_csv(
        out_dir / "threshold_comparison.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(by_keypoint, axis=1).reindex(COCO_KEYPOINT_NAMES).round(4).to_csv(
        out_dir / "threshold_by_keypoint.csv", encoding="utf-8-sig"
    )
    pd.concat(by_condition, axis=1).round(4).to_csv(
        out_dir / "threshold_by_condition.csv", encoding="utf-8-sig"
    )


# ==================================================
# (5) 화면 대조 (교재 2일차 08번 + 4일차 21번 read_frame_at 방식)
# ==================================================
# [역할] (5) 대표 Frame 한 장만 모델에 다시 넣어, 그 장면에서 잡힌 사람 박스를 전부 가져온다
#        ((1)은 주 인물 박스 하나만 저장했기 때문)
# [입력] 이미지 한 장, 모델, 설정, 장치
# [출력] 박스 좌표 목록, 박스 신뢰도 목록 (사람이 없으면 빈 목록)
# [쓰는 곳] save_representatives()
def detect_boxes(
    image: np.ndarray, model, config: dict, device
) -> tuple[np.ndarray, np.ndarray]:
    """대표 Frame만 다시 추론해 검출된 Person BBox 전부와 박스 Confidence 반환 ((1)과 같은 설정)"""
    result = model.predict(
        source=image,
        imgsz=config["image_size"],
        conf=config["person_confidence"],
        device=device,
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 4)), np.empty(0)
    return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()


# [역할] (5) 이미지 위에 사람 박스와 Skeleton을 그린다. 기준값 이상인 관절만 부위별 색
#        (얼굴 초록·팔 하늘·몸통 자홍·다리 주황, 교재 10번과 같은 Ultralytics 색)으로 잇고,
#        기준 미만 관절은 빨간 X로 표시해 '이 기준이면 무엇이 빠지는지' 눈으로 보게 한다
# [입력] 이미지, 관절 좌표·신뢰도, 기준값, 박스 좌표·신뢰도
# [출력] 그림이 그려진 이미지 (원본 이미지는 건드리지 않음)
# [쓰는 곳] save_representatives()
def draw_skeleton(
    image: np.ndarray,
    xy: np.ndarray,
    conf: np.ndarray,
    threshold: float,
    boxes: np.ndarray,
    box_conf: np.ndarray,
) -> np.ndarray:
    """Person BBox(주 인물=파랑, 그 외=노랑) → 기준 이상 관절만 부위별 색으로 연결, 기준 미만 관절은 빨간 X"""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    lw = max(round((height + width) / 2 * 0.003), 2)  # Ultralytics와 같은 굵기 계산
    main = int(np.argmax(box_conf)) if len(box_conf) else -1
    for index, ((x1, y1, x2, y2), score) in enumerate(zip(boxes.astype(int), box_conf)):
        color = MAIN_BOX if index == main else OTHER_BOX
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, lw)
        cv2.putText(
            canvas,
            f"person {score:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    keep = np.nan_to_num(conf, nan=-1.0) >= threshold

    for (start, end), color in zip(COCO_SKELETON, LIMB_COLORS):
        if keep[start] and keep[end]:
            cv2.line(
                canvas,
                tuple(map(int, xy[start])),
                tuple(map(int, xy[end])),
                color,
                int(np.ceil(lw / 2)),
                lineType=cv2.LINE_AA,
            )
    for (x, y), ok, color in zip(xy, keep, KPT_COLORS):
        if np.isnan(x):
            continue
        if ok:
            cv2.circle(canvas, (int(x), int(y)), lw, color, -1, lineType=cv2.LINE_AA)
        else:
            cv2.drawMarker(
                canvas, (int(x), int(y)), BELOW, cv2.MARKER_TILTED_CROSS, lw * 4, lw
            )

    cv2.putText(
        canvas,
        f"kp_conf >= {threshold:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    return canvas


# ==================================================
# [보류] Frame별 요약표 — 기준 결정 후 main()에서 주석 해제
# ==================================================
# [역할] [보류] 관절 17줄로 펼쳐진 Frame을 한 줄로 접어 'Frame 품질 요약표'를 만든다
#        (교재 2일차 24절처럼 품질 낮은 Frame을 정렬해 찾기 위한 표)
# [입력] (1)의 원본 표 전체, 결정된 기준값(LOW 관절 수를 셀 때만 사용)
# [출력] Frame 1장 = 1줄 표 (평균 신뢰도, 가장 낮은 관절과 그 값, LOW 관절 수 등)
# [쓰는 곳] 기준이 정해진 뒤 main() 끝의 주석을 풀면 실행된다
def summarize_frames(raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    [보류] Frame 한 장 = 한 줄 요약표 (교재 2일차 10번 _frame_quality.csv 형식)
    threshold: 결정된 keypoint_confidence. LOW 관절 수를 셀 때만 사용
    사람 미검출 Frame은 교재와 같이 mean 0.0 · lowest NO_PERSON · LOW 17로 기록
    """
    keys = ["clip", "frame_index"]
    frames = raw.drop_duplicates(keys).set_index(keys)[
        ["time_sec", "segment", "people", "main_box_conf"]
    ]
    detected = raw.dropna(subset=["confidence"])
    grouped = detected.groupby(keys)["confidence"]
    lowest = detected.loc[grouped.idxmin()].set_index(keys)

    frames["mean_keypoint_conf"] = grouped.mean().round(4)
    frames["lowest_keypoint"] = lowest["keypoint_name"]
    frames["lowest_keypoint_conf"] = lowest["confidence"]
    frames["low_keypoint_count"] = (
        detected["confidence"]
        .lt(threshold)
        .groupby([detected["clip"], detected["frame_index"]])
        .sum()
    )

    no_person = frames["mean_keypoint_conf"].isna()
    frames.loc[no_person, ["mean_keypoint_conf", "lowest_keypoint_conf"]] = 0.0
    frames.loc[no_person, "lowest_keypoint"] = "NO_PERSON"
    frames.loc[no_person, "low_keypoint_count"] = len(COCO_KEYPOINT_NAMES)
    frames["low_keypoint_count"] = frames["low_keypoint_count"].astype(int)
    frames["keypoint_threshold"] = threshold
    return frames.reset_index()


# 파노라마 아래 범례 띠에 표시할 색 (이름은 영어: cv2 글자는 한글을 못 씀)
LEGEND = [
    ("face", FACE),
    ("arm", ARM),
    ("torso", TORSO),
    ("leg", LEG),
    ("below threshold (X)", BELOW),
    ("main person", MAIN_BOX),
    ("other person", OTHER_BOX),
]


# [역할] 대표 Frame 이미지 여러 장을 원본 크기 그대로 옆으로 이어 붙여 한 장으로 만들고,
#        아래에 색 범례 띠를 붙인다. 각 장 아래에는 구간 번호와 Frame 번호를 적는다
# [입력] 이미지 목록, 각 이미지 아래 적을 글자 목록,
#        height(선택: 크기가 서로 다른 이미지를 이어 붙일 때만 높이를 맞춤, 기본은 원본 그대로)
# [출력] 가로로 긴 이미지 한 장 (가로 스크롤로 보는 용도)
# [쓰는 곳] save_representatives()
def make_panorama(
    tiles: list[np.ndarray], labels: list[str], height: int | None = None
) -> np.ndarray:
    """대표 Frame 이미지들을 가로로 이어 붙이고 아래에 색 범례 띠를 붙임 (기본: 원본 크기 그대로)"""
    prepared = []
    for tile, label in zip(tiles, labels):
        if (
            height is not None
        ):  # 크기가 서로 다른 이미지를 이어 붙여야 할 때만 높이를 맞춤
            tile = cv2.resize(
                tile, (int(tile.shape[1] * height / tile.shape[0]), height)
            )
        tile = tile.copy()  # 원본 이미지에 글자가 남지 않도록 복사본에 적음
        cv2.putText(
            tile,
            label,
            (10, tile.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        prepared.append(tile)
    panorama = cv2.hconcat(prepared)

    legend = np.zeros((50, panorama.shape[1], 3), dtype=np.uint8)
    x = 15
    for name, color in LEGEND:
        cv2.rectangle(legend, (x, 15), (x + 30, 35), color, -1)
        cv2.putText(
            legend,
            name,
            (x + 40, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        x += 40 + cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] + 30
    return cv2.vconcat([panorama, legend])


# [역할] (5) 한 영상의 대표 Frame 5장을 꺼내 와서, 원본 파노라마 1장과
#        그림용 기준마다 Skeleton 파노라마 1장씩을 저장한다
# [입력] 영상 경로, (1)·(2)를 마친 그 영상의 표, 대표 Frame 번호, 저장 폴더, 모델·설정·장치(박스 다시 찾기용)
# [출력] panorama/{입력 폴더명}/{영상이름}_frames.jpg, …_conf30·50·70.jpg (반환값 없음)
# [쓰는 곳] main()
def save_representatives(
    video_path: Path,
    clip_df: pd.DataFrame,
    representatives: dict[int, int],
    out_dir: Path,
    model,
    config: dict,
    device,
) -> None:
    """대표 Frame 원본·Skeleton을 Clip별 파노라마로만 저장 (장별 이미지는 저장하지 않음)"""
    capture = cv2.VideoCapture(str(video_path))
    labels, originals = [], []
    skeletons = {threshold: [] for threshold in draw_thresholds()}

    for segment, frame_index in representatives.items():
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, image = capture.read()
        if not success:
            continue

        points = clip_df[clip_df["frame_index"] == frame_index].sort_values(
            "keypoint_id"
        )
        xy = points[["pixel_x", "pixel_y"]].to_numpy(dtype=float)
        conf = points["confidence"].to_numpy(dtype=float)
        boxes, box_conf = detect_boxes(image, model, config, device)

        labels.append(f"seg{segment} f{frame_index}")
        originals.append(image)
        for threshold in draw_thresholds():
            skeletons[threshold].append(
                draw_skeleton(image, xy, conf, threshold, boxes, box_conf)
            )
    capture.release()

    if originals:  # Clip별 파노라마: 원본 1장 + 그림용 기준마다 1장
        # 입력 영상이 들어 있던 폴더 이름(S01~S03)으로 하위 폴더를 만들어 저장
        save_dir = out_dir / "panorama" / video_path.parent.name
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(save_dir / f"{video_path.stem}_frames.jpg"),
            make_panorama(originals, labels),
        )
        for threshold, drawn in skeletons.items():
            cv2.imwrite(
                str(
                    save_dir
                    / f"{video_path.stem}_conf{int(round(threshold * 100)):02d}.jpg"
                ),
                make_panorama(drawn, labels),
            )


# ==================================================
# 전체 실행
# ==================================================
# [역할] 전체 순서를 지휘한다: 영상 찾기 → 영상마다 (1)·(2)·(5) → 모두 합쳐 원본 표 저장 → (3) → (4) → 확인할 점 출력
# [출력] reports/conf_experiment/ 아래 결과 파일들, 터미널 요약 메시지
def main() -> None:
    args = parse_args()
    videos = sorted(Path(args.video_root).glob(args.pattern))
    if not videos:
        raise FileNotFoundError(
            f"영상이 없습니다: {Path(args.video_root) / args.pattern}"
        )

    out_dir = Path(args.output_dir)

    config = load_config()
    device = resolve_device(config["device"])
    model = YOLO(config["pose_model"])

    clips, failed = [], []
    for video in videos:
        print(f"[1·2·5단계] {video.name}")
        clip_df = extract_clip(video, model, config, device)
        if clip_df is None:
            failed.append(video.name)
            continue

        segments, representatives = segment_frames(
            int(clip_df["frame_index"].max()) + 1, args.trim, args.segments
        )
        clip_df["segment"] = segments[clip_df["frame_index"].to_numpy()]
        clip_df["is_representative"] = clip_df["frame_index"].isin(
            list(representatives.values())
        )
        clip_df = clip_df.assign(clip=video.stem, **parse_clip_name(video.stem))

        save_representatives(
            video, clip_df, representatives, out_dir, model, config, device
        )
        clips.append(clip_df)

    if not clips:
        raise RuntimeError(f"처리된 영상이 없습니다. 실패: {failed}")

    raw = pd.concat(clips, ignore_index=True)
    raw.to_csv(out_dir / "keypoint_conf_raw.csv", index=False, encoding="utf-8-sig")

    # (3)·(4)는 사람을 찾은 Frame + 앞뒤 제외 구간 밖(segment > 0)만 사용
    valid = raw[(raw["people"] > 0) & (raw["segment"] > 0)].dropna(
        subset=["confidence"]
    )
    print(f"[3단계] 분포 묘사: {len(valid)}개 관절 값")
    describe_distribution(valid, out_dir)
    print(f"[4단계] 후보 기준 {len(candidate_thresholds())}개 비교")
    evaluate_thresholds(valid, candidate_thresholds(), args.warning, out_dir)

    # [보류] Frame별 요약표 — keypoint_confidence 기준을 정한 뒤에 실행할 것
    #   1) KP_THRESHOLD에 결정한 값을 넣고  2) 아래 두 줄의 주석을 풀고  3) 다시 실행
    #   결과 frame_quality.csv: 교재 2일차 24절처럼 low_keypoint_count·mean_keypoint_conf로
    #   정렬해 품질 낮은 Frame을 찾고 time_sec로 영상 위치 확인
    #   (추론을 다시 돌리지 않으려면 keypoint_conf_raw.csv를 읽어 summarize_frames에 넣어도 됨)
    # KP_THRESHOLD = 0.50  # ← 결정한 기준으로 교체
    # summarize_frames(raw, KP_THRESHOLD).to_csv(out_dir / "frame_quality.csv", index=False, encoding="utf-8-sig")

    no_person = raw.loc[raw["people"] == 0, ["clip", "frame_index"]].drop_duplicates()
    multi_person = raw.loc[
        raw["people"] >= 2, ["clip", "frame_index"]
    ].drop_duplicates()
    print(f"완료: {len(clips)}개 영상 → {out_dir}")
    print(
        f"사람 미검출 Frame {len(no_person)}개 · 2명 이상 Frame {len(multi_person)}개 (5단계 파노라마에서 확인 권장)"
    )
    too_short = sorted(set(raw["clip"]) - set(valid["clip"]))
    if too_short:
        print(f"3·4단계에 들어가지 못한 영상(너무 짧거나 사람 미검출): {too_short}")
    if failed:
        print(f"열지 못한 영상: {failed}")


if __name__ == "__main__":
    main()
