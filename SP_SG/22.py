import itertools

import re
from collections import defaultdict

import time
from datetime import datetime
import torch, cv2, numpy as np
import glob
from typing import Dict, Tuple
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import sys, os
sys.path.append(os.path.expanduser('~/SHJ/sp_sg'))  # 경로 추가

from models.superpoint import SuperPoint
from models.superglue import SuperGlue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sp = SuperPoint({'nms_radius':3, 'keypoint_threshold':0.007, 'max_keypoints':50000}).to(device).eval()
sg = SuperGlue({'weights':'outdoor','match_threshold':0.25,'sinkhorn_iterations':20}).to(device).eval()


class FeatureExtractor:
    """
    Class for extracting features from images using pycolmap, opencv2 SIFT.
    This class provides methods to extract features, descriptor, and store them in a COLMAP database.
    """
    
    def __init__(self, image_path, database_path="database.db"):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.jpg_name = os.path.splitext(os.path.basename(self.image_path))[0]
        self.database_path = database_path
        if self.image is None:
            raise ValueError(f"Could not read the image at {image_path}")   
        print(f"Image loaded: {image_path}, size: {self.image.shape[0]}x{self.image.shape[1]}")

    def img_filtering(self,out_dir):
        """
        1) 노란색 hsv 마스킹
        2) grayscale
        3) CLAHE 
        """
        img = self.image
        h, w = img.shape[:2] 

        print(f"origin completed {self.jpg_name} Image size: {h}x{w}") 

        # 1) 저장 디렉토리 준비
        os.makedirs(out_dir, exist_ok=True)

        # 2) 천장 부분 검정색으로 마스킹
        cut = int(0.15 * h)
        img[:cut, :] = 0  

        # 3) 노란색 hsv 그래디언트 강하게 설정
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([15,  60,  60])   # H,S,V
        upper = np.array([40, 255, 255])
        mask  = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8), 1)

        # LAB 기반 색대비 보존 그레이
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        Lc = clahe.apply(L)
        bc = clahe.apply(b)
        bnorm = cv2.normalize(bc, None, 0, 255, cv2.NORM_MINMAX)

        gray0 = Lc
        gain  = cv2.normalize(mask, None, 0, 80, cv2.NORM_MINMAX)  # 마스크가 1일수록 +밝기
        gray1 = cv2.add(gray0, gain.astype(np.uint8))
        gray1 = cv2.addWeighted(gray1, 0.7, bnorm, 0.3, 0)        # L과 b* 혼합

        gray1 = clahe.apply(gray1)

        cv2.imwrite(f"{out_dir}/filter_{self.jpg_name}.jpg", gray1)

        return  gray1 


def collect_images(image_root: str,
                   pattern: str = "**/*.jpg",
                   exclude_dirs: tuple[str,...] = ("filtered", ".cache", ".git")) -> list[str]:
    # 1) 수집 (패턴 단순화: "**/*.jpg"면 충분)
    raw = glob.glob(os.path.join(image_root, pattern), recursive=True)

    # 2) 정규화(realpath)로 동일 파일 중복 제거
    norm = [os.path.realpath(p) for p in raw]

    # 3) 제외 폴더 필터링
    def is_excluded(p: str) -> bool:
        parts = set(os.path.normpath(p).split(os.sep))
        return any(ed in parts for ed in exclude_dirs)

    norm = [p for p in norm if not is_excluded(p)]

    # 4) 집합으로 중복 제거 후 정렬
    uniq = sorted(set(norm))
    print(f"[INFO] (collect_images) Found {len(uniq)} unique images "
          f"(from {len(raw)} raw matches).")
    return uniq

# --------- SuperPoint 추론 함수 (GPU) ----------
@torch.inference_mode()
def sp_infer(gray_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    gray_u8: (H,W) uint8
    return:
      kps  (N,2) float32
      desc (N,256) float32 (L2 normalized)
      scr  (N,) float32
    """
    ten = torch.from_numpy(gray_u8)[None, None].float().to(device) / 255.0
    out = sp({"image": ten})
    kps  = out["keypoints"][0].detach().cpu().numpy().astype(np.float32)             # (N,2)
    desc = out["descriptors"][0].permute(1,0).detach().cpu().numpy().astype(np.float32)  # (N,256)
    scr  = out["scores"][0].detach().cpu().numpy().astype(np.float32)                # (N,)
    return kps, desc, scr

# --------- SuperGlue 매칭 (SuperPoint 결과 재사용) ----------
@torch.inference_mode()
def sg_match(grayA_u8: np.ndarray, grayB_u8: np.ndarray,
             kA: np.ndarray, dA: np.ndarray, sA: np.ndarray,
             kB: np.ndarray, dB: np.ndarray, sB: np.ndarray, return_index: bool = False):
    """
    입력: 두 이미지의 그레이스케일, SuperPoint 결과
    출력: 매칭 좌표 xA, xB (둘 다 (M,2)), 매칭 신뢰도 (M,)
    """
    tA = torch.from_numpy(grayA_u8)[None, None].float().to(device) / 255.0
    tB = torch.from_numpy(grayB_u8)[None, None].float().to(device) / 255.0

    data = {
        "image0": tA, "image1": tB,
        "keypoints0": torch.from_numpy(kA)[None].float().to(device),   # (1,N0,2)
        "keypoints1": torch.from_numpy(kB)[None].float().to(device),
        "descriptors0": torch.from_numpy(dA.T)[None].float().to(device),  # (1,256,N0)
        "descriptors1": torch.from_numpy(dB.T)[None].float().to(device),
        "scores0": torch.from_numpy(sA)[None].float().to(device),
        "scores1": torch.from_numpy(sB)[None].float().to(device),
    }

    out = sg(data)
    m0   = out["matches0"][0].detach().cpu().numpy()   # (N0,)
    conf = out["matching_scores0"][0].detach().cpu().numpy()
    idx0 = np.where(m0 > -1)[0]
    idx1 = m0[idx0]
    xA, xB, c = kA[idx0], kB[idx1], conf[idx0]
    if return_index:
        return xA.astype(np.float64), xB.astype(np.float64), c.astype(np.float32), idx0.astype(np.int32), idx1.astype(np.int32)
    return xA.astype(np.float64), xB.astype(np.float64), c.astype(np.float32)


# --------- matching 규칙 1 ------------------
def build_image_pairs_by_rule(image_paths: list[str],
                              allowed_pairs: set[tuple[int,int]] = {(2,5), (4,8), (4,5)}) -> list[tuple[int,int]]:
    """
    파일명 규칙(가운데 숫자) 기반 허용 조합으로만 (i,j) 인덱스 페어 생성.
    반환: (i<j) 보장된 인덱스 쌍 리스트 (정렬됨)
    """
    def _extract_mid_code_by_name(path: str) -> int | None:
        """
        예) 'B1_02_00.jpg' -> 가운데 숫자 '02'를 int(2)로 반환
        """
        stem = os.path.splitext(os.path.basename(path))[0]
        nums = re.findall(r"\d+", stem)          # ['1','02','00'] 같은 꼴
        if len(nums) >= 2:
            try:
                return int(nums[1])
            except ValueError:
                return None
        return None

    # (5,4) == (4,5) 로 정규화
    allowed_pairs = {tuple(sorted(p)) for p in allowed_pairs}

    # 가운데 코드별 이미지 인덱스 모으기
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, p in enumerate(image_paths):
        code = _extract_mid_code_by_name(p)
        if code is not None:
            groups[code].append(idx)

    # 허용된 코드 조합끼리 모든 쌍 생성
    pair_set: set[tuple[int,int]] = set()
    for a, b in allowed_pairs:
        A = groups.get(a, [])
        B = groups.get(b, [])
        if not A or not B:
            continue
        for i in A:
            for j in B:
                if i == j:
                    continue
                i2, j2 = (i, j) if i < j else (j, i)
                pair_set.add((i2, j2))

    pairs = sorted(pair_set, key=lambda ij: (os.path.basename(image_paths[ij[0]]),
                                             os.path.basename(image_paths[ij[1]])))
    return pairs

# ---------- matching 규칙 2 ---------------
def build_cctv_pairs(image_paths: list[str],
    allowed_cam_pairs: set[tuple[int,int]] | None = None,
    include_same_cam_consecutive: bool = True
    ) -> list[tuple[int,int]]:
    """
    주차장 CCTV용 페어 생성:
    - 같은 프레임 번호의 서로 다른 카메라 쌍
    - 같은 카메라의 연속 프레임 쌍
    """
    def parse_cam_and_frame(path: str) -> tuple[int,int]:
        """
        파일명에서 카메라 번호와 프레임 번호를 추출.
        예) '.../B1_02_00.jpg' -> (2, 0)
        """
        stem = os.path.splitext(os.path.basename(path))[0]
        nums = re.findall(r"\d+", stem)  # ['1','02','00']
        if len(nums) >= 3:
            cam = int(nums[1])   # 두 번째 숫자 그룹 → 카메라 번호
            frame = int(nums[2]) # 세 번째 숫자 그룹 → 프레임 번호
            return cam, frame
        raise ValueError(f"Unexpected filename format: {path}")
    
    allowed_norm = None
    if allowed_cam_pairs is not None:
        allowed_norm = {tuple(sorted(p)) for p in allowed_cam_pairs}
        
    # (cam, frame) → 인덱스 매핑
    cam_frame_to_idx = {}
    cam_to_frames = defaultdict(list)
    frame_to_cams = defaultdict(list)

    for idx, p in enumerate(image_paths):
        cam, frame = parse_cam_and_frame(p)
        cam_frame_to_idx[(cam, frame)] = idx
        cam_to_frames[cam].append((frame, idx))
        frame_to_cams[frame].append((cam, idx))

    pairs = set()
                
    # 1) 같은 frame에서 서로 다른 카메라 쌍 (allowed_cam_pairs 적용)
    for frame, lst in frame_to_cams.items():
        # lst: [(cam, idx), ...]
        n = len(lst)
        for a in range(n):
            cam_a, idx_a = lst[a]
            for b in range(a+1, n):
                cam_b, idx_b = lst[b]
                # allowed 필터
                if allowed_norm is not None and tuple(sorted((cam_a, cam_b))) not in allowed_norm:
                    continue
                i, j = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
                pairs.add((i, j))
                
    # 2) 같은 카메라의 연속 프레임 쌍
    for cam, lst in cam_to_frames.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])  # frame 번호 순으로 정렬
        for (f1, i1), (f2, i2) in zip(lst_sorted, lst_sorted[1:]):
            if f2 == f1 + 1:  # 연속 프레임일 때만
                pairs.add(tuple(sorted((i1, i2))))

    # 정렬
    pairs = sorted(pairs, key=lambda ij: (os.path.basename(image_paths[ij[0]]),
                                          os.path.basename(image_paths[ij[1]])))
    return pairs


# --------- F/H 동시 추정 + 선택 ----------
def robust_F_or_H(xA: np.ndarray, xB: np.ndarray,
                  th_px: float = 1.8, conf: float = 0.999):
    # 0) 기본 전처리: dtype/contiguous/유한값/중복 제거
    xA = np.ascontiguousarray(xA, dtype=np.float64)
    xB = np.ascontiguousarray(xB, dtype=np.float64)
    finite = np.isfinite(xA).all(axis=1) & np.isfinite(xB).all(axis=1)
    xA, xB = xA[finite], xB[finite]
    if xA.shape[0] < 8:
        return "F", None, None  # F 추정 불가

    # 중복 제거(동일 좌표쌍 제거)
    pairs = np.hstack([xA, xB])
    _, keep = np.unique(pairs, axis=0, return_index=True)
    xA, xB = xA[keep], xB[keep]
    if xA.shape[0] < 8:
        return "F", None, None

    # 1) F/H 둘 다 시도하되, 실패/예외는 건너뛰기
    F, inlF = None, None
    H, inlH = None, None
    try:
        F, inlF = cv2.findFundamentalMat(
            xA, xB, cv2.USAC_MAGSAC, ransacReprojThreshold=th_px, confidence=conf
        )
    except cv2.error:
        F, inlF = None, None

    try:
        H, inlH = cv2.findHomography(
            xA, xB, cv2.USAC_MAGSAC, ransacReprojThreshold=th_px, confidence=conf
        )
    except cv2.error:
        H, inlH = None, None

    cntF = int(inlF.sum()) if inlF is not None else 0
    cntH = int(inlH.sum()) if inlH is not None else 0

    # 2) 둘 다 실패면 F=RANSAC로 다운그레이드 (USAC이 종종 assert로 죽는 페어 방지)
    if F is None and H is None:
        try:
            F, inlF = cv2.findFundamentalMat(xA, xB, cv2.FM_RANSAC, th_px, conf)
            cntF = int(inlF.sum()) if inlF is not None else 0
        except cv2.error:
            return "F", None, None

    # 3) 선택 규칙
    if cntH >= 1.5 * cntF and H is not None:
        return "H", H, inlH
    else:
        return "F", F, inlF


# ---------  E/R/t 추정 ----------
def pose_from_F(F: np.ndarray, xA: np.ndarray, xB: np.ndarray, K: np.ndarray):
    if F is None or K is None:
        return None, None, None
    E = K.T @ F @ K
    # rank-2 강제
    U, S, Vt = np.linalg.svd(E)
    S[-1] = 0.0
    E = U @ np.diag(S) @ Vt
    ok, R, t, mask = cv2.recoverPose(E, xA, xB, K)
    if ok <= 0:
        return None, None, None
    return E, R, t


# ======================= SfM + Triangulation + BA (projective→metric with guessed K) =======================
# ---------- 유틸 ----------
def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    return R

def R_to_rodrigues(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()

def build_K_guess_from_image(gray: np.ndarray, scale: float = 1.2) -> np.ndarray:
    H, W = gray.shape[:2]
    f = scale * max(H, W)
    cx, cy = W * 0.5, H * 0.5
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0,  1 ]], dtype=np.float64)
    return K

def triangulate_points(K, R1, t1, R2, t2, pts1, pts2):
    """pts1, pts2: (N,2) in pixel coords. Return (N,3) inhomogeneous 3D."""
    P1 = K @ np.hstack([R1, t1.reshape(3,1)])
    P2 = K @ np.hstack([R2, t2.reshape(3,1)])
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1,3).T
    pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1,3).T
    X_h = cv2.triangulatePoints(P1, P2, pts1_h[:2], pts2_h[:2])  # (4,N)
    X = (X_h[:3] / (X_h[3:] + 1e-12)).T  # (N,3)
    return X

def cheirality_filter(K, R, t, X):
    """앞쪽(z>0) 포인트만 통과."""
    P = K @ np.hstack([R, t.reshape(3,1)])
    X_h = np.hstack([X, np.ones((X.shape[0],1))]).T  # (4,N)
    x = P @ X_h
    z = (R @ X.T + t.reshape(3,1))[2]  # 카메라 좌표계 z
    mask = (x[2] > 0) & (z > 0)
    return mask.ravel()

# ======================= Logging-enhanced overrides (append to file) =======================
# 로깅 빈도 설정
LOG_CFG = {
    "img_step": 10,        # 특징추출: N장마다 한 번
    "pair_step": 500,      # 전쌍 매칭: N쌍마다 한 번
    "sfm_pair_step": 500,  # SfM용 F-쌍 스캔: N쌍마다 한 번
}

def L(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def _secs(t):
    return f"{t:.2f}s"

def _gpu_mem():
    try:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**2)
            reserv = torch.cuda.memory_reserved() / (1024**2)
            return f"GPU {alloc:.1f}MB alloc / {reserv:.1f}MB reserved"
    except Exception:
        pass
    return "GPU N/A"

# ---------- build_features 로깅 버전 ----------
def build_features(image_paths: list, out_dir: str = "./filtered") -> Dict[str, dict]:
    L(f"[Stage] Feature build start: {len(image_paths)} images → out_dir='{out_dir}'")
    t0 = time.perf_counter()
    feats = {}
    os.makedirs(out_dir, exist_ok=True)
    for idx, p in enumerate(image_paths, start=1):
        fe = FeatureExtractor(p)
        gray = fe.img_filtering(out_dir)      # (H,W) uint8 (CLAHE)
        kps, desc, scr = sp_infer(gray)
        feats[p] = {"gray": gray, "kps": kps, "desc": desc, "scr": scr}
        if (idx % LOG_CFG["img_step"] == 0) or (idx == len(image_paths)):
            L(f"[SP] {idx}/{len(image_paths)} | keypoints={len(kps)} | last='{os.path.basename(p)}' | {_gpu_mem()} | elapsed={_secs(time.perf_counter()-t0)}")
    L(f"[Stage] Feature build done in {_secs(time.perf_counter()-t0)}")
    return feats


# ----- 설정: BA에서 사용할 최대 포인트 수(관측수 많은 순으로 선택) -----
BA_CFG = {
    "max_points": 15000,    # 메모리 여유에 맞춰 8k~20k 정도로 조절
    "max_nfev": 100,
    "loss": "huber",        # 노이즈/외란에 강한 로버스트 손실
    "f_scale": 1.0,
    "method": "trf",        # 'lm' 금지! 희소야코비안은 'trf' 필요
    "jac_kind": "2-point",  # 수치미분 (빠름/안정)
}

def _downselect_points_for_ba(lm: np.ndarray, obs: list, max_points: int):
    """
    관측 수가 많은 포인트부터 상위 max_points만 남긴다.
    obs: list of (cam_idx, pt3d_idx, uv)
    return: lm_new, obs_new, old2new (dict)
    """
    if lm.shape[0] <= max_points:
        return lm, obs, {i:i for i in range(lm.shape[0])}

    # 포인트별 관측 수 카운트
    counts = np.zeros((lm.shape[0],), dtype=np.int32)
    for _, pid, _ in obs:
        if 0 <= pid < lm.shape[0]:
            counts[pid] += 1
    order = np.argsort(-counts)  # 내림차순
    keep = order[:max_points]
    keep_set = set(keep.tolist())

    old2new = {int(o):int(n) for n,o in enumerate(keep)}
    lm_new = lm[keep]

    obs_new = []
    for cid, pid, uv in obs:
        if pid in old2new:
            obs_new.append((cid, old2new[pid], uv))
    return lm_new, obs_new, old2new

def _build_jacobian_sparsity(num_cams: int, num_pts: int,
                             obs_cam_slot: np.ndarray,
                             obs_pt_idx: np.ndarray,
                             refine_focal: bool):
    """
    희소 야코비안 패턴 구성.
    각 관측(2 residuals)은 해당 카메라의 6개 파라미터 + 해당 포인트의 3개 파라미터(+옵션 intrinsics 4개)에만 의존.
    """
    n_obs = obs_cam_slot.shape[0]
    n_rows = 2 * n_obs
    n_cols = 6 * num_cams + 3 * num_pts + (4 if refine_focal else 0)

    A = lil_matrix((n_rows, n_cols), dtype=np.bool_)
    for k in range(n_obs):
        r0 = 2*k; r1 = r0 + 1
        cslot = int(obs_cam_slot[k])
        pid   = int(obs_pt_idx[k])

        # 카메라 파라미터 6개
        c0 = 6*cslot
        A[r0, c0:c0+6] = True
        A[r1, c0:c0+6] = True

        # 포인트 파라미터 3개
        p0 = 6*num_cams + 3*pid
        A[r0, p0:p0+3] = True
        A[r1, p0:p0+3] = True

        # (옵션) intrinsics 4개
        if refine_focal:
            i0 = 6*num_cams + 3*num_pts
            A[r0, i0:i0+4] = True
            A[r1, i0:i0+4] = True

    return A.tocsr()


# ===== reconstruct_sfm 재정의: BA 부분을 희소-TRF로 교체 + 포인트 다운샘플 =====
def reconstruct_sfm(image_paths: list[str], feats: dict,
                    refine_focal: bool = False,
                    pnp_min_pts: int = 15,
                    min_inliers_seed: int = 20,
                    min_tri_deg: float = 0.5,
                    min_cheir_ratio: float = 0.55,
                    sg_topk: int = 4000,
                    image_pairs: list[tuple[int,int]] | None = None) -> dict:
    # ---------- 내부 유틸 ----------
    @torch.inference_mode()
    def _sg_match_with_indices(fa: dict, fb: dict, cap: int):
        grayA, grayB = fa["gray"], fb["gray"]
        kA0, dA0, sA0 = fa["kps"], fa["desc"], fa["scr"]
        kB0, dB0, sB0 = fb["kps"], fb["desc"], fb["scr"]
        capA = min(cap, len(kA0)); capB = min(cap, len(kB0))
        selA = np.argpartition(sA0, -capA)[-capA:]; selA = selA[np.argsort(sA0[selA])[::-1]]
        selB = np.argpartition(sB0, -capB)[-capB:]; selB = selB[np.argsort(sB0[selB])[::-1]]
        kA, dA, sA = kA0[selA], dA0[selA], sA0[selA]
        kB, dB, sB = kB0[selB], dB0[selB], sB0[selB]
        tA = torch.from_numpy(grayA)[None, None].float().to(device)/255.0
        tB = torch.from_numpy(grayB)[None, None].float().to(device)/255.0
        data = {
            "image0": tA, "image1": tB,
            "keypoints0": torch.from_numpy(kA)[None].float().to(device),
            "keypoints1": torch.from_numpy(kB)[None].float().to(device),
            "descriptors0": torch.from_numpy(dA.T)[None].float().to(device),
            "descriptors1": torch.from_numpy(dB.T)[None].float().to(device),
            "scores0": torch.from_numpy(sA)[None].float().to(device),
            "scores1": torch.from_numpy(sB)[None].float().to(device),
        }
        if torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = sg(data)
        else:
            out = sg(data)
        m0   = out["matches0"][0].detach().cpu().numpy().astype(np.int32)
        conf = out["matching_scores0"][0].detach().cpu().numpy()
        idx0 = np.where(m0 > -1)[0].astype(np.int32); idx1 = m0[idx0]
        xA = kA[idx0].astype(np.float64); xB = kB[idx1].astype(np.float64); c = conf[idx0].astype(np.float32)
        idA = selA[idx0].astype(np.int32); idB = selB[idx1].astype(np.int32)
        del out, data, tA, tB
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return xA, xB, c, idA, idB

    def _sg_match_safe(fa, fb, topk):
        for cap in [topk, max(512, topk//2), max(256, topk//4)]:
            try:
                return _sg_match_with_indices(fa, fb, cap)
            except torch.cuda.OutOfMemoryError:
                L(f"[SG] OOM at cap={cap} → clear & retry smaller")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        L("[SG] fallback to CPU matching")
        sg_cpu = sg.to("cpu")
        try:
            return _sg_match_with_indices(fa, fb, 512)
        finally:
            sg_cpu.to(device)

    def _triangulation_angle_median(x0, x1, K, R, t):
        if x0.size == 0 or x1.size == 0: return 0.0
        u0 = cv2.undistortPoints(x0.reshape(-1,1,2), K, None).reshape(-1,2)
        u1 = cv2.undistortPoints(x1.reshape(-1,1,2), K, None).reshape(-1,2)
        v0 = np.concatenate([u0, np.ones((u0.shape[0],1))], 1)
        v1 = np.concatenate([u1, np.ones((u1.shape[0],1))], 1)
        v0 = v0/(np.linalg.norm(v0, axis=1, keepdims=True)+1e-12)
        v1 = v1/(np.linalg.norm(v1, axis=1, keepdims=True)+1e-12)
        v1_in0 = (R.T @ v1.T).T
        cos = np.clip(np.sum(v0*v1_in0, axis=1), -1, 1)
        return float(np.median(np.degrees(np.arccos(cos))))

    def _pair_key(i, j): return (i, j) if i < j else (j, i)

    # ---------- 시작 ----------
    L("[SfM] Reconstruction start (wider pair graph)")
    t_all = time.perf_counter()
    K = build_K_guess_from_image(feats[image_paths[0]]["gray"], scale=1.2)
    L(f"[SfM] K guess: fx≈fy={K[0,0]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    # 모든 페어 보존 + 시드용 강한 게이트
    pair_all = {}      # 등록/삼각측량용: F-인라이어만 충족하면 저장
    strong = []        # 시드 후보: 게이트 통과
    # 사용할 페어 집합 결정
    if image_pairs is None:
        pairs_iter = itertools.combinations(range(len(image_paths)), 2)
    else:
        # 방어: 범위/정렬 보장
        pairs_iter = [(min(i,j), max(i,j)) for (i,j) in image_pairs
                    if 0 <= i < len(image_paths) and 0 <= j < len(image_paths) and i != j]
        # 중복 제거 + 안정 정렬
        pairs_iter = sorted(set(pairs_iter))

    for i, j in pairs_iter:
        fa, fb = feats[image_paths[i]], feats[image_paths[j]]
        xA, xB, _, idA, idB = _sg_match_safe(fa, fb, sg_topk)
        if xA.shape[0] < 8: 
            continue
        model, M, inliers = robust_F_or_H(xA, xB, th_px=2.5, conf=0.999)
        if model != "F" or inliers is None: 
            continue
        inl = inliers.ravel().astype(bool)
        if inl.sum() < 12: 
            continue

        pair_all[_pair_key(i,j)] = {"x_i":xA, "x_j":xB, "idx_i":idA, "idx_j":idB, "inl":inl}

        # (이하 seed 후보 평가 그대로 유지)
        F, _ = cv2.findFundamentalMat(xA[inl], xB[inl], cv2.USAC_MAGSAC, 2.0, 0.999)
        if F is None: 
            continue
        E = K.T @ F @ K; U,S,Vt = np.linalg.svd(E); S[-1]=0; E = U@np.diag(S)@Vt
        ok, R01, t01, _ = cv2.recoverPose(E, xA[inl], xB[inl], K)
        if ok <= 0: 
            continue
        tri_med = _triangulation_angle_median(xA[inl], xB[inl], K, R01, t01.ravel())
        cheir = cheirality_filter(K, np.eye(3), np.zeros(3),
                                triangulate_points(K, np.eye(3), np.zeros(3), R01, t01.ravel(),
                                                    xA[inl], xB[inl])).mean()
        if tri_med >= min_tri_deg and cheir >= min_cheir_ratio and inl.sum() >= min_inliers_seed:
            strong.append((i, j, inl.sum(), tri_med, R01, t01.ravel()))

    L(f"[SfM] pair_all={len(pair_all)} | strong(seed candidates)={len(strong)}")
    assert len(pair_all) > 0, "No usable pairs found."

    if len(strong)==0:
        # 강한 후보가 없으면 인라이어 최다 페어로 시드 설정(구조가 거칠 수 있음)
        k = max(pair_all.keys(), key=lambda k: pair_all[k]["inl"].sum())
        i0,j0 = k
        x0 = pair_all[k]["x_i"][pair_all[k]["inl"]]; x1 = pair_all[k]["x_j"][pair_all[k]["inl"]]
        F, _ = cv2.findFundamentalMat(x0, x1, cv2.USAC_MAGSAC, 2.0, 0.999)
        E = K.T @ F @ K; U,S,Vt = np.linalg.svd(E); S[-1]=0; E = U@np.diag(S)@Vt
        ok, R01, t01, _ = cv2.recoverPose(E, x0, x1, K)
    else:
        i0,j0, ninl, tri_med, R01, t01 = max(strong, key=lambda x: x[2]*max(x[3],1e-3))
        k = _pair_key(i0,j0)
        inl = pair_all[k]["inl"]
        x0 = pair_all[k]["x_i"][inl]; x1 = pair_all[k]["x_j"][inl]
    L(f"[SfM] Seed: ({i0},{j0})")

    # 초기 두 뷰 & 삼각측량
    cams = { i0:{'R':np.eye(3), 't':np.zeros(3)},
             j0:{'R':R01,       't':t01} }
    Xij = triangulate_points(K, cams[i0]['R'], cams[i0]['t'], cams[j0]['R'], cams[j0]['t'], x0, x1)
    mask_ch = cheirality_filter(K, cams[i0]['R'], cams[i0]['t'], Xij) & cheirality_filter(K, cams[j0]['R'], cams[j0]['t'], Xij)
    X = Xij[mask_ch]; x0v = x0[mask_ch]; x1v = x1[mask_ch]
    id0v = pair_all[_pair_key(i0,j0)]["idx_i"][pair_all[_pair_key(i0,j0)]["inl"]][mask_ch]
    id1v = pair_all[_pair_key(i0,j0)]["idx_j"][pair_all[_pair_key(i0,j0)]["inl"]][mask_ch]
    L(f"[SfM] Seed triangulated points: {len(X)}")

    lm = []; obs = []; obs_map = {}
    for pid, (P,u0,u1,k0,k1) in enumerate(zip(X, x0v, x1v, id0v, id1v)):
        lm.append(P); obs.append((i0,pid,u0)); obs_map[(i0,int(k0))]=pid
        obs.append((j0,pid,u1)); obs_map[(j0,int(k1))]=pid
    lm = np.array(lm, dtype=np.float64)

    # 증분 등록
    registered = set([i0,j0])
    remaining = [k for k in range(len(image_paths)) if k not in registered]
    for k in remaining:
        p3d,p2d=[],[]
        for r in list(registered):
            pm = pair_all.get(_pair_key(r,k))
            if pm is None: continue
            inl = pm["inl"]
            if r<k:
                id_r, id_k = pm["idx_i"][inl], pm["idx_j"][inl]
                xr,   xk   = pm["x_i"][inl],  pm["x_j"][inl]
            else:
                id_r, id_k = pm["idx_j"][inl], pm["idx_i"][inl]
                xr,   xk   = pm["x_j"][inl],  pm["x_i"][inl]
            for kp_r, uv_k in zip(id_r, xk):
                pid = obs_map.get((r,int(kp_r)))
                if pid is not None:
                    p3d.append(lm[pid]); p2d.append(uv_k)
        if len(p3d) < max(pnp_min_pts,6): continue
        p3d = np.asarray(p3d,float); p2d = np.asarray(p2d,float)
        ok, rvec, tvec, inlPnP = cv2.solvePnPRansac(p3d,p2d,K,None,
                                flags=cv2.SOLVEPNP_EPNP,reprojectionError=3.0,
                                confidence=0.999,iterationsCount=5000)
        if not ok or inlPnP is None or len(inlPnP)<6: continue
        ok2, rvec, tvec = cv2.solvePnP(p3d[inlPnP.ravel()], p2d[inlPnP.ravel()], K, None,
                                       rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok2: continue
        Rk = rodrigues_to_R(rvec); tk = tvec.flatten()
        cams[k] = {'R':Rk,'t':tk}; registered.add(k)
        L(f"[SfM] Registered view {k} | inliers={len(inlPnP)} | cams={len(registered)}")

        # 새 포인트 삼각측량(완화: sel>=8)
        new_total = 0
        for r in list(registered):
            if r==k: continue
            pm = pair_all.get(_pair_key(r,k))
            if pm is None: continue
            inl = pm["inl"]
            if r<k:
                id_r, id_k = pm["idx_i"][inl], pm["idx_j"][inl]
                xr,   xk   = pm["x_i"][inl],  pm["x_j"][inl]
            else:
                id_r, id_k = pm["idx_j"][inl], pm["idx_i"][inl]
                xr,   xk   = pm["x_j"][inl],  pm["x_i"][inl]
            sel = [t for t,kp_r in enumerate(id_r) if (r,int(kp_r)) not in obs_map]
            if len(sel) < 8: continue
            xr_sel, xk_sel = xr[sel], xk[sel]
            idr_sel, idk_sel = id_r[sel], id_k[sel]
            X_new = triangulate_points(K, cams[r]['R'], cams[r]['t'], cams[k]['R'], cams[k]['t'], xr_sel, xk_sel)
            mask = cheirality_filter(K, cams[r]['R'], cams[r]['t'], X_new) & cheirality_filter(K, cams[k]['R'], cams[k]['t'], X_new)
            if not np.any(mask): continue
            X_new = X_new[mask]; xr_sel = xr_sel[mask]; xk_sel = xk_sel[mask]
            idr_sel = idr_sel[mask]; idk_sel = idk_sel[mask]
            base = len(lm); lm = np.vstack([lm, X_new])
            for t,(uv_r,uv_k,kp_r,kp_k) in enumerate(zip(xr_sel,xk_sel,idr_sel,idk_sel)):
                pid = base+t
                obs.append((r,pid,uv_r)); obs_map[(r,int(kp_r))]=pid
                obs.append((k,pid,uv_k)); obs_map[(k,int(kp_k))]=pid
            new_total += len(X_new)
        if new_total>0:
            L(f"[SfM] View {k} triangulated new points: {new_total} | total={len(lm)}")

    # ---------- 희소 BA ----------
    cam_indices = sorted(cams.keys())
    cam_id_to_slot = {cid:s for s,cid in enumerate(cam_indices)}
    obs_cam_slot = np.array([cam_id_to_slot[c] for (c,_,_) in obs], np.int32)
    obs_pt_idx   = np.array([p for (_,p,_) in obs], np.int32)
    obs_uv       = np.array([u for (_,_,u) in obs], np.float64)

    lm, obs, old2new = _downselect_points_for_ba(lm, obs, BA_CFG["max_points"])
    if len(old2new)!=0:
        obs_cam_slot = np.array([cam_id_to_slot[c] for (c,_,_) in obs], np.int32)
        obs_pt_idx   = np.array([p for (_,p,_) in obs], np.int32)
        obs_uv       = np.array([u for (_,_,u) in obs], np.float64)

    cams_init = np.hstack([np.hstack([R_to_rodrigues(cams[cid]['R']), cams[cid]['t']]) for cid in cam_indices])
    pts_init  = lm.reshape(-1)
    x0 = (np.hstack([cams_init, pts_init, np.array([K[0,0],K[1,1],K[0,2],K[1,2]])])
          if refine_focal else np.hstack([cams_init, pts_init]))

    def unpack_params(x):
        off=0; cam_params=x[off:off+6*len(cam_indices)]; off+=6*len(cam_indices)
        pts_params=x[off:off+3*lm.shape[0]]; off+=3*lm.shape[0]
        K_=K.copy()
        if refine_focal:
            intr=x[off:off+4]; K_[0,0],K_[1,1],K_[0,2],K_[1,2]=intr
        cams_list=[]; 
        for s in range(len(cam_indices)):
            rvec=cam_params[6*s:6*s+3]; tvec=cam_params[6*s+3:6*s+6]; cams_list.append((rvec,tvec))
        return cams_list, pts_params.reshape(-1,3), K_

    def project(K_, rvec, tvec, X3d):
        R = rodrigues_to_R(rvec); Xc = (R @ X3d.T + tvec.reshape(3,1)).T
        x = (K_ @ Xc.T).T; return x[:,:2]/x[:,2:3]

    def residuals(x):
        cams_list, pts, K_ = unpack_params(x)
        r = np.zeros((obs_uv.shape[0]*2,), np.float64)
        for ii in range(obs_uv.shape[0]):
            cslot=obs_cam_slot[ii]; pid=obs_pt_idx[ii]; uv=obs_uv[ii]
            rvec,tvec=cams_list[cslot]; uv_hat=project(K_,rvec,tvec,pts[pid:pid+1])[0]
            r[2*ii:2*ii+2]=uv_hat-uv
        return r

    L(f"[BA] cameras={len(cam_indices)}, points={lm.shape[0]}, obs={len(obs)} | refine_focal={refine_focal}")
    if len(obs)>0 and lm.shape[0]>0:
        Jsp = _build_jacobian_sparsity(len(cam_indices), lm.shape[0], obs_cam_slot, obs_pt_idx, refine_focal)
        res = least_squares(residuals, x0, method=BA_CFG["method"], jac=BA_CFG["jac_kind"],
                            jac_sparsity=Jsp, max_nfev=BA_CFG["max_nfev"], loss=BA_CFG["loss"],
                            f_scale=BA_CFG["f_scale"], x_scale="jac", verbose=2)
        cams_list, pts_opt, K_opt = unpack_params(res.x)
        for s,cid in enumerate(cam_indices):
            rvec,tvec=cams_list[s]; cams[cid]['R']=rodrigues_to_R(rvec); cams[cid]['t']=tvec
        lm=pts_opt; K=K_opt
        L(f"[BA] done | RMS(px)={np.sqrt(np.mean(res.fun**2)):.3f}")
    else:
        L("[BA] Skipped (insufficient observations).")

    L(f"[SfM] Reconstruction done | cams={len(cams)} | pts={len(lm)}")
    return {'K':K,'cams':cams,'points3d':lm,'observations':obs}


def save_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None):
    pts = points.astype(np.float32)
    if colors is None:
        colors = np.full((pts.shape[0],3), 200, dtype=np.uint8)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {pts.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for p, c in zip(pts, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    L(f"[OUT] saved point cloud: {path} | pts={len(points)}")


def run_sfm_and_ba(image_root: str,
                   pattern: str = "**/*.jpg",
                   refine_focal: bool = False,
                   ply_path: str = "recon.ply"):
    L(f"[MAIN] run_sfm_and_ba start | root='{image_root}' pattern='{pattern}'")
    image_paths = collect_images(image_root, pattern=pattern,
                                 exclude_dirs=("filtered", ".cache", ".git"))
    assert len(image_paths) >= 2
    feats = build_features(image_paths)
    # 카메라=5대, 각 40프레임 가정
    allowed_cam_pairs = {(2,5), (4,8), (4,5), (7,8)}
    
    pairs = build_cctv_pairs(
        image_paths,
        allowed_cam_pairs=allowed_cam_pairs,          # 동시간대 카메라쌍 필터
        include_same_cam_consecutive=True             # 동일카메라 연속프레임도 사용
    )

    recon = reconstruct_sfm(image_paths, feats,
                            refine_focal=refine_focal,
                            image_pairs=pairs)


    pts = recon['points3d']
    if pts.shape[0] > 0:
        save_ply(ply_path, pts)
    else:
        L("[OUT] No 3D points to save.")
    L("[MAIN] run_sfm_and_ba done")

# ---------- 실행 ----------
if __name__ == "__main__":
    
    run_sfm_and_ba(image_root="images", pattern="**/**/*.jpg",
                   refine_focal=True, ply_path="recon.ply")
    