import pycolmap
import numpy as np
import os
import cv2
from database import COLMAPDatabase, blob_to_array, array_to_blob
from database import image_ids_to_pair_id, pair_id_to_image_ids
import torch
from math import ceil, log
import itertools
from scipy.spatial.transform import Rotation as Rscipy
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import pandas as pd 

# OpenCV의 KMP 라이브러리 중복 로딩 방지
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


#==================================유틸리티=================================
# 테이블 시각화 및 저장
def save_table_heatmap(table, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(table, cmap='hot', interpolation='nearest', origin='upper',aspect='auto')
    plt.colorbar(label='Keypoint Count')
    plt.title('Grid Keypoint Count Heatmap')
    plt.xlabel('Grid Col')
    plt.ylabel('Grid Row')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 행렬 E 계산
def compute_E(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Compute Essential matrix from Fundamental matrix and camera intrinsics.
    
    Args:
        F (np.ndarray): Fundamental matrix (3x3)
        K (np.ndarray): Camera intrinsic matrix (3x3)
    
    Returns:
        E (np.ndarray): Essential matrix (3x3)
    """
    if F.shape != (3, 3):
        raise ValueError(f"F must be 3x3, got {F.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    
    # E = K^T * F * K
    E = K.T @ F @ K
    
    # Normalize (선택적) → scale factor 무시하고 안정성 확보
    E /= np.linalg.norm(E)

    return E

# 행렬 K 계산 
def estimate_K(img1):
        img1 = os.path.basename(img1)
        dir = f"filtered_img/filter_{img1}"
        img1 = cv2.imread(dir)
        cx = 0
        cy = 0
        f = 0
        h ,w = img1.shape[0], img1.shape[1]
        f = 1.2 * max(w,h)
        cx = (w-1) /2
        cy = (h-1) /2
        K = np.array([[f, 0.0, cx],
                  [0.0, f, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

        return K
# 행렬 K 계산 
def estimate_K2(img1):
        img1 = os.path.basename(img1)
        dir = f"filtered_img/{img1}"
        img1 = cv2.imread(dir)
        cx = 0
        cy = 0
        f = 0
        h ,w = img1.shape[0], img1.shape[1]
        f = 1.2 * max(w,h)
        cx = (w-1) /2
        cy = (h-1) /2
        K = np.array([[f, 0.0, cx],
                  [0.0, f, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

        return K
# -----------------------------
# 1) 단일 페어 진단 함수 (raw or refined matches)
# -----------------------------
def diag_pair(xA: np.ndarray, xB: np.ndarray, K: np.ndarray,
              thF: float = 1.5, thH: float = 2.0,
              conf: float = 0.999, iters: int = 10000):
    """
    입력:
      xA, xB : (N,2) pixel 좌표 (raw 매칭이든 정제 매칭이든 무방)
      K      : (3,3) 카메라 내부 파라미터
    출력:
      metrics dict = {
        'F_inliers': int,
        'H_inliers': int,
        'R_H_over_F': float,      # H/F 비율
        'tri_med_deg': float,     # F-inliers에서 recoverPose 후 광선 각도 중앙값(도)
        'ok_F': bool, 'ok_H': bool
      }
    """
    metrics = {'F_inliers': 0, 'H_inliers': 0,
               'R_H_over_F': np.inf, 'tri_med_deg': 0.0,
               'ok_F': False, 'ok_H': False}

    if xA.shape[0] < 4 or xB.shape[0] < 4:
        return metrics

    # F (USAC MAGSAC)
    USAC_MAGSAC = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    F, mF = cv2.findFundamentalMat(xA, xB, method=USAC_MAGSAC,
                                   ransacReprojThreshold=float(thF),
                                   confidence=float(conf),
                                   maxIters=int(iters))
    if mF is not None and F is not None:
        inlF = mF.ravel().astype(bool)
        F_inl = int(inlF.sum())
        metrics['F_inliers'] = F_inl
        metrics['ok_F'] = (F_inl >= 8)

        # 삼각측량 각도 중앙값
        if F_inl >= 8:
            x1 = xA[inlF]; x2 = xB[inlF]
            # Essential
            E = K.T @ F @ K
            try:
                _, R, t, _ = cv2.recoverPose(E, x1, x2, K)
                # 정규화 광선
                x1n = cv2.undistortPoints(x1.reshape(-1,1,2), K, None).reshape(-1,2)
                x2n = cv2.undistortPoints(x2.reshape(-1,1,2), K, None).reshape(-1,2)
                r1 = np.column_stack([x1n, np.ones(len(x1n))])
                r1 /= (np.linalg.norm(r1, axis=1, keepdims=True) + 1e-12)
                r2 = np.column_stack([x2n, np.ones(len(x2n))])
                r2 = (R.T @ r2.T).T
                r2 /= (np.linalg.norm(r2, axis=1, keepdims=True) + 1e-12)
                ang = np.degrees(np.arccos(np.clip(np.sum(r1*r2, axis=1), -1, 1)))
                ang = np.minimum(ang, 180.0 - ang)    
                tri_med = float(np.median(ang))
                metrics['tri_med_deg'] = float(np.median(tri_med))
            except Exception:
                # recoverPose 실패 시 0도로 둠
                metrics['tri_med_deg'] = 0.0

    # H (USAC MAGSAC)
    H, mH = cv2.findHomography(xA, xB, method=USAC_MAGSAC,
                               ransacReprojThreshold=float(thH))
    if mH is not None and H is not None:
        inlH = mH.ravel().astype(bool)
        H_inl = int(inlH.sum())
        metrics['H_inliers'] = H_inl
        metrics['ok_H'] = (H_inl >= 4)

    # 비율
    F_inl = metrics['F_inliers']
    metrics['R_H_over_F'] = (metrics['H_inliers'] / F_inl) if F_inl > 0 else np.inf
    return metrics


# -----------------------------
# 2) DB 기반 다수 페어 진단 → 표/CSV + print 요약
# -----------------------------
def diagnose_pairs_from_db(db_conn,
                           get_pairs_fn,      # -> list of (id1,id2,name1,name2)
                           load_kp_fn,        # (db,id) -> (N,4) or (N,2) np.float32
                           load_matches_fn,   # (db,id1,id2)->(M,2) int
                           K_from_name_fn,    # name -> (3,3) np.float64
                           save_csv_path="pair_diagnostics.csv",
                           r_plane_thresh: float = 1.8,
                           tri_med_thresh_deg: float = 1.0,
                           thF: float = 1.5, thH: float = 2.0, conf: float = 0.999, iters: int = 10000):
    """
    DB에서 페어를 순회하여 진단 값을 계산하고 DataFrame/CSV로 저장 + 콘솔 출력 요약.
    """
    pairs = get_pairs_fn(db_conn)
    rows = []
    for (id1, id2, name1, name2) in pairs:
        kpA = load_kp_fn(db_conn, id1)  # (NA,2 or 4)
        kpB = load_kp_fn(db_conn, id2)  # (NB,2 or 4)
        if kpA.size == 0 or kpB.size == 0:
            continue
        # 좌표만 취함
        xA_all = kpA[:, :2].astype(np.float64)
        xB_all = kpB[:, :2].astype(np.float64)

        m = load_matches_fn(db_conn, id1, id2)  # (M,2)
        if m.size == 0:
            continue
        xA = xA_all[m[:,0]]
        xB = xB_all[m[:,1]]

        # K (name1 기준으로 불러오되 동일 카메라면 name2도 동일)
         
        K = np.asarray(K_from_name_fn(os.path.basename(name1)), dtype=np.float64)

        metrics = diag_pair(xA, xB, K, thF=thF, thH=thH, conf=conf, iters=iters)
        R = metrics['R_H_over_F']
        tri = metrics['tri_med_deg']
        plane_like = (R >= r_plane_thresh) and (tri <= tri_med_thresh_deg)

        rows.append({
            "id1": id1, "id2": id2,
            "name1": name1, "name2": name2,
            "F_inliers": metrics['F_inliers'],
            "H_inliers": metrics['H_inliers'],
            "R_H_over_F": R,
            "tri_med_deg": tri,
            "plane_dominant": bool(plane_like)
        })

    if not rows:
        print("[DIAG] 진단 대상 페어가 없습니다.")
        return None

    df = pd.DataFrame(rows).sort_values(
        ["plane_dominant", "F_inliers", "tri_med_deg", "R_H_over_F"],
        ascending=[False, False, False, True]
    )

    # 저장
    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
        df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
        print(f"[DIAG] CSV 저장: {save_csv_path}")

    # 콘솔 요약
    n = len(df)
    n_plane = int(df["plane_dominant"].sum())
    print(f"[DIAG] 총 페어={n}, 평면 지배 판정={n_plane} ({n_plane/n*100:.1f}%)")
    print("[DIAG] 상위 5개 (F_inliers 내림차순):")
    print(df.sort_values("F_inliers", ascending=False)[
        ["name1","name2","F_inliers","H_inliers","R_H_over_F","tri_med_deg","plane_dominant"]
    ])

    return df


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

    def extract_features(self,img,per_cell=50, rows=24, cols=32,
                         use_rootsift=True, upsample_scale=1.5):
        """
        OpenCV SIFT로 많이 뽑은 뒤, grid별 균일 샘플링을 적용.
        COLMAP DB 저장에 맞춘 포맷(keypoints: (N,4) float32, descriptors: (128,N) uint8)으로 반환.

        Args:
            img: 입력 이미지 
            rows, cols: grid 분할 수
            per_cell_hi: 각 셀당 최대 유지할 keypoint 수

        Returns:
            kp_xy: (N, 4) float32, [x, y]
            desc_u8: (128, N) uint8
            table_features: 각 셀에 잡힌 원시 keypoint 개수 (진단용)
        """
        # 0) 업샘플 후 검출
        if upsample_scale != 1.0:
            img_up = cv2.resize(img, None, fx=upsample_scale, fy=upsample_scale,
                                interpolation=cv2.INTER_LANCZOS4)
        else:
            img_up = img

        sift = cv2.SIFT_create(
            nfeatures=50000,
            contrastThreshold=0.01,   
            edgeThreshold=12,
            sigma=1.6
        )

        
        keypoints, descriptors = sift.detectAndCompute(img_up, None)
        
        if len(keypoints) == 0:
            return (np.zeros((0,4), np.float32),  # x,y,scale,angle
                    np.zeros((128,0), np.uint8),
                    np.zeros((rows, cols), int))
        
        idx_sorted = np.argsort([-kp.response for kp in keypoints])

        # 1) grid 설정
        h, w = img.shape
        cell_h, cell_w = max(1, h // rows), max(1, w // cols) 

        # 2) 2단계 샘플링: (a) 공간 균형, (b) 전역 상위 보강
        per_cell_hi = per_cell  # 1단계 상한
        counts_raw = np.zeros((rows, cols), dtype=int)
        picked = []
        kept = np.zeros((rows, cols), dtype=int)

        # (a) 공간(그리드) 균형
        for i in idx_sorted:
            kp = keypoints[i]
            x, y = kp.pt
            c = min(int(x // cell_w), cols - 1)
            r = min(int(y // cell_h), rows - 1)
            counts_raw[r, c] += 1
            if kept[r, c] < per_cell_hi:
                picked.append(i)
                kept[r, c] += 1

        # (b) 전역 상위 보강: 남은 특징점에서 상위 반응 추가
        cap = int(rows * cols * per_cell_hi * 1.5)
        if len(picked) < cap:
            for i in idx_sorted:
                if i in picked:
                    continue
                picked.append(i)
                if len(picked) >= cap:
                    break

        print(f"|{self.jpg_name}의 총 feature 수: {len(picked)}")

        picked = np.array(picked, dtype=int)

        # 3) COLMAP 호환: (x,y,scale,angle[rad]) + RootSIFT→uint8
        xs, ys, scales, angles = [], [], [], []
        for i in picked:
            kp = keypoints[i]
            x, y = kp.pt
            if upsample_scale != 1.0:
                x /= upsample_scale
                y /= upsample_scale

            xs.append(x)
            ys.append(y)
            scales.append(kp.size)  # COLMAP은 scale(σ)로 사용, size 그대로 넣어도 OK
            angles.append(np.deg2rad(kp.angle if kp.angle is not None else 0.0))

        kp_xy4 = np.vstack([xs, ys, scales, angles]).T.astype(np.float32)  # (N,4)

        # descriptors: (N,128) float32
        desc = descriptors[picked, :].astype(np.float32)

        if use_rootsift:
            # RootSIFT: L1 normalize → sqrt → L2 normalize → 512× → uint8
            desc += 1e-12 # zerodivision error 방지
            desc /= np.sum(desc, axis=1, keepdims=True)   # L1
            desc = np.sqrt(desc)                          # sqrt
            desc /= (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12) # L2
        else:
            # 안전 L2
            desc /= (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12)

        desc_u8 = np.clip(desc * 512.0, 0, 255).astype(np.uint8).T  # (128,N)

        return kp_xy4, desc_u8, counts_raw , kept
    

    def ensure_camera(self,db: COLMAPDatabase, model="SIMPLE_RADIAL", width=None, height=None, params=None):
        """
        DB에 카메라가 없으면 하나 추가하고, 있으면 첫 카메라 id 반환.
        실제로는 카메라별(=캠 ID별)로 하나씩 관리하는 게 좋습니다.
        """
        # 기본 파라미터 대충 세팅 (EXIF 없을 때)
        # f ~ 1.2 * max(W,H), cx=W/2, cy=H/2, k1=0
        if width is None or height is None:
            raise ValueError("width/height가 필요합니다.")
        if params is None:
            f = 1.2 * max(width, height)
            cx, cy = width / 2.0, height / 2.0
            k1 = 0.0
            params = np.array([f, cx, cy, k1], dtype=np.float64)

        # 모델 맵핑
        CAMERA_MODELS = {
            "SIMPLE_PINHOLE": 0,
            "PINHOLE": 1,
            "SIMPLE_RADIAL": 2,
            "RADIAL": 3,
            "OPENCV": 4,
            "OPENCV_FISHEYE": 8,
            "FOV": 9,
        }
        if model not in CAMERA_MODELS:
            raise ValueError(f"Unknown camera model: {model}")
        # 동일한 카메라가 이미 있는지 확인
        rows = db.execute(
            "SELECT camera_id, model, width, height, params FROM cameras"
        ).fetchall()
        for row in rows:
            cam_id, db_model, db_width, db_height, db_params_blob = row
            db_params = blob_to_array(db_params_blob, np.float64)
            if (
                db_model == CAMERA_MODELS[model]
                and db_width == width
                and db_height == height
                and np.allclose(db_params, params)
            ):
                return cam_id  # 이미 존재하는 카메라 id 반환

        cam_id = db.add_camera(
            model=CAMERA_MODELS[model],
            width=width,
            height=height,
            params=params,
            prior_focal_length=False
        )
        return cam_id


    def store_features(self, image_path, kp_xy, desc_u8):
        """
        COLMAP DB에 keypoints/descriptors 저장.
        Args:
            database_path: 'database.db'
            image_path: 이미지 파일 경로(또는 이름). DB엔 파일명 문자열로만 들어갑니다.
            image_size: (width, height)
            kp_xy: (N,2) float32
            desc_u8: (128,N) uint8
        """
        # 1) DB, img open (없으면 생성)
        db = COLMAPDatabase.connect(self.database_path)
        db.create_tables()  # 테이블이 없으면 생성
        image = cv2.imread(image_path)

        try:
            # 2) 카메라 보장
            width, height = image.shape[1], image.shape[0]

            # 카메라가 없으면 새로 추가
            camera_id = self.ensure_camera(db, model="SIMPLE_RADIAL", width=width, height=height, params=None)

            # 3) 이미지 등록 (이미 있으면 그 id 재사용)
            row = db.execute("SELECT image_id FROM images WHERE name=?", (os.path.basename(image_path),)).fetchone()
            if row is not None:
                image_id = row[0]
            else:
                # 새 이미지 등록
                image_id = db.add_image(name=os.path.basename(image_path), camera_id=camera_id)

            # 4) 키포인트/디스크립터 저장
            db.execute("DELETE FROM keypoints WHERE image_id=?", (image_id,))
            db.execute("DELETE FROM descriptors WHERE image_id=?", (image_id,))

            # keypoints: (N,4) float32
            if kp_xy.dtype != np.float32:
                kp_xy = kp_xy.astype(np.float32)
            db.add_keypoints(image_id, kp_xy)

            # descriptors: (128,N) uint8
            if desc_u8.dtype != np.uint8:
                desc_u8 = desc_u8.astype(np.uint8)
            
            db.add_descriptors(image_id, desc_u8)

            db.commit()
        finally:
            db.close()

    def main(self,image_path):
        """
        Main function to run the feature extraction and storage process.
        """

        A = FeatureExtractor(image_path, database_path=self.database_path)

        out_dir = "filtered_img"

        output = A.img_filtering(out_dir)
        
        kp_xy, desc_u8, table , kept = A.extract_features(output,per_cell=50, rows=24, cols=32)
        
        print("kp:", kp_xy.shape, "desc:", desc_u8.shape)
        
        A.store_features(f"{out_dir}/filter_{self.jpg_name}.jpg", kp_xy, desc_u8)

        # --- table 시각화 및 저장 ---
        #heatmap_path = f"{out_dir}/countraw_{self.jpg_name}.png"
        #save_table_heatmap(table, heatmap_path)
        #keptmap_path = f"{out_dir}/kept_{self.jpg_name}.png"
        #save_table_heatmap(kept, keptmap_path)

# =================================================================================
# BFMatcher Implementation
# =================================================================================
class BFMatcher:
    def __init__(self, database_path, ratio=0.9, cross_check=True, max_iters: int = 5000,
                 ransac_th_px=1.0, ransac_conf=0.999, min_matches=8, 
                rng_seed: int | None = 0,
                save_two_view_geom: bool = True,):
        """
        Args:
            database_path: COLMAP database.db 경로
            ratio: Lowe ratio (0.75~0.85 권장)
            cross_check: 상호검증 사용 여부 
            min_matches: 이보다 적으면 해당 쌍은 저장하지 않음
            save_two_view_geom: two_view_geometries 테이블까지 기록 여부 (colmap과의 호환성)
        """
        self.database_path = database_path
        self.ratio = ratio # 두 최근접 매칭의 거리비율
        self.cross_check = cross_check
        self.min_matches = min_matches
        self.conf = ransac_conf
        self.th        = float(ransac_th_px)  # Sampson 거리 임계(픽셀)
        self.max_iters = int(max_iters)
        self.rng       = np.random.default_rng(rng_seed)
        self.save_two_view_geom = save_two_view_geom




    # ---------- 기하: 정규화 8-point + Sampson ----------    
    @staticmethod
    def _to_h(xy: np.ndarray) -> np.ndarray:
        '''
        입력 : xy (N,2)형태의 2D 점 좌표 
        출력 : (N,3) 형태의 동차좌표 배열
        '''
        return np.hstack([xy, np.ones((xy.shape[0],1), dtype=xy.dtype)])

    @staticmethod
    def _normalize_points(xy: np.ndarray):
        """Hartley 정규화: 평균 원점, 평균 거리 sqrt(2) => (xh_norm, T)"""
        m = xy.mean(axis=0)
        d = np.sqrt(((xy - m)**2).sum(axis=1)).mean() + 1e-12
        s = np.sqrt(2.0) / d
        T = np.array([[s, 0, -s*m[0]],
                      [0, s, -s*m[1]],
                      [0, 0, 1.0]], dtype=np.float64)
        xh = BFMatcher._to_h(xy).astype(np.float64)
        xhn = (T @ xh.T).T
        return xhn, T

    @staticmethod
    def _fit_F_eight_point(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """정규화 8-point (x1↔x2: 픽셀좌표, 동일 길이)"""
        x1n, T1 = BFMatcher._normalize_points(x1)
        x2n, T2 = BFMatcher._normalize_points(x2)
        X = x1n
        Xp = x2n  # x (first image), x' (second)
        # A f = 0 구성
        u, v, w  = X[:,0],  X[:,1],  X[:,2]
        up, vp, wp = Xp[:,0], Xp[:,1], Xp[:,2]
        A = np.stack([up*u, up*v, up*w,
                      vp*u, vp*v, vp*w,
                      wp*u, wp*v, wp*w], axis=1).astype(np.float64)  # (N,9)
        # SVD => f
        _, _, VT = np.linalg.svd(A)
        F = VT[-1,:].reshape(3,3)
        # rank-2 강제
        U,S,VT = np.linalg.svd(F)
        S[-1] = 0.0
        F = (U @ np.diag(S) @ VT)
        # denormalize
        F = (T2.T @ F @ T1)
        return F / (np.linalg.norm(F) + 1e-12)

    @staticmethod
    def _sampson_dist(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Sampson distance (픽셀^2 단위)"""
        x1h = BFMatcher._to_h(x1).astype(np.float64)   # (N,3)
        x2h = BFMatcher._to_h(x2).astype(np.float64)
        Fx1  = (F @ x1h.T).T                          # (N,3)
        Ftx2 = (F.T @ x2h.T).T
        e = np.sum(x2h * (F @ x1h.T).T, axis=1)       # x'^T F x
        denom = Fx1[:,0]**2 + Fx1[:,1]**2 + Ftx2[:,0]**2 + Ftx2[:,1]**2 + 1e-12
        return (e**2) / denom  
    
    def _load_descriptors(self, db: COLMAPDatabase, image_id: int) -> np.ndarray:
        """
        반환: (N,128) float32, L2 정규화, C-연속 메모리
        DB의 descriptors는 (128,N) uint8가 표준이지만, 혹시 뒤집혀 저장돼도 자동 보정.
        """
        r = db.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?",
                    (image_id,)).fetchone()
        if r is None:
            return np.zeros((0,128), np.float32)

        rows, cols = int(r[0]), int(r[1])
        buf = np.frombuffer(r[2], dtype=np.uint8)

        # 안전 리셰이프
        if rows * cols != buf.size:
            raise ValueError(f"[desc] corrupted blob: rows*cols != size ({rows}*{cols}!={buf.size})")

        # 표준은 (128,N). 열/행 어디에 128이 들어있든 (N,128)로 만들어줌.
        if rows == 128:
            desc_u8 = buf.reshape(128, cols)         # (128,N)
            desc = desc_u8.T.astype(np.float32)      # (N,128)
        elif cols == 128:
            desc_u8 = buf.reshape(rows, 128)         # (N,128)
            desc = desc_u8.astype(np.float32)        # (N,128)
        else:
            raise ValueError(f"[desc] neither dimension is 128 (rows,cols)=({rows},{cols})")

        # RootSIFT 전처리 스케일링(업캐스트 방지) + L2 정규화
        desc *= np.float32(1.0/512.0)
        desc /= (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12)

        # C-연속 & dtype 보장
        return np.ascontiguousarray(desc, dtype=np.float32)

    
    def _load_keypoints(self, db: COLMAPDatabase, image_id: int) -> np.ndarray:
        r = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
        if r is None: return np.zeros((0,2), np.float32)
        return blob_to_array(r[2], np.float32, (int(r[0]), int(r[1])))
    
    def _load_matches(self, db: COLMAPDatabase, id1: int, id2: int) -> np.ndarray:
        pair_id = image_ids_to_pair_id(id1, id2)
        r = db.execute("SELECT rows, cols, data FROM matches WHERE pair_id=?", (pair_id,)).fetchone()
        if r is None: return np.zeros((0,2), np.int32)
        mat = blob_to_array(r[2], np.uint32, (int(r[0]), int(r[1])))
        if mat.shape[1] == 2:
            m = mat.astype(np.int32)
        elif mat.shape[0] == 2:
            m = mat.T.astype(np.int32)
        else:
            m = mat.reshape(-1,2).astype(np.int32)
        return m

    def _build_image_pairs(self, db: COLMAPDatabase):
        """
        이미지 이름의 '가운데 숫자' 기준으로 허용된 조합만 매칭 페어 생성.
        허용 조합:
        - 02 ↔ 05
        - 04 ↔ 08
        - 05 ↔ 04 (== 04 ↔ 05)
        반환: [(id1, id2, name1, name2), ...]  (id1 < id2 보장)
        """
        rows = db.execute("SELECT image_id, name FROM images ORDER BY name").fetchall()
        if len(rows) < 2:
            raise ValueError("매칭할 이미지가 2개 이상 있어야 합니다.")

        # 이름에서 '두 번째 숫자 그룹'을 가운데 코드로 사용
        def mid_code(name: str) -> int | None:
            stem = os.path.splitext(name)[0]
            nums = re.findall(r"\d+", stem)   # 예) B1_02_00 -> ['1','02','00']
            return int(nums[1]) if len(nums) >= 2 else None

        # 중앙 코드별로 이미지들을 모음
        groups: dict[int, list[tuple[int,str]]] = defaultdict(list)
        for img_id, name in rows:
            code = mid_code(name)
            if code is None:
                continue  # 필요시 경고/예외로 바꿔도 됨
            groups[code].append((img_id, name))

        # 허용 조합(순서 무관하게 처리)
        allowed_pairs = {(2, 5), (4, 8), (4, 5)}
        allowed_pairs = {tuple(sorted(p)) for p in allowed_pairs}  # (5,4) == (4,5)

        pairs = []
        seen = set()

        for a, b in allowed_pairs:
            A = groups.get(a, [])
            B = groups.get(b, [])
            if not A or not B:
                continue
            for id1, name1 in A:
                for id2, name2 in B:
                    if id1 == id2:
                        continue
                    i, j = (id1, id2) if id1 < id2 else (id2, id1)
                    n1, n2 = (name1, name2) if id1 < id2 else (name2, name1)
                    key = (i, j)
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append((i, j, n1, n2))

        # 정렬: 페어를 안정적으로 반환
        pairs.sort(key=lambda x: (x[2], x[3]))
        return pairs

    def _bf_match(self, d0: np.ndarray, d1: np.ndarray):
        """OpenCV BFMatcher + Lowe ratio + cross-check
        Returns:
            matches_ij: (M,2) int32, conf: (M,) float32
        """
        if d0.shape[0] == 0 or d1.shape[0] == 0:
            return np.zeros((0,2), np.int32), np.zeros((0,), np.float32)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = bf.knnMatch(d0, d1, k=2)

        # 1) ratio test 통과 후보 (A→B). cand를 DMatch로 보관!
        cand = []
        for lst in knn:
            if len(lst) < 2:
                continue
            m, n = lst[0], lst[1]
            if m.distance < self.ratio * n.distance:
                cand.append(m)  # ← tuple이 아니라 DMatch를 그대로 저장

        # 2) cross-check: 역방향 ratio 후 reciprocal 확인
        if self.cross_check:
            knn_rev = bf.knnMatch(d1, d0, k=2)
            best_rev = {}
            for lst in knn_rev:
                if len(lst) < 2:
                    continue
                m, n = lst[0], lst[1]
                if m.distance < self.ratio * n.distance:
                    best_rev[m.queryIdx] = m.trainIdx
            dmatches = [m for m in cand if best_rev.get(m.trainIdx, -1) == m.queryIdx]
        else:
            dmatches = cand

        if not dmatches:
            return np.zeros((0,2), np.int32), np.zeros((0,), np.float32), []

        matches_ij = np.array([[m.queryIdx, m.trainIdx] for m in dmatches], dtype=np.int32)
        conf = 1.0 / (np.array([m.distance for m in dmatches], dtype=np.float32) + 1e-6)
        return matches_ij, conf, dmatches

    def _get_image_id_by_name(self, db: COLMAPDatabase, name: str) -> int:
        name = os.path.basename(name)
        name = "filter_" + name  # 전처리된 이미지 이름으로 변경
        row = db.execute("SELECT image_id FROM images WHERE name=?", (name,)).fetchone()
        if row is None:
            raise ValueError(f"images.name='{name}' 이(가) DB에 없습니다.")
        return int(row[0])


        # ---------- GMS 정제 ----------
    
    def _apply_gms(self, db: COLMAPDatabase, id1: int, id2: int,
               kpA_xy: np.ndarray, kpB_xy: np.ndarray,
               dmatches: list[cv2.DMatch]):
        """
        GMS 정제.
        - OpenCV 버전에 따라 입력 타입/반환 타입이 달라지므로 모두 안전하게 처리.
        - 실패/미지원이면 원본 매칭을 그대로 반환.
        """
        if not dmatches:
            return np.zeros((0,2), np.int32), []

        # 1) 이미지 크기 확보 (실패 시 좌표로 추정)
        try:
            w1, h1 = self._get_image_size(db, id1)
            w2, h2 = self._get_image_size(db, id2)
        except Exception:
            h1 = int(kpA_xy[:,1].max() + 1); w1 = int(kpA_xy[:,0].max() + 1)
            h2 = int(kpB_xy[:,1].max() + 1); w2 = int(kpB_xy[:,0].max() + 1)

        # 공용 실행 함수 (반환 타입 표준화)
        def _call_gms(pointsA, pointsB):
            ret = cv2.xfeatures2d.matchGMS(
                (w1, h1), (w2, h2),
                pointsA, pointsB, dmatches,
                withRotation=True, withScale=True, thresholdFactor=6
            )
            return ret

        keep_matches = dmatches
        try:
            # 2) 먼저 KeyPoint 리스트로 시도
            kpsA = [cv2.KeyPoint(float(x), float(y), 1.0) for (x, y) in kpA_xy]
            kpsB = [cv2.KeyPoint(float(x), float(y), 1.0) for (x, y) in kpB_xy]
            ret = _call_gms(kpsA, kpsB)

            # ---- 반환 타입 분기 ----
            if isinstance(ret, (list, tuple)) and len(ret) > 0 and isinstance(ret[0], cv2.DMatch):
                # (A) 이미 필터링된 DMatch 리스트를 반환하는 빌드
                keep_matches = list(ret)

            else:
                # (B) 마스크(0/1) 반환
                mask_np = np.asarray(ret).reshape(-1)
                if mask_np.size == len(dmatches):
                    keep_matches = [m for m, k in zip(dmatches, mask_np) if int(k) != 0]
                else:
                    # 3) 좌표 Nx2(float32) 방식으로 재시도
                    ptsA = kpA_xy.astype(np.float32)
                    ptsB = kpB_xy.astype(np.float32)
                    ret2 = _call_gms(ptsA, ptsB)

                    if isinstance(ret2, (list, tuple)) and len(ret2) > 0 and isinstance(ret2[0], cv2.DMatch):
                        keep_matches = list(ret2)
                    else:
                        mask_np2 = np.asarray(ret2).reshape(-1)
                        if mask_np2.size == len(dmatches):
                            keep_matches = [m for m, k in zip(dmatches, mask_np2) if int(k) != 0]
                        else:
                            print("[WARN] GMS returned mismatched size; using raw matches.")
                            keep_matches = dmatches

        except AttributeError as e:
            # xfeatures2d 자체가 없는 경우(=opencv-contrib 미설치)
            print(f"[WARN] GMS unavailable (xfeatures2d missing): {e}")
            keep_matches = dmatches
        except Exception as e:
            print(f"[WARN] GMS failed, skip. ({e})")
            keep_matches = dmatches

        matches_ij = np.array([[m.queryIdx, m.trainIdx] for m in keep_matches], dtype=np.int32)
        return matches_ij, keep_matches
    
    def _save_inlier_matches(self, db: COLMAPDatabase, id1: int, id2: int, inliers_ij: np.ndarray):
        pair_id = image_ids_to_pair_id(id1, id2)
        db.execute("DELETE FROM matches WHERE pair_id=?", (pair_id,))
        if inliers_ij.size == 0: return
        db.add_matches(id1, id2, inliers_ij.astype(np.uint32))
        print(f"[DB] matches({id1},{id2})에 {inliers_ij.shape[0]}개의 인라이어를 저장했습니다.")

    def _save_two_view_geometry(self, db: COLMAPDatabase, inlier_matches, id1: int, id2: int, F, E,H , qvec, tvec):
        """two_view_geometries 테이블에 F 저장"""
        pair_id = image_ids_to_pair_id(id1, id2)
        img1, img2 = pair_id_to_image_ids(pair_id)
        db.execute("DELETE FROM two_view_geometries WHERE pair_id=?", (pair_id,))
        if F is None:
            return

        db.add_two_view_geometry(img1,img2, inlier_matches, F,E,H , qvec, tvec, config=2)

        print(f"[DB] two_view_geometry({id1},{id2})에 F를 저장했습니다.")

    def _save_matches(self, db, id1, id2, matches_ij):
        """matches_ij: (N,2) int"""
        if matches_ij.size == 0:
            return
        mat = matches_ij.astype(np.uint32)  # (N,2)
        db.add_matches(id1, id2, mat)


    # ---------- USAC 계열 RANSAC: Coarse → Fine ----------
    def ransac(self, xA: np.ndarray, xB: np.ndarray):
        """
        USAC(MAGSAC++)로 F 추정, Coarse→Fine 정제.
        반환: F_best, inlier_mask(bool,N), inlier_count(int)
        """
        N = xA.shape[0]
        if N < 8: return None, np.zeros(N, bool), 0

        # USAC method 선택 
        USAC_MAGSAC = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        USAC_ACCURATE = getattr(cv2, "USAC_ACCURATE", USAC_MAGSAC)

        # 1) Coarse: 큰 임계치
        th_coarse = max(self.th*2.0, self.th + 0.5)
        F1, m1 = cv2.findFundamentalMat(
            xA, xB,
            method=USAC_MAGSAC,
            ransacReprojThreshold=float(th_coarse),
            confidence=float(self.conf),
            maxIters=int(self.max_iters)
        )
        if F1 is None or m1 is None:
            return None, np.zeros(N, bool), 0
        m1 = m1.ravel().astype(bool)
        if m1.sum() < 8:
            return None, np.zeros(N, bool), 0

        # 2) Fine: 타이트한 임계치로 인라이어 정제 (정확도 중시 모드)
        xA1, xB1 = xA[m1], xB[m1]
        F2, m2 = cv2.findFundamentalMat(
            xA1, xB1,
            method=USAC_ACCURATE,  # LO/퇴화검사 포함 모드(가능한 경우)
            ransacReprojThreshold=float(self.th),
            confidence=float(self.conf),
            maxIters=int(self.max_iters)
        )

        if F2 is None or m2 is None or m2.sum() < 8:
            # Fine에서 실패하면 Coarse 결과로 반환
            return F1, m1, int(m1.sum())

        m2 = m2.ravel().astype(bool)
        # 전체 길이의 마스크로 확장
        final_mask = np.zeros(N, dtype=bool)
        final_mask[np.where(m1)[0][m2]] = True
        return F2, final_mask, int(final_mask.sum())



    def match(self):
        """BF → (GMS) → USAC(Coarse→Fine) → 인라이어 저장 → 2-view geom 저장 """
        db = COLMAPDatabase.connect(self.database_path)
        db.create_tables()
        try:
            pairs = self._build_image_pairs(db)
            print(f"총 {len(pairs)} 이미지 쌍을 매칭합니다.")
            num_saved = 0

            for (id1, id2, name1, name2) in pairs:
                # 1) 디스크립터 로드 & BF 매칭 (ratio=0.9)
                d0 = self._load_descriptors(db, id1)
                d1 = self._load_descriptors(db, id2)
                matches_ij, _, dmatches = self._bf_match(d0, d1)
                M0 = matches_ij.shape[0]

                # 2) 키포인트 로드
                kpA = self._load_keypoints(db, id1)  # (NA,2)
                kpB = self._load_keypoints(db, id2)  # (NB,2)

                # 3)  GMS 정제
                gms_ij, gms_dmatches = self._apply_gms(db, id1, id2, kpA[:, :2], kpB[:, :2], dmatches)
                if gms_ij.size > 0:
                    matches_ij = gms_ij
                M = matches_ij.shape[0]
                if M < self.min_matches:
                    print(f"[SKIP] {name1} ↔ {name2}: after GMS, tentative={M} (<{self.min_matches})")
                    continue
                print(f"[OK] {name1} ↔ {name2}: tentative={M} ")

                self._save_matches(db, id1, id2,matches_ij)

                # 4) 좌표 뽑기
                xA = kpA[matches_ij[:,0], :2].astype(np.float64)
                xB = kpB[matches_ij[:,1], :2].astype(np.float64)

                # 5) USAC 계열 RANSAC (Coarse→Fine)
                F_best, inl_mask, inl_cnt = self.ransac(xA, xB)
                if inl_cnt < 8 or F_best is None:
                    print(f"[FAIL] {name1} ↔ {name2}: USAC inliers={inl_cnt} (<8)")
                    continue

                inlier_matches = matches_ij[inl_mask]
                self._save_inlier_matches(db, id1, id2, inlier_matches)
                print(f"[OK] {name1} ↔ {name2}: tentative={M} | inliers={inl_cnt} ({inl_cnt/max(M,1)*100:.1f}%)")
                num_saved += 1

                # 6) two_view_geometries 저장 (E/H/pose)
                if self.save_two_view_geom:
                    try:
                        def raw_name(n: str) -> str:
                            b = os.path.basename(n)
                            return b[7:] if b.startswith("filter_") else b

                        xA_inl = xA[inl_mask]
                        xB_inl = xB[inl_mask]

                        # Essential
                        K = estimate_K(raw_name(name1))
                        E = compute_E(F_best, K)

                        # Homography (USAC로)
                        USAC_MAGSAC = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
                        H, _ = cv2.findHomography(
                            xA_inl, xB_inl,
                            method=USAC_MAGSAC,
                            ransacReprojThreshold=float(self.th),
                            confidence=float(self.conf),
                            maxIters=int(self.max_iters)
                        )
                        if H is not None and abs(H[2,2]) > 1e-12:
                            H = H / H[2,2]

                        # Pose
                        _, R, t, _ = cv2.recoverPose(E, xA_inl, xB_inl, K)
                        q = Rscipy.from_matrix(R).as_quat()  # [qx,qy,qz,qw]
                        qvec = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)  # [qw,qx,qy,qz]
                        tvec = t.reshape(3) / (np.linalg.norm(t) + 1e-12)

                        self._save_two_view_geometry(db, inlier_matches, id1, id2, F_best, E, H, qvec, tvec)

                    except Exception as e:
                        print(f"[WARN] two_view_geometry 저장 실패: {name1}↔{name2} | {e}")

            db.commit()
            print(f"저장 완료: {num_saved} 쌍.")
        finally:
            db.close()
    
    def diag_match(self):
        db = COLMAPDatabase.connect(self.database_path)
        df = diagnose_pairs_from_db(
                            db_conn=db,
                            get_pairs_fn=self._build_image_pairs,      # (db)->[(id1,id2,name1,name2), ...]
                            load_kp_fn=self._load_keypoints,           # (db,id)->(N,4) or (N,2)
                            load_matches_fn=self._load_matches,        # (db,id1,id2)->(M,2)
                            K_from_name_fn=estimate_K2,                 # (name)->(3,3)
                            save_csv_path="pair_diagnostics.csv",      # 저장 경로
                            r_plane_thresh=1.8,                        # H/F 비율 임계
                            tri_med_thresh_deg=1.0,                    # 각도(°) 임계
                            thF=1.5, thH=2.0, conf=0.999, iters=10000  # USAC 파라미터
                        )
        return df


    def run_triangulation_ba_and_export(self,
                                        image_path: str,
                                        output_path: str,
                                        export_ply_path: str | None = None,
                                        refine_intrinsics: bool = False):
        """
        pycolmap를 사용해 DB의 매칭을 기반으로:
        1) Incremental SfM (포즈 추정/등록 포함)
        2) 삼각측량 (3D 포인트 생성)
        3) BA (bundle adjustment)
        4) PLY 내보내기
        를 한 번에 수행합니다.

        Args:
            image_path: DB(images.name)와 파일명이 일치하는 이미지 디렉토리
            output_path: COLMAP sparse 모델이 저장될 폴더 (cameras/images/points3D 생성)
            export_ply_path: 결과 sparse 포인트클라우드를 PLY로 저장할 경로(미지정시 output_path/sparse_points.ply)
            refine_intrinsics: BA에서 K(초점 등)까지 같이 미세조정할지 (기본 False)
        """
        import os
        os.makedirs(output_path, exist_ok=True)

        # 1) 파이프라인 옵션 설정 (필요시 여기서 추가 튜닝)
        opts = pycolmap.IncrementalPipelineOptions()
        # BA에서 내장 파라미터를 건드릴지 선택
        opts.ba_refine_focal_length   = bool(refine_intrinsics)
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params    = False
        # (원하면) 매칭 최소치/검증 등 옵션도 조정 가능
        # opts.min_num_matches = 15
        # opts.verify_images = True

        # 2) Incremental SfM: (등록→삼각측량→BA) 자동 수행
        maps = pycolmap.incremental_mapping(
            database_path=self.database_path,
            image_path=image_path,
            output_path=output_path,
            options=opts,
        )

        if not maps:
            raise RuntimeError("[pycolmap] 재구성이 생성되지 않았습니다. DB의 features/matches를 확인하세요.")

        # 여러 sub-model 중 첫 번째를 사용 (필요시 병합/선택 로직 추가)
        rec = next(iter(maps.values()))
        rec.write(output_path)  # cameras.bin / images.bin / points3D.bin 저장

        # 3) 색상 입히기(선택) + PLY 내보내기
        try:
            rec.extract_colors_for_all_images(image_path)
        except Exception as e:
            print(f"[WARN] 색 추출 실패(컬러 없는 그레이 사진일 수 있음): {e}")

        ply_path = export_ply_path or os.path.join(output_path, "sparse_points.ply")
        rec.export_PLY(ply_path)
        print(f"[PLY] 저장: {ply_path}")

        # 간단 요약 출력
        print(f"[SfM] 등록 이미지수={rec.num_reg_images()}, 3D 포인트수={rec.num_points3D()}")
        return ply_path


if __name__ == "__main__":

    database_path = "db_case3.db"

    img_case3 = [
        "images/B1/02/B1_02_00.jpg",
        "images/B1/02/B1_02_01.jpg",
        "images/B1/02/B1_02_02.jpg",
        "images/B1/04/B1_04_00.jpg",
        "images/B1/04/B1_04_01.jpg",
        "images/B1/04/B1_04_02.jpg",
        "images/B1/05/B1_05_00.jpg",
        "images/B1/05/B1_05_01.jpg",
        "images/B1/05/B1_05_02.jpg",
        "images/B1/08/B1_08_00.jpg",
        "images/B1/08/B1_08_01.jpg",
        "images/B1/08/B1_08_02.jpg",
        ]
    
    
    #for img_path in img_case3:
    #    A = FeatureExtractor(img_path ,database_path)
    #    A.main(img_path)
    #    print(f"Feature extraction and storage completed for {os.path.basename(img_path)}") 

    
    matcher = BFMatcher(database_path, ratio=0.9, cross_check=True, max_iters= 5000,
                ransac_th_px=1.0, ransac_conf=0.999, min_matches=8, save_two_view_geom=True)
    # matcher.match()

    matcher.diag_match() # 평면 지배 여부 판단 

    # 2) pycolmap으로 SfM + BA -> PLY 형태로 저장 
    ply_path = matcher.run_triangulation_ba_and_export(
        image_path="filtered_img",           # DB의 images.name과 파일명이 꼭 일치해야 함
        output_path="sparse_out",      # cameras/images/points3D 저장 폴더
        export_ply_path=None,                  # None이면 sparse_out/sparse_points.ply
        refine_intrinsics=False                # K 고정(권장). 필요시 True로 미세조정
    )

