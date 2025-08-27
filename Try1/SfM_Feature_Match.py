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
        1) grayscale
        2) CLAHE 
        """
        h, w = self.image.shape[:2] 

        print(f"origin completed {self.jpg_name} Image size: {h}x{w}")  # image size and grid size

        # 1) 저장 디렉토리 준비
        os.makedirs(out_dir, exist_ok=True)


        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # 노랑 범위(필요시 조정)
        lower = np.array([15,  60,  60])   # H,S,V
        upper = np.array([40, 255, 255])
        mask  = cv2.inRange(hsv, lower, upper)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), 1)

        # 회색 영상 + 마스크 강조
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        boost = cv2.normalize(gray + (mask//2), None, 0, 255, cv2.NORM_MINMAX)
        
        # 3) CLAHE
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
        output = clahe.apply(boost)

        cv2.imwrite(f"{out_dir}/filter_{self.jpg_name}.jpg", output)

        return  output  

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
        # 예: 목표 총량 cap = rows*cols*per_cell_hi*1.5 (*1.5: 여유있게 설정)
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
        heatmap_path = f"{out_dir}/countraw_{self.jpg_name}.png"
        save_table_heatmap(table, heatmap_path)
        keptmap_path = f"{out_dir}/kept_{self.jpg_name}.png"
        save_table_heatmap(kept, keptmap_path)

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
    
    def _load_descriptors(self, db: COLMAPDatabase, image_id: int):
        """
        DB에서 keypoints 로드
        descriptors: (N,128) float32 (L2 정규화)
        """
        r = db.execute("SELECT rows, cols, data FROM descriptors WHERE image_id=?", (image_id,)).fetchone()
        if r is None:
            return np.zeros((0,2), np.float32), np.zeros((0,128), np.float32)

        desc_u8 = blob_to_array(r[2], np.uint8, (r[0], r[1]))    # (128, N)
        desc = desc_u8.T.astype(np.float32) / 512.0              # (N,128)
        desc /= (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12)
        return  desc.astype(np.float32)

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

        cand = []
        for m, n in knn:
            if m.distance < self.ratio * n.distance:
                cand.append((m.queryIdx, m.trainIdx, m.distance))

        if self.cross_check: # 교차 검증
            bf2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            knn_rev = bf2.knnMatch(d1, d0, k=2)
            best_rev = {}
            for m, n in knn_rev:
                if m.distance < self.ratio * n.distance:
                    best_rev[m.queryIdx] = m.trainIdx

            keep = []
            conf = []
            for qi, tj, dist in cand:
                if best_rev.get(tj, -1) == qi:
                    keep.append([qi, tj])
                    conf.append(1.0 / (dist + 1e-6))  # 간단한 신뢰도
            matches_ij = np.array(keep, dtype=np.int32)
            conf = np.array(conf, dtype=np.float32)
        else:
            matches_ij = np.array([[qi, tj] for qi, tj, _ in cand], dtype=np.int32)
            conf = np.ones((matches_ij.shape[0],), dtype=np.float32)

        return matches_ij, conf

    def _save_matches(self, db, id1, id2, matches_ij):
        """matches_ij: (N,2) int"""
        if matches_ij.size == 0:
            return
        mat = matches_ij.astype(np.uint32)  # (N,2)
        db.add_matches(id1, id2, mat)


    # ---------- DB I/O ----------
    def _get_image_id_by_name(self, db: COLMAPDatabase, name: str) -> int:
        name = os.path.basename(name)
        name = "filter_" + name  # 전처리된 이미지 이름으로 변경
        row = db.execute("SELECT image_id FROM images WHERE name=?", (name,)).fetchone()
        if row is None:
            raise ValueError(f"images.name='{name}' 이(가) DB에 없습니다.")
        return int(row[0])

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

    # ---------- RANSAC  ----------
    def ransac(self, xA, xB):
        """
        입력: xA, xB  (N,2)  매칭된 2D 좌표들 (픽셀)
        출력: F_best(3x3), inlier_mask(bool, N), best_cnt(int)
        """
        N = xA.shape[0] # A와 B 사이의 매칭 갯수
        if N < 8:
            raise ValueError(f"RANSAC 최소 8쌍 필요, 현재 {N}쌍.")

        # RANSAC 루프
        best_inl = np.zeros(N, dtype=bool)
        best_F   = None
        best_cnt = 0
        max_iters = self.max_iters
        EPS = 1e-12
        i = 0

        while i < max_iters:
            # [1] 무작위 8쌍
            idx = self.rng.choice(N, size=8, replace=False)
            F_candidate = self._fit_F_eight_point(xA[idx], xB[idx])

            # [2] Sampson 거리
            d = self._sampson_dist(F_candidate, xA, xB)
            inl = d <= (self.th**2)
            cnt = int(inl.sum())

            if cnt > best_cnt:
                best_cnt = cnt
                best_inl = inl.copy()
                best_F   = F_candidate

                # cnt: 이번 모델의 인라이어 개수, N: 전체 매칭 수
                w = max(min(cnt / N, 1.0 - EPS), EPS)   # (0,1) 범위로 안전 클램프
                s = 8                                    # 8-point 최소 샘플 크기
                denom = log(max(1.0 - (w ** s), EPS))   # < 0

                # 필요한 반복 수 N_need = ln(1-conf) / ln(1 - w^s)
                need  = int(ceil(log(1.0 - self.conf) / (denom - EPS)))

                # 현재 상한보다 작으면 상한을 줄여서 조기종료를 유도
                max_iters = min(max_iters, need)
            i += 1

        # [3] 베스트 인라이어로 F 재추정
        F_best = self._fit_F_eight_point(xA[best_inl], xB[best_inl]) if best_cnt >= 8 else best_F

        return F_best, best_inl, best_cnt
            
    def match(self):
        """DB 내 모든 이미지쌍 매칭, matches(및 선택적으로 two_view_geometries) 테이블에 저장"""
        db = COLMAPDatabase.connect(self.database_path)
        db.create_tables()  # 스키마 보장

        try:
            pairs = self._build_image_pairs(db)
            print(f"총 {len(pairs)} 이미지 쌍을 매칭합니다.")
            num_saved = 0

            for (id1, id2, name1, name2) in pairs:
                # 1) 디스크립터 로드 & BF + ratio(+mutual) 매칭
                d0 = self._load_descriptors(db, id1)
                d1 = self._load_descriptors(db, id2)

                matches_ij, conf = self._bf_match(d0, d1)
                M = matches_ij.shape[0]
                if M < self.min_matches:
                    print(f"[SKIP] {name1} ↔ {name2}: matches={matches_ij.shape[0]} (<{self.min_matches})")
                    continue

                # 2) 키포인트 로드 → 좌표 추출
                kpA = self._load_keypoints(db, id1)   # (NA,2)
                kpB = self._load_keypoints(db, id2)   # (NB,2)
                xA  = kpA[matches_ij[:,0], :2].astype(np.float64)
                xB  = kpB[matches_ij[:,1], :2].astype(np.float64)

                # 3) RANSAC(F) → 인라이어만 남기기
                F_best, inl_mask, inl_cnt = self.ransac(xA, xB)
                if inl_cnt < 8 or F_best is None:
                    print(f"[FAIL] {name1} ↔ {name2}: RANSAC inliers={inl_cnt} (<8)")
                    continue

                inlier_matches = matches_ij[inl_mask]
                self._save_inlier_matches(db, id1, id2, inlier_matches)
                print(f"[OK] {name1} ↔ {name2}: tentative={M} | inliers={inl_cnt} ({inl_cnt/M*100:.1f}%)")
                num_saved += 1
                
                # 4) two_view_geometries 저장
                if self.save_two_view_geom:
                    try:
                        # 이름에서 'filter_' 접두어를 떼서 내부파라미터 함수가 기대하는 형태로 맞춤
                        def raw_name(n: str) -> str:
                            b = os.path.basename(n)
                            return b[7:] if b.startswith("filter_") else b

                        # E, H, pose
                        K = estimate_K(raw_name(name1))                 # 사용자 환경 함수
                        E = compute_E(F_best, K)                        # 사용자 환경 함수

                        xA_inl = xA[inl_mask]
                        xB_inl = xB[inl_mask]
                        H, _ = cv2.findHomography(xA_inl, xB_inl, method=cv2.RANSAC,
                                                ransacReprojThreshold=self.th)
                        if H is not None and abs(H[2,2]) > 1e-12:
                            H = H / H[2,2]

                        _, R, t, _ = cv2.recoverPose(E, xA_inl, xB_inl, K)
                        q = Rscipy.from_matrix(R).as_quat()             # [qx,qy,qz,qw]
                        qvec = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
                        tvec = t.reshape(3) / (np.linalg.norm(t) + 1e-12)

                        self._save_two_view_geometry(db, inlier_matches, id1, id2,
                                                    F_best, E, H, qvec, tvec)
                    except Exception as e:
                        # 내부파라미터 함수가 아직 준비 안 됐거나 실패한 경우에도 전체 파이프라인은 진행
                        print(f"[WARN] two_view_geometry 저장 실패: {name1}↔{name2} | {e}")

            db.commit()

            print(f"저장 완료: {num_saved} 쌍.")

        finally:
            db.close()
                

    # 삼각측량 (Triangulation))
    def triangulate_points(self, nameA, nameB):
        
        db = COLMAPDatabase.connect(self.database_path)
        
        idA = self._get_image_id_by_name(db, nameA)
        idB = self._get_image_id_by_name(db, nameB)

        kpA = self._load_keypoints(db, idA)   # (NA,2) : A의 모든 키포인트
        kpB = self._load_keypoints(db, idB)   # (NB,2) : B의 모든 키포인트
        matches = self._load_matches(db, idA, idB)  # (N,2) : A와 B 사이의 매칭정보
        xA = kpA[matches[:,0]].astype(np.float64)  # (N,2)
        xB = kpB[matches[:,1]].astype(np.float64)

        # E를 DB에서 불러오기 
        def load_E(db: COLMAPDatabase, img_name1: str, img_name2: str):
            # 이미지 이름에서 image_id 추출
            row1 = db.execute("SELECT image_id, camera_id FROM images WHERE name=?", (img_name1,)).fetchone()
            row2 = db.execute("SELECT image_id, camera_id FROM images WHERE name=?", (img_name2,)).fetchone()
            if row1 is None or row2 is None:
                raise ValueError("이미지 이름이 DB에 없습니다.")
            id1, cam_id1 = row1
            id2, cam_id2 = row2

            # pair_id 계산
            pair_id = image_ids_to_pair_id(id1, id2)

            # two_view_geometries에서 E 추출
            row = db.execute("SELECT E FROM two_view_geometries WHERE pair_id=?", (pair_id,)).fetchone()
            if row is None:
                raise ValueError("two_view_geometries에 해당 쌍이 없습니다.")
            E = blob_to_array(row[0], np.float64, (3, 3))

            return E
        
        E = load_E(db, nameA, nameB)

        # K 계산하기
        K = estimate_K(nameA) # 카메라 내부 파라미터 계산


        # 정규화 좌표로 변환 
        pts1_u = cv2.undistortPoints(xA.reshape(-1,1,2), K, None).reshape(-1,2)
        pts2_u = cv2.undistortPoints(xB.reshape(-1,1,2), K, None).reshape(-1,2)
        
        # P1,P2 계산
        _, R, t, mask = cv2.recoverPose(E, xA, xB, K)
        P1 = np.hstack([np.eye(3), np.zeros((3,1))])
        P2 = np.hstack([R, t.reshape(3,1)])
        X_h = cv2.triangulatePoints(P1,P2,pts1_u.T,pts2_u.T)
        X = (X_h[:3] / (X_h[3] + 1e-12)).T

        return X
    
    def bA(self,):
        return NotImplementedError
    





# =================================================================================
class choose_state(FeatureExtractor,BFMatcher):
    """
    매칭 방법을 선택하는 클래스입니다.
    현재는 BFMatcher만 구현되어 있지만, 추후 SuperGlue 등 다른 매처를 추가할 수 있습니다.
    """
    def __init__(self, state="feature", choose_matcher="BFMatcher", database_path="database.db", ratio=0.8, cross_check=True, min_matches=5, save_two_view_geom=False):
        self.matcher = BFMatcher(database_path, ratio, cross_check, min_matches, save_two_view_geom)
        self.superglue = None  # 추후 SuperGlue 매처를 추가할 수 있음
        self.state = state
        
    def choose_state(self):
        if self.state == "feature":
            print("Feature extraction state selected.")
            return FeatureExtractor
        elif self.state == "matcher":
            print("Matcher state selected.")
            return BFMatcher
        else:
            raise ValueError(f"Unknown state: {self.state}")
        
    def match(self):
        if self.choose == "SuperGlue":
            print("Using SuperGlue for matching.")
            self.superglue.match()
        elif self.choose == "BFMatcher":
            print("Using BFMatcher for matching.")
            self.matcher.match()





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
        #A = FeatureExtractor(img_path ,database_path)
        #A.main(img_path)
        #print(f"Feature extraction and storage completed for {os.path.basename(img_path)}") 

    
    matcher = BFMatcher(database_path, ratio=0.9, cross_check=True, max_iters= 5000,
                ransac_th_px=1.0, ransac_conf=0.999, min_matches=8, save_two_view_geom=True)
    matcher.match()