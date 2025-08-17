import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from ultralytics import YOLO
import tempfile
import os
import time
from collections import defaultdict, Counter
import base64

def grab_img_b64(pic_path):
    try:
        with open(pic_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    except:
        return ""

st.set_page_config(
    page_title="Advanced Productivity Monitor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

#================================
class mohsenPoseChecker:
    def __init__(self):
        try:
            if os.path.exists('models/4-aiolearn.pt'):
                self.mdl = YOLO('models/4-aiolearn.pt')
                
                if getattr(self.mdl, 'names', None) and self.mdl.names:
                    self.cls_names = self.mdl.names
                else:
                    self.cls_names = {}
                    for idx in range(100):
                        self.cls_names[idx] = f"Class_{idx}"
            else:
                st.error("Object Detection Model (best.pt) not found in models folder!")
                raise FileNotFoundError("4-aiolearn.pt not found")
            
            if os.path.exists('models/yolov8n-pose.pt'):
                self.poseMdl = YOLO('models/yolov8n-pose.pt')
                self.use_pose = True
            else:
                self.use_pose = False
            
            self.movHist = defaultdict(list)
            self.posHist = defaultdict(list)
            self.hipMovHist = defaultdict(list)
            self.frame_cnt = 0
            self.person_cnt = 0
            self.id_map = {}
            
            self.track_conf = {
                'max_distance': 200,
                'max_frame_gap': 5,
                'min_detection_confidence': 0.5,
                'area_similarity_weight': 100,
                'frame_penalty_weight': 10
            }
            
            try:
                import torch
                if torch.cuda.is_available():
                    self.mdl.to('cuda')
                    if self.use_pose:
                        self.poseMdl.to('cuda')
                else:
                    self.mdl.to('cpu')
                    if self.use_pose:
                        self.poseMdl.to('cpu')
            except:
                self.mdl.to('cpu')
                if self.use_pose:
                    self.poseMdl.to('cpu')
                
        except Exception as err:
            st.error(f"Oops, something went wrong loading models: {err}")
            raise err
    
    def get_eng_lbl(self, cls_id, walking=False):
        cls_name = self.cls_names.get(cls_id, str(cls_id))
        
        if walking:
            return "Leaving Work"
        
        nm_lower = cls_name.lower()
        if nm_lower in ['sitting', 'sit down']:
            return "Useful Work"
        elif nm_lower in ['standing']:
            return "Non-Productive"
        elif nm_lower in ['walking', 'walking_on_stairs']:
            return "Leaving Work"
        elif nm_lower in ['fall-detected', 'fall_down', 'falling', 'nearly_fall']:
            return "Emergency"
        elif nm_lower in ['lying_down', 'crawling']:
            return "Danger"
        elif nm_lower in ['drinking']:
            return "Transition"
        else:
            return cls_name
    
    def calc_iou(self, b1, b2):
        x1_min, y1_min, x1_max, y1_max = b1
        x2_min, y2_min, x2_max, y2_max = b2
        
        interXmin = max(x1_min, x2_min)
        interYmin = max(y1_min, y2_min)
        interXmax = min(x1_max, x2_max)
        interYmax = min(y1_max, y2_max)
        
        if interXmax <= interXmin or interYmax <= interYmin:
            return 0.0
        
        inter_area = (interXmax - interXmin) * (interYmax - interYmin)
        
        b1_area = (x1_max - x1_min) * (y1_max - y1_min)
        b2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = b1_area + b2_area - inter_area
        
        if union_area > 0:
            return inter_area / union_area
        else:
            return 0.0
    
    def extract_skel_features(self, keypts):
        if not keypts or len(keypts) < 17:
            return None
        
        feats = {}
        
        nose = keypts[0]
        l_shoulder = keypts[5]
        r_shoulder = keypts[6]
        l_hip = keypts[11]
        r_hip = keypts[12]
        l_knee = keypts[13]
        r_knee = keypts[14]
        l_ankle = keypts[15]
        r_ankle = keypts[16]
        
        vis_pts = []
        for pt in [nose, l_shoulder, r_shoulder, l_hip, r_hip]:
            if pt[0] > 0 and pt[1] > 0:
                vis_pts.append(pt)
        
        if len(vis_pts) < 4:
            return None
        
        try:
            if l_shoulder[0] > 0 and r_shoulder[0] > 0:
                sh_width = abs(l_shoulder[0] - r_shoulder[0])
                feats['shoulder_width'] = sh_width
            
            if nose[1] > 0 and l_hip[1] > 0 and r_hip[1] > 0:
                hip_ctr_y = (l_hip[1] + r_hip[1]) / 2
                torso_len = abs(nose[1] - hip_ctr_y)
                feats['torso_length'] = torso_len
            
            if l_hip[0] > 0 and r_hip[0] > 0:
                hip_w = abs(l_hip[0] - r_hip[0])
                feats['hip_width'] = hip_w
            
            if 'shoulder_width' in feats and 'torso_length' in feats:
                if feats['torso_length'] > 0:
                    feats['shoulder_torso_ratio'] = feats['shoulder_width'] / feats['torso_length']
            
            if l_hip[1] > 0 and l_ankle[1] > 0:
                l_leg_len = abs(l_hip[1] - l_ankle[1])
                feats['leg_length'] = l_leg_len
            
            return feats
            
        except:
            return None
    
    def compare_skel_feats(self, f1, f2):
        if not f1 or not f2:
            return 0.0
        
        common_feats = set(f1.keys()) & set(f2.keys())
        if len(common_feats) < 2:
            return 0.0
        
        sims = []
        
        for feat in common_feats:
            v1 = f1[feat]
            v2 = f2[feat]
            
            if v1 == 0 or v2 == 0:
                continue
            
            diff = abs(v1 - v2) / max(v1, v2)
            sim = max(0, 1 - diff)
            sims.append(sim)
        
        if sims:
            return sum(sims) / len(sims)
        else:
            return 0.0
    
    def assignPersonID(self, bbox, frm_idx, cur_keypts=None):
        ctr_x = (bbox[0] + bbox[2]) / 2
        ctr_y = (bbox[1] + bbox[3]) / 2
        bbox_ar = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        best_match = None
        best_dist = float('inf')
        
        for prsn_id, data in self.id_map.items():
            last_pos = data['last_position']
            last_frm = data['last_frame']
            
            if frm_idx - last_frm > 20:
                continue
            
            dist = ((ctr_x - last_pos[0])**2 + (ctr_y - last_pos[1])**2)**0.5
            
            if dist < 150:
                if dist < best_dist:
                    best_dist = dist
                    best_match = prsn_id
        
        if best_match:
            det_cnt = self.id_map[best_match].get('detection_count', 0)
            self.id_map[best_match].update({
                'last_position': [ctr_x, ctr_y],
                'last_frame': frm_idx,
                'last_area': bbox_ar,
                'detection_count': det_cnt + 1
            })
            return best_match
        
        self.person_cnt += 1
        new_id = f"Employee_{self.person_cnt}"
        
        self.id_map[new_id] = {
            'last_position': [ctr_x, ctr_y],
            'last_frame': frm_idx,
            'last_area': bbox_ar,
            'detection_count': 1,
            'first_seen': frm_idx
        }
        
        return new_id
    
    def detectOverlapBoxes(self, frm_dets):
        overlaps = []
        
        for i, d1 in enumerate(frm_dets):
            for j, d2 in enumerate(frm_dets[i+1:], i+1):
                iou = self.calc_iou(d1['bbox'], d2['bbox'])
                
                if iou > 0.2:
                    c1 = [(d1['bbox'][0] + d1['bbox'][2])/2, (d1['bbox'][1] + d1['bbox'][3])/2]
                    c2 = [(d2['bbox'][0] + d2['bbox'][2])/2, (d2['bbox'][1] + d2['bbox'][3])/2]
                    
                    ctr_dist = ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5
                    
                    ar1 = (d1['bbox'][2] - d1['bbox'][0]) * (d1['bbox'][3] - d1['bbox'][1])
                    ar2 = (d2['bbox'][2] - d2['bbox'][0]) * (d2['bbox'][3] - d2['bbox'][1])
                    ar_ratio = min(ar1, ar2) / max(ar1, ar2)
                    
                    overlaps.append({
                        'detection1': i,
                        'detection2': j,
                        'iou': iou,
                        'id1': d1.get('id'),
                        'id2': d2.get('id'),
                        'center_distance': ctr_dist,
                        'area_ratio': ar_ratio,
                        'confidence1': d1.get('confidence', 0),
                        'confidence2': d2.get('confidence', 0)
                    })
        
        return overlaps
    
    def resolveIDConflicts(self, overlaps, frm_dets):
        for ovrlp in overlaps:
            idx1 = ovrlp['detection1']
            idx2 = ovrlp['detection2']
            
            d1 = frm_dets[idx1]
            d2 = frm_dets[idx2]
            
            if ovrlp['iou'] > 0.7:
                if d1.get('confidence', 0) > d2.get('confidence', 0):
                    frm_dets[idx2]['id'] = None
                else:
                    frm_dets[idx1]['id'] = None
                    
            elif ovrlp['iou'] > 0.4:
                sc1 = self.calcDetScore(d1, ovrlp)
                sc2 = self.calcDetScore(d2, ovrlp)
                
                if sc1 > sc2:
                    frm_dets[idx2]['id'] = None
                else:
                    frm_dets[idx1]['id'] = None
                    
            else:
                if self.samePerson(d1, d2):
                    if self.betterSkel(d1, d2):
                        frm_dets[idx2]['id'] = None
                    else:
                        frm_dets[idx1]['id'] = None
        
        out = []
        for det in frm_dets:
            if det.get('id') is not None:
                out.append(det)
        return out
    
    def calcDetScore(self, det, ovrlp):
        score = 0.0
        
        conf = det.get('confidence', 0)
        score += conf * 0.4
        
        if 'skeleton_features' in det:
            skel_feats = det['skeleton_features']
            if skel_feats:
                vis_pts = 0
                for feat in skel_feats.values():
                    if feat > 0:
                        vis_pts += 1
                skel_score = min(1.0, vis_pts / 5)
                score += skel_score * 0.3
        
        ar_ratio = ovrlp.get('area_ratio', 1.0)
        score += ar_ratio * 0.2
        
        ctr_dist = ovrlp.get('center_distance', 0)
        dist_score = max(0, 1 - (ctr_dist / 200))
        score += dist_score * 0.1
        
        return score
    
    def samePerson(self, d1, d2):
        if d1.get('id') == d2.get('id'):
            return True
        
        if 'skeleton_features' in d1 and 'skeleton_features' in d2:
            sk1 = d1['skeleton_features']
            sk2 = d2['skeleton_features']
            if sk1 and sk2:
                similarity = self.compare_skel_feats(sk1, sk2)
                return similarity > 0.6
        
        return False
    
    def betterSkel(self, d1, d2):
        sc1 = 0
        sc2 = 0
        
        if 'skeleton_features' in d1:
            sk1 = d1['skeleton_features']
            if sk1:
                for f in sk1.values():
                    if f > 0:
                        sc1 += 1
        
        if 'skeleton_features' in d2:
            sk2 = d2['skeleton_features']
            if sk2:
                for f in sk2.values():
                    if f > 0:
                        sc2 += 1
        
        return sc1 > sc2
    
    def cleanupOldTracks(self, cur_frm_idx, max_gone=30):
        cur_time = cur_frm_idx
        to_remove = []
        
        for prsn_id, data in self.id_map.items():
            frms_since_last = cur_time - data['last_frame']
            if frms_since_last > max_gone:
                to_remove.append(prsn_id)
        
        for prsn_id in to_remove:
            del self.id_map[prsn_id]
            if prsn_id in self.posHist:
                del self.posHist[prsn_id]
            if prsn_id in self.hipMovHist:
                del self.hipMovHist[prsn_id]
    
    def getPoseKpts(self, frame, bbox):
        if not self.use_pose:
            return None
        
        try:
            x1, y1, x2, y2 = [int(c) for c in bbox]
            bbox_h = y2 - y1
            bbox_w = x2 - x1
            pad = int(min(bbox_h, bbox_w) * 0.3)
            pad = max(30, min(pad, 100))
            
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)
            
            prsn_crop = frame[y1:y2, x1:x2]
            if prsn_crop.size == 0:
                return None
            
            pose_res = self.poseMdl(prsn_crop, verbose=False)
            
            for r in pose_res:
                kpts = r.keypoints
                if kpts is not None:
                    kpts_xy = kpts.xy[0].cpu().numpy()
                    
                    if len(kpts_xy) >= 17:
                        adj_kpts = []
                        for kpt in kpts_xy:
                            if kpt[0] > 0 and kpt[1] > 0:
                                adj_kpts.append([kpt[0] + x1, kpt[1] + y1])
                            else:
                                adj_kpts.append([0, 0])
                        return adj_kpts
            
            return None
            
        except Exception as e:
            return None
    
    def detectWalk(self, det, fr_w, fr_h, frame):
        prsn_id = det['id']
        bbox = det['bbox']
        
        ctr_x = (bbox[0] + bbox[2]) / 2
        ctr_y = (bbox[1] + bbox[3]) / 2
        cur_pos = [ctr_x, ctr_y]
        
        if prsn_id not in self.posHist:
            self.posHist[prsn_id] = []
        
        self.posHist[prsn_id].append(cur_pos)
        
        if len(self.posHist[prsn_id]) > 10:
            self.posHist[prsn_id] = self.posHist[prsn_id][-10:]
        
        if len(self.posHist[prsn_id]) < 5:
            return False
        
        recent_pos = self.posHist[prsn_id][-5:]
        
        tot_dist = 0
        for i in range(1, len(recent_pos)):
            dx = recent_pos[i][0] - recent_pos[i-1][0]
            dy = recent_pos[i][1] - recent_pos[i-1][1]
            dist = (dx**2 + dy**2)**0.5
            tot_dist += dist
        
        avg_mov = tot_dist / 4
        
        movs = []
        for i in range(1, len(recent_pos)):
            dx = recent_pos[i][0] - recent_pos[i-1][0]
            dy = recent_pos[i][1] - recent_pos[i-1][1]
            movs.append([dx, dy])
        
        if len(movs) > 1:
            dx_vals = [m[0] for m in movs]
            dy_vals = [m[1] for m in movs]
            dx_var = np.var(dx_vals)
            dy_var = np.var(dy_vals)
            mov_var = dx_var + dy_var
        else:
            mov_var = 0
        
        fr_ctr_x = fr_w / 2
        hor_mov = abs(ctr_x - fr_ctr_x)
        
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        if bbox_h > 0:
            asp_ratio = bbox_w / bbox_h
        else:
            asp_ratio = 1
        
        hip_mov_score = 0
        if self.use_pose:
            hip_ctr = self.getPoseKpts(frame, bbox)
            if hip_ctr and len(hip_ctr) >= 17:
                l_hip = hip_ctr[11]
                r_hip = hip_ctr[12]
                
                if l_hip[0] > 0 and l_hip[1] > 0 and r_hip[0] > 0 and r_hip[1] > 0:
                    if prsn_id not in self.hipMovHist:
                        self.hipMovHist[prsn_id] = []
                    
                    hip_ctr_pos = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
                    self.hipMovHist[prsn_id].append(hip_ctr_pos)
                    
                    if len(self.hipMovHist[prsn_id]) > 5:
                        self.hipMovHist[prsn_id] = self.hipMovHist[prsn_id][-5:]
                    
                    if len(self.hipMovHist[prsn_id]) >= 3:
                        hip_movs = []
                        for i in range(1, len(self.hipMovHist[prsn_id])):
                            dx = self.hipMovHist[prsn_id][i][0] - self.hipMovHist[prsn_id][i-1][0]
                            dy = self.hipMovHist[prsn_id][i][1] - self.hipMovHist[prsn_id][i-1][1]
                            hip_movs.append((dx**2 + dy**2)**0.5)
                        
                        if len(hip_movs) > 1:
                            hip_mov_var = np.var(hip_movs)
                        else:
                            hip_mov_var = 0
                        hip_mov_score = min(3, int(hip_mov_var / 10))
        
        knee_mov = mov_var > 50
        ankle_mov = mov_var > 30
        
        is_walk = False
        
        if avg_mov > 15 and (knee_mov or ankle_mov):
            is_walk = True
        
        if not is_walk and hip_mov_score >= 2 and avg_mov > 3:
            is_walk = True
        
        return is_walk
    
    def procFrame(self, frame, frm_idx, fps):
        fr_w = frame.shape[1]
        fr_h = frame.shape[0]
        
        results = self.mdl(frame, verbose=False)
        
        frm_data = {
            'frame_idx': frm_idx,
            'timestamp': frm_idx / fps,
            'detections': []
        }
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    if conf > 0.5:
                        bbox = [x1, y1, x2, y2]
                        
                        pose_kpts = self.getPoseKpts(frame, bbox)
                        
                        prsn_id = self.assignPersonID(bbox, frm_idx, pose_kpts)
                        
                        det_data = {
                            'bbox': bbox,
                            'confidence': conf,
                            'class_id': cls_id,
                            'id': prsn_id
                        }
                        
                        if pose_kpts:
                            det_data['pose_keypoints'] = pose_kpts
                            skel_feats = self.extract_skel_features(pose_kpts)
                            if skel_feats:
                                det_data['skeleton_features'] = skel_feats
                        
                        cls_nm = self.cls_names.get(cls_id, str(cls_id))
                        if cls_nm.lower() in ['standing', 'sitting', 'walking']:
                            is_walk = self.detectWalk(det_data, fr_w, fr_h, frame)
                            if is_walk:
                                det_data['is_walking'] = True
                                det_data['original_class'] = cls_id
                        
                        det_data['english_label'] = self.get_eng_lbl(
                            det_data['class_id'], 
                            is_walk if 'is_walk' in locals() else False
                        )
                        
                        frm_data['detections'].append(det_data)
        
        ovrlps = self.detectOverlapBoxes(frm_data['detections'])
        if ovrlps:
            frm_data['detections'] = self.resolveIDConflicts(ovrlps, frm_data['detections'])
        
        if self.frame_cnt % 10 == 0:
            self.cleanupOldTracks(frm_idx)
        
        self.frame_cnt += 1
        return frm_data
    
    def drawSkel(self, frame, keypts, color=(0, 255, 0)):
        connections = [
            (0, 1), (0, 2),
            (5, 6),
            (5, 7), (7, 9),
            (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15),
            (12, 14), (14, 16)
        ]
        
        for conn in connections:
            pt1_idx, pt2_idx = conn
            if pt1_idx < len(keypts) and pt2_idx < len(keypts):
                pt1 = keypts[pt1_idx]
                pt2 = keypts[pt2_idx]
                
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)
        
        for i, kpt in enumerate(keypts):
            if kpt[0] > 0 and kpt[1] > 0:
                if i in [9, 10]:
                    cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 5, (0, 255, 255), -1)
                elif i in [11, 12]:
                    cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 5, (255, 0, 0), -1)
                else:
                    cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, color, -1)
    
    def drawAnnots(self, frame, frm_data):
        ann_frame = frame.copy()
        
        colors = {
            'standing': (72, 61, 139),
            'sitting': (160, 82, 45),
            'walking': (72, 61, 139),
            'fall-detected': (255, 0, 0),
            'fall_down': (255, 0, 0),
            'falling': (255, 0, 0),
            'nearly_fall': (255, 165, 0),
            'lying_down': (128, 0, 128),
            'crawling': (128, 0, 128),
            'drinking': (0, 255, 255)
        }
        
        for det in frm_data['detections']:
            x1, y1, x2, y2 = det['bbox']
            cls_id = det['class_id']
            conf = det['confidence']
            prsn_id = det['id']
            eng_lbl = det.get('english_label', 'Unknown')
            
            cls_nm = self.cls_names.get(cls_id, str(cls_id))
            color = colors.get(cls_nm.lower(), (0, 255, 0))
            
            cv2.rectangle(ann_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            label = f"{prsn_id}: {eng_lbl} ({conf:.2f})"
            
            if 'original_class' in det:
                orig_nm = self.cls_names.get(det['original_class'], str(det['original_class']))
                label += f" [was {orig_nm}]"
            
            lbl_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(ann_frame, 
                         (int(x1), int(y1) - lbl_size[1] - 10), 
                         (int(x1) + lbl_size[0], int(y1)), 
                         color, -1)
            
            cv2.putText(ann_frame, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if 'pose_keypoints' in det and cls_nm.lower() in ['standing', 'walking']:
                self.drawSkel(ann_frame, det['pose_keypoints'], color)
        
        return ann_frame

#*****************************
class maryamAnalyzer:
    def __init__(self):
        self.res_data = []
        
    def analyzeVideoLive(self, vid_path):
        detector = mohsenPoseChecker()
        cap = cv2.VideoCapture(vid_path)
        
        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if orig_fps <= 0 or orig_fps > 60:
            fps = 25
            st.warning(f"Something looks off with FPS ({orig_fps}), using 25 FPS instead")
        else:
            fps = int(orig_fps)
        
        orig_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        orig_codec = "".join([chr((orig_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        st.info(f"📹 Input Video: {width}x{height} @ {fps}fps | Codec: {orig_codec}")
        
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1
        
        st.info(f"🔧 Adjusted dimensions: {width}x{height} (even numbers for codec compatibility)")
        
        os.makedirs('video_temp', exist_ok=True)
        
        import time
        timestamp = int(time.time())
        
        output_path = f'video_temp/processed_video_{timestamp}.mp4'
        
        codecs_to_try = []
        
        st.info(f"🔍 Detected codec: {orig_codec} - Selecting best output format...")
        
        if orig_codec.lower() in ['h264', 'avc1', 'x264']:
            codecs_to_try.extend([
                ('mp4v', '.mp4'),
                ('MJPG', '.mp4'),
                ('XVID', '.avi'),
                ('mp4v', '.avi'),
            ])
        elif orig_codec in ['mp4v', 'MP4V']:
            codecs_to_try.append(('mp4v', '.mp4'))
        elif orig_codec in ['XVID', 'xvid']:
            codecs_to_try.append(('XVID', '.avi'))
        
        codecs_to_try.extend([
            ('mp4v', '.mp4'),
            ('MJPG', '.mp4'),
            ('MJPG', '.avi'),
            ('XVID', '.avi'),
        ])
        
        out = None
        successful_codec = None
        
        for codec, ext in codecs_to_try:
            try:
                test_out_path = f'video_temp/processed_video_{timestamp}{ext}'
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_out = cv2.VideoWriter(test_out_path, fourcc, max(fps, 1), (width, height))
                
                st.info(f"Trying {codec} codec in {ext} container...")
                
                if test_out.isOpened():
                    out = test_out
                    output_path = test_out_path
                    successful_codec = codec
                    st.success(f"✅ Using {codec} codec in {ext} container")
                    break
                else:
                    st.warning(f"Hmm, {codec} didn't work")
                    test_out.release()
                    
            except Exception as e:
                st.warning(f"Oops! {codec} had issues: {str(e)}")
                continue
        
        if out is None or not out.isOpened():
            st.error("We tried everything but couldn't set up video writer...")
            raise Exception("Video writer setup failed")
        
        all_frm_data = []
        
        prog_bar = st.progress(0)
        status_txt = st.empty()
        live_disp = st.empty()
        
        use_gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                use_gpu = True
                st.info("🚀 GPU detected! Enhanced live processing enabled")
            else:
                st.info("💻 CPU processing mode - Optimized display")
        except:
            st.info("💻 CPU processing mode - Optimized display")
        
        if use_gpu:
            frames_to_show = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            show_after_percentage = 25
            frame_display_interval = 3
        else:
            frames_to_show = [0, 1, 2, 3, 4, 9]
            show_after_percentage = 50
            frame_display_interval = 10
        
        frm_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frm_data = detector.procFrame(frame, frm_idx, fps)
            ann_frame = detector.drawAnnots(frame, frm_data)
            
            all_frm_data.append(frm_data)
            out.write(ann_frame)
            
            progress = (frm_idx + 1) / tot_frames
            prog_bar.progress(progress)
            
            should_show = False
            if frm_idx in frames_to_show:
                should_show = True
            elif progress >= show_after_percentage / 100:
                if frm_idx % frame_display_interval == 0:
                    should_show = True
            
            if should_show:
                frame_rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
                live_disp.image(frame_rgb, caption=f"Processing Frame {frm_idx + 1}/{tot_frames}", use_column_width=True)
            
            det_cnt = len(frm_data['detections'])
            proc_mode = "🚀 GPU" if use_gpu else "💻 CPU"
            status_txt.text(f'{proc_mode} | Frame {frm_idx + 1}/{tot_frames} | Detections: {det_cnt} | Progress: {progress*100:.1f}%')
            
            frm_idx += 1
        
        cap.release()
        out.release()
        status_txt.text('✅ Analysis completed!')
        
        final_out_path = self.convertVidForBrowser(output_path, detector)
        
        return all_frm_data, fps, final_out_path, detector.cls_names, successful_codec
    
    def genReport(self, frm_data_list, fps):
        cls_stats = defaultdict(int)
        prod_stats = defaultdict(int)
        prsn_prod = defaultdict(lambda: defaultdict(int))
        tot_dets = 0
        
        if not frm_data_list:
            st.warning("Looks like there's no frame data for analysis")
            return {
                'class_stats': {},
                'productivity_stats': {},
                'person_productivity': {},
                'total_frames': 0,
                'fps': fps if fps > 0 else 1,
                'duration': 0,
                'total_detections': 0
            }
        
        for frm_data in frm_data_list:
            for det in frm_data['detections']:
                cls_id = det['class_id']
                prsn_id = det['id']
                prod_lbl = det.get('english_label', 'Unknown')
                
                cls_stats[cls_id] += 1
                prod_stats[prod_lbl] += 1
                prsn_prod[prsn_id][prod_lbl] += 1
                tot_dets += 1
        
        tot_frames = len(frm_data_list)
        safe_fps = max(fps, 1)
        if tot_frames > 0:
            duration = tot_frames / safe_fps
        else:
            duration = 0
        
        report = {
            'class_stats': dict(cls_stats),
            'productivity_stats': dict(prod_stats),
            'person_productivity': dict(prsn_prod),
            'total_frames': tot_frames,
            'fps': safe_fps,
            'duration': duration,
            'total_detections': tot_dets
        }
        
        return report

    def convertVidForBrowser(self, inp_path, detector):
        try:
            import subprocess
            import os
            
            out_dir = os.path.dirname(inp_path)
            filename = os.path.splitext(os.path.basename(inp_path))[0]
            out_path = os.path.join(out_dir, f"{filename}_browser.mp4")
            
            cmd = [
                'ffmpeg', '-i', inp_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                '-y',
                out_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(out_path):
                st.success(f"Nice! Video converted for browser playback")
                return out_path
            else:
                st.warning(f"FFmpeg didn't work out, using original video")
                return inp_path
                
        except Exception as e:
            st.warning(f"Video conversion had some issues: {str(e)}")
            return inp_path

def makeCharts(report, cls_names):
    if not report['class_stats']:
        st.warning("No detections to show")
        return
    
    st.subheader("📊 Productivity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Productivity by Category")
        prod_data = []
        for cat, cnt in report['productivity_stats'].items():
            prod_data.append({'Category': cat, 'Count': cnt})
        
        prod_df = pd.DataFrame(prod_data)
        
        if not prod_df.empty:
            fig_pie = px.pie(prod_df, values='Count', names='Category',
                            title='Productivity Distribution',
                            color_discrete_map={
                                'Useful Work': '#00FF00',
                                'Non-Productive': '#FFA500',
                                'Leaving Work': '#FF0000',
                                'Emergency': '#FF00FF',
                                'Danger': '#800080',
                                'Transition': '#FFFF00'
                            })
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Activity by Class")
        cls_data = []
        for cls_id, cnt in report['class_stats'].items():
            cls_data.append({
                'Class': cls_names.get(cls_id, f"Class {cls_id}"),
                'Count': cnt
            })
        
        cls_df = pd.DataFrame(cls_data)
        
        if not cls_df.empty:
            fig_bar = px.bar(cls_df, x='Class', y='Count', title='Detection Count by Class')
            st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    st.subheader("👥 Individual Person Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prsn_prod_data = []
        for prsn_id, activities in report['person_productivity'].items():
            tot_acts = sum(activities.values())
            useful_work = activities.get('Useful Work', 0)
            if tot_acts > 0:
                prod_pct = (useful_work / max(tot_acts, 1) * 100)
            else:
                prod_pct = 0
            
            prsn_prod_data.append({
                'Person': prsn_id,
                'Productivity %': prod_pct,
                'Total Activities': tot_acts
            })
        
        if prsn_prod_data:
            prsn_df = pd.DataFrame(prsn_prod_data)
            fig_prsn = px.bar(prsn_df, x='Person', y='Productivity %', 
                               title='Productivity Percentage by Person',
                               color='Productivity %',
                               color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_prsn, use_container_width=True)
    
    with col2:
        if prsn_prod_data:
            prsn_df = pd.DataFrame(prsn_prod_data)
            st.dataframe(prsn_df, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📈 Summary Statistics")
    
    tot_frames = report['total_frames']
    tot_dets = report['total_detections']
    duration = report['duration']
    
    useful_work_frames = report['productivity_stats'].get('Useful Work', 0)
    non_prod_frames = report['productivity_stats'].get('Non-Productive', 0)
    leaving_work_frames = report['productivity_stats'].get('Leaving Work', 0)
    emergency_frames = report['productivity_stats'].get('Emergency', 0)
    danger_frames = report['productivity_stats'].get('Danger', 0)
    transition_frames = report['productivity_stats'].get('Transition', 0)
    
    if tot_frames > 0:
        overall_prod = (useful_work_frames / max(tot_frames, 1) * 100)
    else:
        overall_prod = 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Frames", tot_frames)
        st.metric("Total Detections", tot_dets)
        st.metric("Duration", f"{duration:.1f}s")
    
    with col2:
        st.metric("Useful Work", useful_work_frames)
        st.metric("Non-Productive", non_prod_frames)
        st.metric("Leaving Work", leaving_work_frames)
    
    with col3:
        st.metric("Emergency", emergency_frames)
        st.metric("Danger", danger_frames)
        st.metric("Transition", transition_frames)
    
    st.markdown("---")
    st.subheader("📊 Overall Productivity Score")
    
    summary_data = {
        'Metric': ['Useful Work', 'Non-Productive', 'Leaving Work', 'Emergency', 'Danger', 'Transition', 'Overall Productivity'],
        'Value': [
            useful_work_frames,
            non_prod_frames,
            leaving_work_frames,
            emergency_frames,
            danger_frames,
            transition_frames,
            f"{overall_prod:.1f}%"
        ]
    }
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

def main():
    
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; margin-bottom: 30px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
        <img src="data:image/png;base64,{}" width="120" style="margin-bottom: 15px;">
        <h1 style="color: white; margin: 0; font-size: 2.5em;">🎯 Employee Monitoring System</h1>
        <p style="color: #f0f0f0; margin: 10px 0 0 0; font-size: 1.2em; font-style: italic;">
            Advanced AI-powered workplace productivity analysis
        </p>
    </div>
    """.format(grab_img_b64("img/usa.jpg")), unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background: rgba(255, 255, 255, 0.95); border-radius: 20px; padding: 30px; 
                box-shadow: 0 10px 40px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.2);">
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <img src="data:image/png;base64,{}" width="120" style="border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
        </div>
        """.format(grab_img_b64("img/1.png")), unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📹 Video Upload")
        
        st.info("""
        **Formats:** MP4, AVI, MOV, MKV
        
        **Best:** 30-60s, clear view, good lighting
        """)
        
        uploaded_file = st.file_uploader(
            "Choose video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload any video format for productivity analysis"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ {uploaded_file.name} uploaded!")
            
            st.session_state.analysis_started = True
            st.session_state.uploaded_file = uploaded_file
    
    if 'analysis_started' in st.session_state and st.session_state.analysis_started:
        uploaded_file = st.session_state.uploaded_file
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            vid_path = tmp_file.name
        
        st.subheader("📹 Video Player")
        st.video(vid_path)
        
        st.markdown("---")
        
        st.subheader("🎯 Start Analysis")
        if st.button("🎯 Start Analysis", type="primary", use_container_width=True, key="start_analysis_right"):
            st.session_state.analysis_started_right = True
            st.rerun()
        
        st.markdown("---")
        
        gpu_info = "💻 CPU Mode"
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_info = f"🚀 GPU Mode ({gpu_name})"
        except:
            pass
        
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.9); border-radius: 10px; padding: 15px; 
                    border-left: 4px solid #667eea; margin: 15px 0;">
            <h4 style="color: #333; margin: 0 0 10px 0;">🤖 Model Information</h4>
            <p style="color: #666; margin: 5px 0; font-size: 0.9em;">
                <strong>Object Detection:</strong> best.pt (Trained on 10,000+ images)
            </p>
            <p style="color: #666; margin: 5px 0; font-size: 0.9em;">
                <strong>Pose Detection:</strong> YOLOv8n-pose (Lightweight for weak hardware)
            </p>
            <p style="color: #666; margin: 5px 0; font-size: 0.9em;">
                <strong>Hardware:</strong> {gpu_info}
            </p>
            <p style="color: #666; margin: 5px 0; font-size: 0.9em;">
                <strong>Live Processing:</strong> {"Enhanced (GPU)" if "GPU" in gpu_info else "Optimized (CPU)"}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'analysis_started_right' in st.session_state and st.session_state.analysis_started_right:
            try:
                analyzer = maryamAnalyzer()
                frm_data_list, fps, out_vid_path, cls_names, used_codec = analyzer.analyzeVideoLive(vid_path)
                report = analyzer.genReport(frm_data_list, fps)
                
                st.success(f"✅ Analysis completed! ({len(frm_data_list)} frames)")
                
                st.markdown("---")
                st.header("📹 Processed Video")
                
                if os.path.exists(out_vid_path) and os.path.getsize(out_vid_path) > 0:
                    st.success(f"✅ Video processed successfully using {used_codec if used_codec else 'unknown'} codec")
                    st.video(out_vid_path)
                    
                    file_size = os.path.getsize(out_vid_path) / (1024 * 1024)
                    st.info(f"""
                    **📁 Video File Info:**
                    - **Path:** `{out_vid_path}`
                    - **Size:** {file_size:.1f} MB
                    - **Format:** {out_vid_path.split('.')[-1].upper()}
                    """)
                else:
                    st.error("Something's not right with the video file...")
                    st.warning("Maybe try uploading again?")
                
                st.markdown("---")
                st.header("🎯 Activity Classification System")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 5 Main Activity Categories")
                    st.markdown("""
                    **🟢 Useful Work (Class 3)**
                    - Sitting and productive work
                    - Production activities
                    - Desk work
                    
                    **🟠 Non-Productive (Class 4)**
                    - Standing without work
                    - Waiting and stopping
                    - Non-productive activity
                    
                    **🔴 Leaving Work (Class 5)**
                    - Walking
                    - Leaving workplace
                    - Movement
                    """)
                
                with col2:
                    st.markdown("""
                    **🟣 Emergency (Class 0)**
                    - Fall detection
                    - Emergency situation
                    - Need for help
                    
                    **🟡 Transition (Class 1)**
                    - Sitting down
                    - Changing position
                    - Transition between states
                    
                    **🟤 Danger (Class 2)**
                    - Falling
                    - Dangerous situation
                    - Need for attention
                    """)
                
                st.markdown("---")
                st.header("🤖 Model Information")
                st.info("""
                **Model:** `best.pt` (Custom Trained)
                **Training Data:** 10,000+ images
                **Classes:** 15 different human activities
                **Accuracy:** High precision for workplace monitoring
                **Features:** Object detection + Pose estimation
                """)
                
                st.markdown("---")
                st.header("📊 Productivity Results")
                
                makeCharts(report, cls_names)
                
                st.markdown("---")
                st.subheader("💾 Save Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Save Report as CSV", type="secondary", use_container_width=True):
                        rep_data = []
                        for prsn_id, acts in report['person_productivity'].items():
                            for act, cnt in acts.items():
                                rep_data.append({
                                    'Person_ID': prsn_id,
                                    'Activity': act,
                                    'Count': cnt
                                })
                        
                        for metric, val in report['productivity_stats'].items():
                            rep_data.append({
                                'Person_ID': 'SUMMARY',
                                'Activity': metric,
                                'Count': val
                            })
                        
                        df = pd.DataFrame(rep_data)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download CSV Report",
                            data=csv,
                            file_name=f"productivity_report_{int(time.time())}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    st.info(f"""
                    **Video saved to:** `{out_vid_path}`
                    
                    **Report contains:**
                    - Individual person activities
                    - Productivity statistics
                    - Summary metrics
                    """)
                
                                                               
                    
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
            finally:
                if os.path.exists(vid_path):
                    os.unlink(vid_path)
                
                del st.session_state.analysis_started_right
    
    else:
        pass
        
        st.markdown("---")
        st.subheader("🎯 Advanced Productivity Monitoring Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔍 Smart Detection**
            - Object + Pose detection
            - Unique person tracking
            - Movement analysis
            - Hip movement detection
            """)
        
        with col2:
            st.markdown("""
            **📊 Productivity Analysis**
            - Useful Work detection
            - Non-productive activity
            - Individual person stats
            - Real-time monitoring
            """)
        
        with col3:
            st.markdown("""
            **🎯 Walking Detection**
            - Advanced movement tracking
            - Hip movement analysis
            - Position history
            - Accurate classification
            """)
        
        st.markdown("---")
        st.subheader("📈 What You'll Get")
        
        st.markdown("""
        - **Individual Person Tracking**: Each person gets a unique ID
        - **Productivity Categories**: Useful Work vs Non-Productive activities
        - **Walking Detection**: Advanced analysis combining object detection and pose detection
        - **Detailed Reports**: Separate statistics for each activity type
        - **Real-time Analysis**: Live video processing with progress tracking
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 30px; margin-top: 50px;">
        <div style="display: inline-block; padding: 20px; border-radius: 15px; 
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1); background: rgba(255,255,255,0.9);">
            <img src="data:image/png;base64,{}" style="border-radius: 10px; max-width: 100%;">
        </div>
    </div>
    """.format(grab_img_b64("img/3.png")), unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()