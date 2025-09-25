import os
import sys
import signal
import argparse

import numpy as np      #  <-- MOVE UP

import cv2
import torch
import clip
import joblib
import json
import copy
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

'''
OpenBLAS bugs may require you to run in terminal:

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

'''

from PyQt5.QtWidgets import (
    QApplication, QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QLineEdit, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

LOG_FILENAME = "skip_log.json"
MIN_CLASS_COUNT = 2   # Only do class-wide suggestions if >=2 bboxes for that YOLO class
MAX_IMG_DIM = 650    # We limit displayed images to 650×650 in both mismatch & class remap preview

# Distinct pastel background colors for class buttons:
PASTEL_COLORS = [
    "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",
    "#e0bbe4", "#f8d5a3", "#ffc8dd", "#bde0fe", "#c8f5f6",
    "#ffd6ff", "#f2f0a1", "#f1dca7", "#ffcca5", "#e6ccb2",
    "#c5dedd", "#b5ead7", "#f8d5a3", "#ecc8af", "#f1dca7",
]

###############################################################################
# 1) Arg Parsing
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO bounding-box checks with CLIP. Full-screen class remap, scaled bounding boxes up to 1024×1024, forced/auto class remap, skip-log, undo, pastel UI."
    )
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="./my_logreg_model.pkl")
    parser.add_argument("--labelmap_path", type=str, default="./my_label_list.pkl")
    parser.add_argument("--corrected_labels_path", type=str, default="./corrected_labels")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--clip_auto", type=float, default=0.0,
                        help="If CLIP conf >= this => auto fix mismatch.")
    parser.add_argument("--class_remap_ratio", type=float, default=0.8,
                        help="If >= ratio => auto-suggest class remap for that YOLO-labeled class.")
    return parser.parse_args()

###############################################################################
# 2) YOLO I/O
###############################################################################
def load_labelmap(labelmap_path):
    label_list = joblib.load(labelmap_path)
    label_to_id = {lbl: i for i, lbl in enumerate(label_list)}
    return label_list, label_to_id

def load_yolo_file(txt_path):
    if not os.path.isfile(txt_path):
        return []
    lines = open(txt_path,"r").read().strip().splitlines()
    records = []
    for line in lines:
        parts= line.split()
        if len(parts) < 5:
            continue
        cid= int(parts[0])
        x= float(parts[1])
        y= float(parts[2])
        w= float(parts[3])
        h= float(parts[4])
        records.append((cid,x,y,w,h))
    return records

def save_yolo_file(txt_path, records):
    if not records:
        with open(txt_path,"w") as f: pass
        return
    with open(txt_path,"w") as f:
        for (cid,x,y,w,h) in records:
            f.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

###############################################################################
# 3) Skip Log
###############################################################################
def load_skip_log(skip_log_path):
    if not os.path.isfile(skip_log_path):
        return {}
    return json.load(open(skip_log_path,"r"))

def save_skip_log(skip_log_path, skip_data):
    with open(skip_log_path,"w") as f:
        json.dump(skip_data, f, indent=2)

###############################################################################
# 4) BBox Signature
###############################################################################
def make_bbox_signature(x_center,y_center,w_norm,h_norm, precision=5):
    xc = round(x_center, precision)
    yc = round(y_center, precision)
    wn = round(w_norm, precision)
    hn = round(h_norm, precision)
    return f"x{xc}_y{yc}_w{wn}_h{hn}"

###############################################################################
# 5) CLIP
###############################################################################
def clip_predict_label_and_conf(pil_img, clip_model, preprocess, clf, label_list, device):
    inp= preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats= clip_model.encode_image(inp)
    feats= feats / feats.norm(dim=-1, keepdim=True)
    feats_np= feats.squeeze(0).cpu().numpy().reshape(1, -1)

    proba= clf.predict_proba(feats_np)
    best_idx= np.argmax(proba, axis=1)[0]
    confidence= proba[0, best_idx]

    pred_class= clf.classes_[best_idx]
    if isinstance(pred_class,str):
        predicted_label= pred_class
    else:
        predicted_label= label_list[pred_class]
    return predicted_label, float(confidence)

def crop_pil_image(full_img, x_min, y_min, x_max, y_max):
    w_img,h_img= full_img.size
    x_min_c= max(0, min(x_min,w_img))
    x_max_c= max(0, min(x_max,w_img))
    y_min_c= max(0, min(y_min,h_img))
    y_max_c= max(0, min(y_max,h_img))
    if x_max_c<= x_min_c or y_max_c<= y_min_c:
        return Image.new("RGB",(10,10),(128,128,128))
    return full_img.crop((x_min_c,y_min_c,x_max_c,y_max_c))

###############################################################################
# 6) StateHistory => for Undo
###############################################################################
class StateHistory:
    def __init__(self):
        self.stack= []
    def push(self, updated_records, skip_data):
        import copy
        snap_rec= copy.deepcopy(updated_records)
        snap_skip= copy.deepcopy(skip_data)
        self.stack.append( (snap_rec, snap_skip) )
    def can_undo(self):
        return len(self.stack)>0
    def pop(self):
        if self.can_undo():
            return self.stack.pop()
        return None, None

###############################################################################
# 7) limit_image_size => returns (resized_img, scale_factor)
###############################################################################
def limit_image_size(img_bgr):
    """
    Scale down if w>MAX_IMG_DIM or h>MAX_IMG_DIM. Return (resized_img, scale).
    scale=1.0 if no resizing needed, else <1.
    """
    h,w= img_bgr.shape[:2]
    if w>MAX_IMG_DIM or h>MAX_IMG_DIM:
        scale= min(MAX_IMG_DIM/w, MAX_IMG_DIM/h)
        new_w= int(w*scale)
        new_h= int(h*scale)
        out= cv2.resize(img_bgr,(new_w,new_h), interpolation=cv2.INTER_AREA)
        return out, scale
    else:
        return img_bgr,1.0

###############################################################################
# 8) ClassRemapPreviewDialog => now full screen + scaled bounding boxes
###############################################################################
class ClassRemapPreviewDialog(QDialog):
    """
    user_choice => "remap" or "skip"
    chosen_new_cid => label_list index
    """
    def __init__(
        self,
        base_img_path,
        old_name,
        new_name,
        ratio,
        box_indices,
        yolo_records,
        label_list,
        pil_img,
        parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("Class Remap Preview")
        # Dark grey gradient
        self.setStyleSheet("""
        QDialog {
          background: QLinearGradient(spread:pad, x1:0, y1:0, x2:1, y2:1, 
            stop:0 #2E2E2E, stop:1 #1C1C1C);
        }
        """)

        # Force full screen for the class remap preview as requested
        self.showFullScreen()

        self.user_choice= None
        self.chosen_new_cid= None

        self.base_img_path= base_img_path
        self.old_name= old_name
        self.new_name= new_name
        self.ratio= ratio
        self.box_indices= box_indices
        self.yolo_records= yolo_records
        self.label_list= label_list
        self.pil_img= pil_img

        self.init_ui()

    def init_ui(self):
        layout= QVBoxLayout()
        self.setLayout(layout)

        msg_text= (f"{self.ratio*100:.1f}% of '{self.old_name}' => '{self.new_name}'\n"
                   f"Remap all '{self.old_name}' -> ?")
        lbl_top= QLabel(msg_text)
        lbl_top.setAlignment(Qt.AlignCenter)
        lbl_top.setWordWrap(True)
        layout.addWidget(lbl_top)

        pm= self.draw_orange()
        lbl_img= QLabel()
        lbl_img.setPixmap(pm)
        layout.addWidget(lbl_img)

        combo_layout= QHBoxLayout()
        lbl_c= QLabel("Select new class:")
        combo_layout.addWidget(lbl_c)

        self.combo= QComboBox()
        for c in self.label_list:
            self.combo.addItem(c)
        if self.new_name in self.label_list:
            self.combo.setCurrentIndex(self.label_list.index(self.new_name))
        combo_layout.addWidget(self.combo)
        layout.addLayout(combo_layout)

        btn_layout= QHBoxLayout()

        b_ok= QPushButton("Confirm")
        b_ok.setStyleSheet("QPushButton { background-color: #FFB3BA; color: #000000; font-weight: bold; }")
        b_skip= QPushButton("Skip")
        b_skip.setStyleSheet("QPushButton { background-color: #BAE1FF; color: #000000; font-weight: bold; }")

        b_ok.clicked.connect(self.on_confirm)
        b_skip.clicked.connect(self.on_skip)
        btn_layout.addWidget(b_ok)
        btn_layout.addWidget(b_skip)
        layout.addLayout(btn_layout)

    def draw_orange(self):
        if not os.path.exists(self.base_img_path):
            pm= QPixmap(200,200)
            pm.fill(Qt.gray)
            return pm
        img_bgr= cv2.imread(self.base_img_path)
        if img_bgr is None:
            pm= QPixmap(200,200)
            pm.fill(Qt.darkGray)
            return pm

        (img_bgr, scale)= limit_image_size(img_bgr)

        if self.old_name in self.label_list:
            old_cid= self.label_list.index(self.old_name)
        else:
            old_cid= -1

        w_img,h_img= self.pil_img.size
        color= (0,165,255)
        thick=3
        for i in self.box_indices:
            rec= self.yolo_records[i]
            if rec is None:
                continue
            (cid,x_c,y_c,w_n,h_n)= rec
            if cid== old_cid:
                x_min= (x_c-0.5*w_n)* w_img
                y_min= (y_c-0.5*h_n)* h_img
                x1= int(x_min)
                y1= int(y_min)
                x2= x1+ int(w_n*w_img)
                y2= y1+ int(h_n*h_img)
                # scale them
                x1= int(x1*scale)
                y1= int(y1*scale)
                x2= int(x2*scale)
                y2= int(y2*scale)
                cv2.rectangle(img_bgr,(x1,y1),(x2,y2), color, thick)

        rgb= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qi= QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                   rgb.shape[1]*3, QImage.Format_RGB888)
        pm= QPixmap.fromImage(qi)
        return pm

    def on_confirm(self):
        self.user_choice= "remap"
        sel_str= self.combo.currentText()
        if sel_str in self.label_list:
            self.chosen_new_cid= self.label_list.index(sel_str)
        else:
            self.chosen_new_cid= None
        self.close()
    def on_skip(self):
        self.user_choice= "skip"
        self.close()

###############################################################################
# 9) MismatchDialog
###############################################################################
class MismatchDialog(QDialog):
    def __init__(
        self,
        base_img_path,
        pil_crop,
        mismatch_index,
        total_mismatches,
        labeled_as,
        clip_prediction,
        clip_confidence,
        all_bboxes_info,
        all_labels,
        state,
        updated_records,
        skip_data,
        img_fn,
        history,
        partial_save_skip_fn,
        parent=None
    ):
        super().__init__(parent)
        self.base_img_path= base_img_path
        self.pil_crop= pil_crop
        self.mismatch_index= mismatch_index
        self.total_mismatches= total_mismatches
        self.labeled_as= labeled_as
        self.clip_prediction= clip_prediction
        self.clip_confidence= clip_confidence
        self.all_bboxes_info= all_bboxes_info
        self.all_labels= all_labels
        self.state= state
        self.updated_records= updated_records
        self.skip_data= skip_data
        self.img_fn= img_fn
        self.history= history
        self.partial_save_skip_fn= partial_save_skip_fn

        self.assigned_label= None
        self.user_action= "do_nothing"

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Mismatch")
        self.setStyleSheet("""
        QDialog { background: #1C1C1C; color: #F5F5F5; }
        QLabel { color: #F5F5F5; }
        QLineEdit { background: #2A2A2A; color: #F5F5F5; border: 1px solid #444; }
        QPushButton { background: #333; color: #F5F5F5; border: 1px solid #555; padding: 6px; }
        QPushButton:hover { background: #444; }
        """)

        main_layout= QVBoxLayout()
        self.setLayout(main_layout)

        row_top= QHBoxLayout()
        main_layout.addLayout(row_top)

        mismatch_text=(
            f"Mismatch {self.mismatch_index+1}/{self.total_mismatches}\n"
            f"Labeled='{self.labeled_as}', CLIP='{self.clip_prediction}', conf={self.clip_confidence:.2f}"
        )
        lbl_top= QLabel(mismatch_text)
        lbl_top.setAlignment(Qt.AlignLeft|Qt.AlignTop)
        lbl_top.setWordWrap(True)
        row_top.addWidget(lbl_top)

        thr_box= QVBoxLayout()
        row_top.addLayout(thr_box)
        self.line_thr= QLineEdit(str(self.state["image_auto_threshold"]))
        thr_box.addWidget(self.line_thr)
        b_thr= QPushButton("Set Thr")
        b_thr.setStyleSheet("QPushButton { background-color: #BFFCC6; color: #000000; font-weight: bold; }")
        b_thr.clicked.connect(self.on_set_thr)
        thr_box.addWidget(b_thr)

        img_layout= QHBoxLayout()
        main_layout.addLayout(img_layout)

        self.label_base= QLabel()
        img_layout.addWidget(self.label_base)
        pm_base= self.draw_bboxes()
        self.label_base.setPixmap(pm_base)

        self.label_crop= QLabel()
        img_layout.addWidget(self.label_crop)
        pm_crop= self.load_crop_400(self.pil_crop)
        self.label_crop.setPixmap(pm_crop)

        btn_layout= QHBoxLayout()
        main_layout.addLayout(btn_layout)

        b_acc= QPushButton("Accept CLIP")
        b_acc.setStyleSheet("QPushButton { background-color: #FFB3BA; color: #000000; font-weight:bold; }")
        b_acc.clicked.connect(self.on_accept_clip)
        btn_layout.addWidget(b_acc)

        b_not= QPushButton("Do Nothing")
        b_not.setStyleSheet("QPushButton { background-color: #B3B3FF; color: #000000; font-weight:bold; }")
        b_not.clicked.connect(self.on_do_nothing)
        btn_layout.addWidget(b_not)

        b_del= QPushButton("Delete BBox")
        b_del.setStyleSheet("QPushButton { background-color: #FFD5CD; color: #000000; font-weight:bold; }")
        b_del.clicked.connect(self.on_delete_bbox)
        btn_layout.addWidget(b_del)

        b_remap= QPushButton("Remap Entire Class")
        b_remap.setStyleSheet("QPushButton { background-color: #FFFFBA; color: #000000; font-weight:bold; }")
        b_remap.clicked.connect(self.on_remap_class)
        btn_layout.addWidget(b_remap)

        b_undo= QPushButton("Undo")
        b_undo.setStyleSheet("QPushButton { background-color: #BFFCC6; color: #000000; font-weight:bold; }")
        b_undo.clicked.connect(self.on_undo)
        btn_layout.addWidget(b_undo)

        # row => each class => unique pastel
        row2= QHBoxLayout()
        main_layout.addLayout(row2)
        w2= QWidget()
        w2l= QHBoxLayout(w2)
        for i, lbl in enumerate(self.all_labels):
            color_idx= i % len(PASTEL_COLORS)
            color_hex= PASTEL_COLORS[color_idx]
            b= QPushButton(lbl)
            b.setStyleSheet(f"QPushButton {{ background-color: {color_hex}; color: #000000; font-weight:bold; }}")
            b.clicked.connect(lambda checked,lab=lbl: self.on_assign_label(lab))
            w2l.addWidget(b)
        row2.addWidget(w2)

    def draw_bboxes(self):
        if not os.path.exists(self.base_img_path):
            pm= QPixmap(200,200)
            pm.fill(Qt.gray)
            return pm
        img_bgr= cv2.imread(self.base_img_path)
        if img_bgr is None:
            pm= QPixmap(200,200)
            pm.fill(Qt.darkGray)
            return pm

        img_bgr, scale= limit_image_size(img_bgr)

        for i, bb in self.all_bboxes_info.items():
            x1= int(bb["x_min"])
            y1= int(bb["y_min"])
            x2= x1+ int(bb["w_px"])
            y2= y1+ int(bb["h_px"])
            x1= int(x1*scale)
            y1= int(y1*scale)
            x2= int(x2*scale)
            y2= int(y2*scale)

            if bb.get("is_current",False):
                color= (0,0,255)  # red
                thick=3
            elif bb.get("is_amended",False):
                color= (255,0,0)  # blue
                thick=2
            elif bb.get("is_correct",False):
                color= (0,255,0)  # green
                thick=2
            else:
                continue

            cv2.rectangle(img_bgr,(x1,y1),(x2,y2), color, thick)

        rgb= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qi= QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                   rgb.shape[1]*3, QImage.Format_RGB888)
        return QPixmap.fromImage(qi)

    def load_crop_400(self, pil_img):
        arr= np.array(pil_img.convert("RGB"))
        bgr= cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        h,w= bgr.shape[:2]
        largest= max(h,w)
        scale=1.0
        if largest>0:
            scale= 400.0/largest
        if scale!=1.0:
            new_w= int(w*scale)
            new_h= int(h*scale)
            bgr= cv2.resize(bgr,(new_w,new_h), interpolation=cv2.INTER_CUBIC)
        bgr2= cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qi= QImage(
            bgr2.data,
            bgr2.shape[1],
            bgr2.shape[0],
            bgr2.shape[1]*3,
            QImage.Format_RGB888
        )
        return QPixmap.fromImage(qi)

    def on_set_thr(self):
        txt= self.line_thr.text().strip()
        try:
            val= float(txt)
            if val<0:
                val=0.0
            self.state["image_auto_threshold"]= val
            print(f"[DEBUG] new auto thr => {val}")
        except ValueError:
            print("[DEBUG] invalid threshold => ignoring")

    def on_accept_clip(self):
        self.history.push(self.updated_records, self.skip_data)
        self.user_action= "accept_clip"
        self.mark_amended()
        self.close()

    def on_do_nothing(self):
        self.history.push(self.updated_records, self.skip_data)
        self.user_action= "do_nothing"
        self.mark_amended()
        self.close()

    def on_delete_bbox(self):
        self.history.push(self.updated_records, self.skip_data)
        self.user_action= "delete_bbox"
        self.mark_amended()
        self.close()

    def on_assign_label(self, label):
        self.history.push(self.updated_records, self.skip_data)
        self.user_action= "assign_label"
        self.assigned_label= label
        self.mark_amended()
        self.close()

    def on_remap_class(self):
        self.history.push(self.updated_records, self.skip_data)
        if self.mismatch_index<0 or self.mismatch_index>= len(self.updated_records):
            print("[ERROR] mismatch idx out of range => can't forced class remap.")
            return
        rec= self.updated_records[self.mismatch_index]
        if rec is None:
            print("[ERROR] bounding box= None => can't forced class remap.")
            return
        (old_cid,x_c,y_c,w_n,h_n)= rec
        if old_cid<0 or old_cid>= len(self.all_labels):
            print("[ERROR] invalid old_cid => can't forced class remap.")
            return

        old_name= self.all_labels[old_cid]
        same_indices= []
        for i, r2 in enumerate(self.updated_records):
            if r2 is None:
                continue
            (c2,xx,yy,ww,hh)= r2
            if c2== old_cid:
                same_indices.append(i)

        from PIL import Image
        forced_ratio=1.0
        pil_now= Image.open(self.base_img_path).convert("RGB")
        dlg= ClassRemapPreviewDialog(
            base_img_path= self.base_img_path,
            old_name= old_name,
            new_name= old_name,
            ratio= forced_ratio,
            box_indices= same_indices,
            yolo_records= self.updated_records,
            label_list= self.all_labels,
            pil_img= pil_now
        )
        dlg.exec_()

        if dlg.user_choice=="remap" and dlg.chosen_new_cid is not None:
            new_cid= dlg.chosen_new_cid
            for i, r3 in enumerate(self.updated_records):
                if r3 is None:
                    continue
                (cid3,xx,yy,ww,hh)= r3
                if cid3== old_cid:
                    self.updated_records[i]= (new_cid,xx,yy,ww,hh)
                    sig= make_bbox_signature(xx,yy,ww,hh)
                    self.skip_data[self.img_fn][sig]= True
            print(f"[INFO] forced class remap => old_cid={old_cid} => new_cid={new_cid}")
        else:
            print("[INFO] user canceled forced class remap")

        self.user_action= "do_nothing"
        self.mark_amended()
        self.close()

    def on_undo(self):
        old_r, old_s= self.history.pop()
        if old_r is None:
            print("[INFO] no older state => can't undo.")
            return
        self.updated_records[:] = old_r
        self.skip_data.clear()
        self.skip_data.update(old_s)
        print("[UNDO] Reverted => partial save skip.")
        self.partial_save_skip_fn()
        self.user_action= "do_nothing"
        self.close()

    def mark_amended(self):
        if self.mismatch_index in self.all_bboxes_info:
            self.all_bboxes_info[self.mismatch_index]["is_current"]= False
            self.all_bboxes_info[self.mismatch_index]["is_amended"]= True

###############################################################################
# 10) analyze_and_offer_class_remap => auto suggestion
###############################################################################
def analyze_and_offer_class_remap(
    yolo_records,
    label_list,
    pred_map,
    ratio_threshold,
    interactive,
    base_img_path,
    pil_full,
    skip_data,
    img_fn,
    history,
    partial_save_skip_fn
):
    # 1) Build counts per YOLO-labeled class => predicted class
    #    pred_map[i] = predicted label (string) for bounding box index i
    counts= defaultdict(lambda: defaultdict(int))
    class_to_indices= defaultdict(list)
    for i, rec in enumerate(yolo_records):
        if rec is None:
            continue
        (cid, x_c, y_c, w_n, h_n) = rec
        if cid <0 or cid>= len(label_list):
            continue
        yolo_lbl= label_list[cid]
        pred_lbl= pred_map.get(i, None)
        if pred_lbl is None:
            continue
        counts[yolo_lbl][pred_lbl]+= 1
        class_to_indices[yolo_lbl].append(i)

    # 2) For each YOLO-labeled class, see if the ratio of a single predicted label >= threshold
    for old_lbl, inner in counts.items():
        total= sum(inner.values())
        if total< MIN_CLASS_COUNT:
            continue
        best_pred, best_count= None, 0
        for k,v in inner.items():
            if v> best_count:
                best_pred, best_count= k, v
        ratio= best_count/ total if total>0 else 0
        if ratio>= ratio_threshold and best_pred is not None and best_pred!= old_lbl:
            # Offer a class remap
            idx_list= class_to_indices.get(old_lbl,[])
            # Show a full-screen preview dialog of these bboxes highlighted in orange
            if interactive:
                dlg= ClassRemapPreviewDialog(
                    base_img_path= base_img_path,
                    old_name= old_lbl,
                    new_name= best_pred,
                    ratio= ratio,
                    box_indices= idx_list,
                    yolo_records= yolo_records,
                    label_list= label_list,
                    pil_img= pil_full
                )
                dlg.exec_()
                if dlg.user_choice== "remap" and dlg.chosen_new_cid is not None:
                    new_idx= dlg.chosen_new_cid
                    for idx in idx_list:
                        (cid, x_c, y_c, w_n, h_n)= yolo_records[idx]
                        yolo_records[idx]= (new_idx, x_c, y_c, w_n, h_n)
                        sig= make_bbox_signature(x_c, y_c, w_n, h_n)
                        skip_data[img_fn][sig]= True
                    partial_save_skip_fn()
            else:
                print(f"[SUGGEST] Consider remapping class '{old_lbl}' => '{best_pred}' (ratio={ratio:.2f}).")

###############################################################################
# 11) main
###############################################################################
def main():
    args= parse_args()
    os.makedirs(args.corrected_labels_path, exist_ok=True)

    # skip-log => dict[img_fn][bbox_signature]= True
    if os.path.isfile(LOG_FILENAME):
        skip_data= load_skip_log(LOG_FILENAME)
    else:
        skip_data= {}

    device= "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading CLIP...")
    clip_model, preprocess= clip.load("ViT-B/32", device=device)
    print("Loading logistic regression model...")
    clf= joblib.load(args.model_path)
    label_list= joblib.load(args.labelmap_path)

    image_files= [f for f in os.listdir(args.images_path) if f.lower().endswith((".jpg",".jpeg",".png"))]
    image_files.sort()

    app= QApplication(sys.argv)

    def partial_save_skip():
        save_skip_log(LOG_FILENAME, skip_data)

    for img_fn in image_files:
        full_img_path= os.path.join(args.images_path, img_fn)
        base_name, _= os.path.splitext(img_fn)
        label_path= os.path.join(args.labels_path, base_name+ ".txt")
        yolo_records= load_yolo_file(label_path)
        if not yolo_records:
            continue
        pil_full= Image.open(full_img_path).convert("RGB")
        w_img, h_img= pil_full.size

        if img_fn not in skip_data:
            skip_data[img_fn]= {}

        history= StateHistory()

        # Step 1 => compute predicted labels for each bbox
        pred_map= {}
        for i,(cid,x_c,y_c,w_n,h_n) in enumerate(yolo_records):
            x_min= (x_c-0.5*w_n)* w_img
            y_min= (y_c-0.5*h_n)* h_img
            x_max= x_min+ w_n*w_img
            y_max= y_min+ h_n*h_img
            sub_pil= crop_pil_image(pil_full, x_min,y_min,x_max,y_max)
            pl, conf= clip_predict_label_and_conf(sub_pil, clip_model, preprocess, clf, label_list, device)
            pred_map[i]= pl

        # Step 2 => offer class remap if many boxes of a YOLO class go to the same predicted class
        analyze_and_offer_class_remap(
            yolo_records= yolo_records,
            label_list= label_list,
            pred_map= pred_map,
            ratio_threshold= args.class_remap_ratio,
            interactive= args.interactive,
            base_img_path= full_img_path,
            pil_full= pil_full,
            skip_data= skip_data,
            img_fn= img_fn,
            history= history,
            partial_save_skip_fn= partial_save_skip
        )

        # final mismatch => dictionary-based
        all_bboxes_info={}
        final_preds=[]
        for i,(cid,x_c,y_c,w_n,h_n) in enumerate(yolo_records):
            if cid is None:
                continue
            sig= make_bbox_signature(x_c,y_c,w_n,h_n)
            if sig in skip_data[img_fn]:
                continue
            x_min= (x_c-0.5*w_n)* w_img
            y_min= (y_c-0.5*h_n)* h_img
            x_max= x_min+ w_n*w_img
            y_max= y_min+ h_n*h_img
            sub_pil= crop_pil_image(pil_full, x_min,y_min,x_max,y_max)
            pl, cval= clip_predict_label_and_conf(sub_pil, clip_model, preprocess, clf, label_list, device)
            if 0<= cid< len(label_list):
                final_lbl= label_list[cid]
            else:
                final_lbl= f"unknown_id_{cid}"

            is_correct= (pl== final_lbl)
            all_bboxes_info[i]= {
                "x_min": x_min,
                "y_min": y_min,
                "w_px": (x_max-x_min),
                "h_px": (y_max-y_min),
                "is_correct": is_correct,
                "is_current": False,
                "is_amended": False
            }
            final_preds.append((i,cid,pl,cval,final_lbl))

        image_state= {"image_auto_threshold": args.clip_auto}

        for (idx,cid, p_label, conf, final_lbl) in final_preds:
            sig=None
            if idx< len(yolo_records) and yolo_records[idx] is not None:
                (cc,xx,yy,ww,hh)= yolo_records[idx]
                sig= make_bbox_signature(xx,yy,ww,hh)
            if sig is None or sig in skip_data[img_fn]:
                continue

            if p_label!= final_lbl:
                if image_state["image_auto_threshold"]>0 and conf>= image_state["image_auto_threshold"]:
                    history.push(yolo_records, skip_data)
                    if p_label in label_list:
                        new_cid= label_list.index(p_label)
                        x_c,y_c,wn,hh= yolo_records[idx][1:]
                        yolo_records[idx]= (new_cid,x_c,y_c,wn,hh)
                        print(f"[AUTO-FIX] {img_fn}, BBox#{idx}: {final_lbl} => {p_label}, conf={conf:.2f}")
                    else:
                        print(f"[AUTO-FIX FAIL] unknown label {p_label}")
                    skip_data[img_fn][sig]= True
                    partial_save_skip()
                else:
                    if args.interactive:
                        all_bboxes_info[idx]["is_current"]= True
                        mismatch_crop= crop_pil_image(
                            pil_full,
                            all_bboxes_info[idx]["x_min"],
                            all_bboxes_info[idx]["y_min"],
                            all_bboxes_info[idx]["x_min"]+ all_bboxes_info[idx]["w_px"],
                            all_bboxes_info[idx]["y_min"]+ all_bboxes_info[idx]["h_px"]
                        )
                        d= MismatchDialog(
                            base_img_path= full_img_path,
                            pil_crop= mismatch_crop,
                            mismatch_index= idx,
                            total_mismatches= len(yolo_records),
                            labeled_as= final_lbl,
                            clip_prediction= p_label,
                            clip_confidence= conf,
                            all_bboxes_info= all_bboxes_info,
                            all_labels= label_list,
                            state= image_state,
                            updated_records= yolo_records,
                            skip_data= skip_data,
                            img_fn= img_fn,
                            history= history,
                            partial_save_skip_fn= partial_save_skip
                        )
                        d.exec_()

                        if d.user_action=="accept_clip":
                            if p_label in label_list:
                                new_cid= label_list.index(p_label)
                                x_c,y_c,wn,hh= yolo_records[idx][1:]
                                yolo_records[idx]= (new_cid,x_c,y_c,wn,hh)
                        elif d.user_action=="delete_bbox":
                            yolo_records[idx]= None
                        elif d.user_action=="assign_label":
                            c_lbl= d.assigned_label
                            if c_lbl in label_list:
                                new_cid= label_list.index(c_lbl)
                                x_c,y_c,wn,hh= yolo_records[idx][1:]
                                yolo_records[idx]= (new_cid,x_c,y_c,wn,hh)

                    print(f"[MISMATCH] {img_fn}, BBox#{idx}, Labeled='{final_lbl}', Pred='{p_label}', conf={conf:.2f}")
                    skip_data[img_fn][sig]= True
                    partial_save_skip()
            else:
                print(f"[OK] {img_fn}, BBox#{idx} => '{final_lbl}' conf={conf:.2f}")
            # tqdm update was here in the root version; omitted to keep UI responsive

        final_list= [r for r in yolo_records if r is not None]
        out_txt= os.path.join(args.corrected_labels_path, base_name+".txt")
        with open(out_txt, "w") as f:
            for (cid,x_c,y_c,w_n,h_n) in final_list:
                f.write(f"{cid} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

    # Save skip log on exit
    save_skip_log(LOG_FILENAME, skip_data)

if __name__=="__main__":
    main()

