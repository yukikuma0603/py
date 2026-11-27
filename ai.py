import os
import pickle
import shutil
import cv2
import numpy as np
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as trns
from PIL import Image

# ------------------ 設定儲存路徑 ------------------
MODEL_DIR = "model_data"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "face_model.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pickle")

# COCO 類別中文對照表
_COCO_ZH_NAMES = {
    "person": "人",
    "bicycle": "腳踏車",
    "car": "汽車",
    "motorcycle": "機車",
    "airplane": "飛機",
    "bus": "公車",
    "train": "火車",
    "truck": "卡車",
    "boat": "船",
    "traffic light": "紅綠燈",
    "stop sign": "停止標誌",
    "bench": "長凳",
    "bird": "鳥",
    "cat": "貓",
    "dog": "狗",
    "horse": "馬",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "大象",
    "bear": "熊",
    "zebra": "斑馬",
    "giraffe": "長頸鹿",
    "backpack": "後背包",
    "umbrella": "雨傘",
    "handbag": "手提包",
    "tie": "領帶",
    "suitcase": "行李箱",
    "frisbee": "飛盤",
    "skis": "滑雪板",
    "sports ball": "球",
    "bottle": "瓶子",
    "wine glass": "酒杯",
    "cup": "杯子",
    "fork": "叉子",
    "knife": "刀",
    "spoon": "湯匙",
    "bowl": "碗",
    "banana": "香蕉",
    "apple": "蘋果",
    "sandwich": "三明治",
    "orange": "橘子",
    "broccoli": "花椰菜",
    "carrot": "紅蘿蔔",
    "hot dog": "熱狗",
    "pizza": "披薩",
    "donut": "甜甜圈",
    "cake": "蛋糕",
    "chair": "椅子",
    "couch": "沙發",
    "potted plant": "盆栽",
    "bed": "床",
    "dining table": "餐桌",
    "toilet": "馬桶",
    "tv": "電視",
    "laptop": "筆電",
    "mouse": "滑鼠",
    "remote": "遙控器",
    "keyboard": "鍵盤",
    "cell phone": "手機",
    "microwave": "微波爐",
    "oven": "烤箱",
    "toaster": "烤吐司機",
    "sink": "洗手台",
    "refrigerator": "冰箱",
    "book": "書本",
    "clock": "時鐘",
    "vase": "花瓶",
    "scissors": "剪刀",
    "teddy bear": "泰迪熊",
    "hair drier": "吹風機",
    "toothbrush": "牙刷",
}

_COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# 快取載入 PyTorch 模型
@st.cache_resource
def load_detection_model():
  model = models.detection.fasterrcnn_resnet50_fpn(
      weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
  )
  model.eval()
  return model


# 載入自定義 LBPH 人臉模型
def load_face_model():
  recognizer = None
  label_map = {}
  has_custom_model = False
  if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    try:
      recognizer = cv2.face.LBPHFaceRecognizer_create()
      recognizer.read(MODEL_PATH)
      with open(LABELS_PATH, "rb") as f:
        loaded_data = pickle.load(f)

      for k, v in loaded_data.items():
        if isinstance(v, str):
          label_map[k] = {"name": v, "code": ""}
        else:
          label_map[k] = v

      if len(label_map) > 0:
        has_custom_model = True
    except Exception:
      pass
  return recognizer, label_map, has_custom_model


# 過濾重複框（NMS）
def filter_valid_indices(boxes, labels, scores):
  valid_indices = []
  for i, (bbox, label_idx, score) in enumerate(zip(boxes, labels, scores)):
    eng_name = _COCO_INSTANCE_CATEGORY_NAMES[label_idx]
    if score >= 0.7:
      if eng_name == "person":
        is_redundant = False
        for j in valid_indices:
          jb = boxes[j]
          inter_x0 = max(bbox[0], jb[0])
          inter_y0 = max(bbox[1], jb[1])
          inter_x1 = min(bbox[2], jb[2])
          inter_y1 = min(bbox[3], jb[3])
          inter_area = max(0, inter_x1 - inter_x0) * max(
              0, inter_y1 - inter_y0
          )
          box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
          if inter_area / (box_area + 1e-5) > 0.3:
            is_redundant = True
            break
        if not is_redundant:
          valid_indices.append(i)
      else:
        if score >= 0.6:
          valid_indices.append(i)
  return valid_indices


# 畫外框與標籤
def draw_box_with_label(display_image, x0, y0, x1, y1, label_text):
  cv2.rectangle(display_image, (x0, y0), (x1, y1), (18, 127, 15), 2)

  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.5
  ((text_w, text_h), _) = cv2.getTextSize(label_text, font, font_scale, 1)
  back_tl = (x0, max(0, y0 - int(1.4 * text_h)))
  back_br = (x0 + text_w + 6, max(text_h + 4, y0))
  cv2.rectangle(display_image, back_tl, back_br, (18, 127, 15), -1)
  cv2.putText(
      display_image,
      label_text,
      (x0 + 3, max(text_h + 1, y0 - int(0.4 * text_h))),
      font,
      font_scale,
      (255, 255, 255),
      1,
      cv2.LINE_AA,
  )


# 依目前 dataset 與標籤紀錄重建 LBPH 模型；若無資料則清理模型檔
def rebuild_model_files(current_label_map):
  dataset_path = os.path.join(MODEL_DIR, "dataset")
  all_faces = []
  all_ids = []
  new_label_map = {}

  if os.path.exists(dataset_path):
    name_list = sorted(
        [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
    )
    for current_id, name in enumerate(name_list):
      existing_code = ""
      for _, v in current_label_map.items():
        if v.get("name") == name:
          existing_code = v.get("code", "")
          break
      new_label_map[current_id] = {"name": name, "code": existing_code}

      person_dir = os.path.join(dataset_path, name)
      for img_name in os.listdir(person_dir):
        if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
          img_path = os.path.join(person_dir, img_name)
          try:
            pil_img = Image.open(img_path).convert("L")
            all_faces.append(np.array(pil_img))
            all_ids.append(current_id)
          except Exception:
            pass

  if len(all_faces) == 1:
    all_faces.append(cv2.flip(all_faces[0], 1))
    all_ids.append(all_ids[0])

  if len(all_faces) > 0:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(all_faces, np.array(all_ids))
    recognizer.save(MODEL_PATH)
    with open(LABELS_PATH, "wb") as f:
      pickle.dump(new_label_map, f)
  else:
    for path in (MODEL_PATH, LABELS_PATH):
      if os.path.exists(path):
        os.remove(path)


# ------------------ Streamlit 介面佈局與 CSS 樣式優化 ------------------
st.set_page_config(page_title="AI 綜合智慧辨識與人臉學習系統", layout="centered")

st.markdown(
    """
    <style>
    div.row-widget.stRadio > div {
        flex-direction: column;
        gap: 8px;
    }
    div.row-widget.stRadio label {
        background-color: #f8f9fa;
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        cursor: pointer;
        width: 100%;
        font-weight: 500;
    }
    div.row-widget.stRadio label:hover {
        background-color: #e9ecef;
        border-color: #2e6930;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🧠 AI 綜合智慧辨識與人臉學習系統")

st.sidebar.markdown("### 導航選單")
menu = st.sidebar.radio(
    "選擇功能模式",
    [
        "1. 註冊與訓練新人物",
        "2. COCO 物件識別",
        "3. 機器學習人臉識別",
        "4. 管理與修改已儲存模型",
    ],
    label_visibility="collapsed",
)

# ------------------ 功能一：註冊與訓練新人物 ------------------
if menu == "1. 註冊與訓練新人物":
  st.header("👤 步驟一：新增人物與上傳照片訓練")
  st.write(
      "輸入人物姓名與編號，並上傳照片（可支援單張大頭照/證件照或多張照片）。"
  )

  raw_input_name = st.text_input(
      "請輸入目標姓名與編號",
      placeholder="輸入格式請為: 目標姓名 編號 (若無編號則無需填寫)",
  )
  uploaded_files = st.file_uploader(
      "上傳該人物的臉部照片（可接受單張或多張）",
      type=["jpg", "jpeg", "png", "webp"],
      accept_multiple_files=True,
  )

  if st.button("🚀 開始訓練並儲存模型", type="primary"):
    if not raw_input_name.strip():
      st.error("請先輸入目標姓名！")
    elif not uploaded_files:
      st.error("請至少上傳一張照片！")
    else:
      parts = raw_input_name.strip().split()
      person_name = parts[0]
      person_code = parts[1] if len(parts) > 1 else ""

      with st.spinner("正在處理影像並訓練 AI 模型中..."):
        label_map = {}
        if os.path.exists(LABELS_PATH):
          try:
            with open(LABELS_PATH, "rb") as f:
              loaded = pickle.load(f)
              for k, v in loaded.items():
                if isinstance(v, str):
                  label_map[k] = {"name": v, "code": ""}
                else:
                  label_map[k] = v
          except Exception:
            label_map = {}

        person_folder = os.path.join(MODEL_DIR, "dataset", person_name)
        os.makedirs(person_folder, exist_ok=True)

        for uploaded_file in uploaded_files:
          try:
            image = Image.open(uploaded_file).convert("L")
            w, h = image.size
            image_face = image.crop((0, 0, w, int(h * 0.75)))
            image_resized = image_face.resize((200, 200))
            file_name = f"face_{os.urandom(4).hex()}.jpg"
            image_resized.save(os.path.join(person_folder, file_name))
          except Exception:
            pass

        temp_map = dict(label_map)
        updated_existing = False
        for k, v in temp_map.items():
          if v.get("name") == person_name:
            temp_map[k] = {"name": person_name, "code": person_code}
            updated_existing = True
            break
        if not updated_existing:
          next_id = (max(temp_map.keys()) + 1) if temp_map else 0
          temp_map[next_id] = {"name": person_name, "code": person_code}

        rebuild_model_files(temp_map)

        if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
          st.success(
              f"🎉 成功為「{person_name}」完成訓練與資料累積！模型已更新並儲存。"
          )
          st.balloons()
        else:
          st.error("訓練資料不足！請確認上傳的圖片檔案格式正確。")

# ------------------ 功能二：COCO 物件識別 ------------------
elif menu == "2. COCO 物件識別":
  st.header("📦 COCO 物件識別")
  st.write("上傳照片進行通用物件檢測，僅顯示標註框，不輸出下方點名文字。")

  coco_file = st.file_uploader(
      "上傳要進行物件識別的照片",
      type=["jpg", "jpeg", "png", "webp"],
      key="coco_uploader",
  )

  if st.button("🚀 開始 COCO 物件識別", type="primary", key="btn_coco"):
    if coco_file is None:
      st.error("請先上傳圖片！")
    else:
      with st.spinner("AI 正在辨識 COCO 物件中..."):
        det_model = load_detection_model()
        image = Image.open(coco_file).convert("RGB")
        image_tensor = trns.ToTensor()(image)

        with torch.no_grad():
          predictions = det_model([image_tensor])[0]

        display_image = np.array(image).astype(np.uint8)
        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()

        valid_indices = filter_valid_indices(boxes, labels, scores)
        detected_count = 0

        for i in valid_indices:
          bbox = boxes[i]
          label_idx = labels[i]
          score = scores[i]

          x0, y0, x1, y1 = map(int, bbox)
          eng_name = _COCO_INSTANCE_CATEGORY_NAMES[label_idx]
          zh_name = _COCO_ZH_NAMES.get(eng_name, eng_name)
          label_text = f"{zh_name} ({int(score * 100)}%)"

          draw_box_with_label(display_image, x0, y0, x1, y1, label_text)
          detected_count += 1

        if detected_count == 0:
          h_img, w_img, _ = display_image.shape
          cv2.rectangle(
              display_image,
              (10, 10),
              (w_img - 10, h_img - 10),
              (255, 165, 0),
              3,
          )
          cv2.putText(
              display_image,
              "一般目標辨識",
              (20, 40),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.8,
              (255, 165, 0),
              2,
              cv2.LINE_AA,
          )

        st.image(
            display_image, caption="COCO 物件識別結果", use_container_width=True
        )

# ------------------ 功能三：機器學習人臉識別 ------------------
elif menu == "3. 機器學習人臉識別":
  st.header("🧑‍💼 機器學習人臉識別")
  st.write(
      "上傳照片比對已訓練的人物模型；只有匹配到自定義模型的人物才會在下方輸出點名文字。"
  )

  ml_file = st.file_uploader(
      "上傳要進行人臉識別的照片",
      type=["jpg", "jpeg", "png", "webp"],
      key="ml_uploader",
  )

  if st.button("🚀 開始機器學習人臉識別", type="primary", key="btn_ml"):
    if ml_file is None:
      st.error("請先上傳圖片！")
    else:
      recognizer, label_map, has_custom_model = load_face_model()
      if not has_custom_model or recognizer is None:
        st.warning("目前尚無已儲存的人物模型，請先前往「1. 註冊與訓練新人物」。")
      else:
        with st.spinner("AI 正在進行機器學習人臉比對中..."):
          det_model = load_detection_model()
          image = Image.open(ml_file).convert("RGB")
          image_tensor = trns.ToTensor()(image)

          with torch.no_grad():
            predictions = det_model([image_tensor])[0]

          display_image = np.array(image).astype(np.uint8)
          gray = cv2.cvtColor(display_image, cv2.COLOR_RGB2GRAY)

          boxes = predictions["boxes"].cpu().numpy()
          labels = predictions["labels"].cpu().numpy()
          scores = predictions["scores"].cpu().numpy()

          person_indices = [
              i
              for i in filter_valid_indices(boxes, labels, scores)
              if _COCO_INSTANCE_CATEGORY_NAMES[labels[i]] == "person"
          ]

          detection_records = []

          for i in person_indices:
            x0, y0, x1, y1 = map(int, boxes[i])
            box_h = y1 - y0
            face_y1 = y0 + int(box_h * 0.75)
            person_crop_gray = gray[y0:face_y1, x0:x1]

            label_text = "未知人物"
            if person_crop_gray.size > 100:
              try:
                face_resized = cv2.resize(person_crop_gray, (200, 200))
                p_id, conf = recognizer.predict(face_resized)

                if p_id in label_map and conf < 110:
                  person_info = label_map.get(p_id)
                  custom_name = person_info["name"]
                  custom_code = person_info.get("code", "")
                  sim_score = max(0, int(100 - conf / 1.5))
                  label_text = f"{custom_name} ({sim_score}%)"

                  detection_records.append(
                      {"name": custom_name, "code": custom_code}
                  )
              except Exception:
                pass

            draw_box_with_label(display_image, x0, y0, x1, y1, label_text)

          st.image(
              display_image,
              caption="機器學習人臉識別結果",
              use_container_width=True,
          )

          if len(detection_records) > 0:
            total_count = len(detection_records)
            fen_bie_str = "分別" if total_count > 1 else ""

            result_lines = [
                f"成功偵測並分析 {total_count} 個臉部目標！",
                f"目標{fen_bie_str}為：",
            ]

            for rec in detection_records:
              p_name = rec["name"]
              p_code = rec.get("code", "")
              if p_code and p_code.strip():
                result_lines.append(f"{p_name} 編號:{p_code}")
              else:
                result_lines.append(f"{p_name}")

            st.success("\n\n".join(result_lines))

# ------------------ 功能四：管理與修改已儲存模型 ------------------
elif menu == "4. 管理與修改已儲存模型":
  st.header("⚙️ 管理與修改已儲存的模型資料")

  if not os.path.exists(LABELS_PATH) or not os.path.exists(MODEL_PATH):
    st.warning("目前尚無任何已儲存的模型與標籤資料！")
  else:
    with open(LABELS_PATH, "rb") as f:
      loaded_data = pickle.load(f)

    label_map = {}
    for k, v in loaded_data.items():
      if isinstance(v, str):
        label_map[k] = {"name": v, "code": ""}
      else:
        label_map[k] = v

    st.write(f"目前共儲存了 **{len(label_map)}** 位人物模型：")

    for idx, (_, p_info) in enumerate(label_map.items()):
      p_name = p_info["name"]
      p_code = p_info.get("code", "")
      code_str = p_code if p_code else "無"
      st.markdown(f"- **模型 {idx + 1}**：{p_name}，編號：`{code_str}`")

    st.markdown("---")
    st.subheader("✏️ 修改或刪除特定人物模型")

    selected_name = st.selectbox(
        "選擇要修改或刪除的人物", [v["name"] for v in label_map.values()]
    )

    selected_id = [
        k for k, v in label_map.items() if v["name"] == selected_name
    ][0]
    current_info = label_map[selected_id]

    new_input_val = st.text_input(
        "修改後的姓名與編號 (格式：姓名 編號)",
        value=f"{current_info['name']} {current_info.get('code', '')}".strip(),
    )

    col1, col2 = st.columns(2)

    with col1:
      if st.button("💾 儲存修改名稱", type="primary"):
        if not new_input_val.strip():
          st.error("輸入不能為空！")
        else:
          parts = new_input_val.strip().split()
          new_name = parts[0]
          new_code = parts[1] if len(parts) > 1 else ""

          old_folder = os.path.join(MODEL_DIR, "dataset", selected_name)
          new_folder = os.path.join(MODEL_DIR, "dataset", new_name)

          if os.path.exists(old_folder):
            if old_folder != new_folder:
              if os.path.exists(new_folder):
                shutil.rmtree(new_folder)
              os.rename(old_folder, new_folder)

          label_map[selected_id] = {"name": new_name, "code": new_code}

          with open(LABELS_PATH, "wb") as f:
            pickle.dump(label_map, f)

          st.success(f"已成功更新「{selected_name}」的模型資料！")
          st.rerun()

    with col2:
      if st.button("🗑️ 刪除此人物模型", type="secondary"):
        target_folder = os.path.join(MODEL_DIR, "dataset", selected_name)
        if os.path.exists(target_folder):
          shutil.rmtree(target_folder)

        updated_label_map = {
            k: v
            for k, v in label_map.items()
            if v.get("name") != selected_name
        }
        rebuild_model_files(updated_label_map)

        st.success(f"已成功刪除「{selected_name}」並更新模型！")
        st.rerun()
