import os
import pickle
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


# ------------------ Streamlit 介面佈局與 CSS 樣式優化 ------------------
st.set_page_config(page_title="AI 綜合智慧辨識與人臉學習系統", layout="centered")

st.markdown(
    """
    <style>
    /* 將側邊欄單選按鈕美化成現代化區塊按鈕，徹底避開 selectbox 的游標與紅框問題 */
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

# 側邊欄導航：改用 st.radio 並賦予精美按鈕外觀，解決所有邊框與游標問題
st.sidebar.markdown("### 導航選單")
menu = st.sidebar.radio(
    "選擇功能模式",
    ["1. 註冊與訓練新人物", "2. 綜合識別與人臉測試", "3. 管理與修改已儲存模型"],
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
        if not os.path.exists(person_folder):
          os.makedirs(person_folder)

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

        all_faces = []
        all_ids = []
        new_label_map = {}
        dataset_path = os.path.join(MODEL_DIR, "dataset")

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
            for k, v in label_map.items():
              if v.get("name") == name:
                existing_code = v.get("code", "")
                break
            if name == person_name:
              existing_code = person_code

            new_label_map[current_id] = {"name": name, "code": existing_code}

            person_dir = os.path.join(dataset_path, name)
            for img_name in os.listdir(person_dir):
              if img_name.lower().endswith(
                  (".png", ".jpg", ".jpeg", ".webp")
              ):
                img_path = os.path.join(person_dir, img_name)
                try:
                  pil_img = Image.open(img_path).convert("L")
                  img_np = np.array(pil_img)
                  all_faces.append(img_np)
                  all_ids.append(current_id)
                except Exception:
                  pass

          if len(new_label_map) > 0:
            label_map = new_label_map

        if len(all_faces) == 1:
          all_faces.append(cv2.flip(all_faces[0], 1))
          all_ids.append(all_ids[0])

        if len(all_faces) > 0:
          recognizer = cv2.face.LBPHFaceRecognizer_create()
          recognizer.train(all_faces, np.array(all_ids))
          recognizer.save(MODEL_PATH)
          with open(LABELS_PATH, "wb") as f:
            pickle.dump(label_map, f)

          st.success(
              f"🎉 成功為「{person_name}」完成訓練與資料累積！模型已更新並儲存。"
          )
          st.balloons()
        else:
          st.error("訓練資料不足！請確認上傳的圖片檔案格式正確。")

# ------------------ 功能二：綜合識別與人臉測試 ------------------
elif menu == "2. 綜合識別與人臉測試":
  st.header("🔍 步驟二：上傳照片進行綜合辨識")
  st.write(
      "上傳照片後，若成功匹配自定義人臉模型將進行點名並輸出下方文字；若為陌生人或一般物件則僅顯示純淨標註框。"
  )

  test_file = st.file_uploader(
      "上傳要辨識的照片", type=["jpg", "jpeg", "png", "webp"]
  )

  if st.button("🚀 開始綜合辨識", type="primary"):
    if test_file is None:
      st.error("請先上傳測試圖片！")
    else:
      with st.spinner("AI 正在進行智慧識別與標註中..."):
        det_model = load_detection_model()
        image = Image.open(test_file).convert("RGB")

        transform = trns.ToTensor()
        image_tensor = transform(image)

        with torch.no_grad():
          predictions = det_model([image_tensor])[0]

        display_image = np.array(image).astype(np.float32)
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

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

        boxes = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()

        # 【強力 NMS 與重複框過濾】：嚴格過濾多人合照時產生的多餘重疊框
        valid_indices = []
        for i, (bbox, label_idx, score) in enumerate(zip(boxes, labels, scores)):
          eng_name = _COCO_INSTANCE_CATEGORY_NAMES[label_idx]
          if score >= 0.7:  # 提高信心門檻，過濾掉低品質雜訊框
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
                # 若重疊面積大於 30%，視為同一個人的重複冗餘框，直接濾除
                if inter_area / (box_area + 1e-5) > 0.3:
                  is_redundant = True
                  break
              if not is_redundant:
                valid_indices.append(i)
            else:
              if score >= 0.6:
                valid_indices.append(i)

        detected_count = 0
        detection_records = []

        for i in valid_indices:
          bbox = boxes[i]
          label_idx = labels[i]
          score = scores[i]

          x0, y0, x1, y1 = map(int, bbox)
          eng_name = _COCO_INSTANCE_CATEGORY_NAMES[label_idx]
          zh_name = _COCO_ZH_NAMES.get(eng_name, eng_name)
          label_text = f"{zh_name} ({int(score * 100)}%)"

          if eng_name == "person" and recognizer is not None and has_custom_model:
            box_h = y1 - y0
            face_y1 = y0 + int(box_h * 0.75)
            person_crop_gray = gray[y0:face_y1, x0:x1]

            if person_crop_gray.size > 100:
              try:
                face_resized = cv2.resize(person_crop_gray, (200, 200))
                p_id, conf = recognizer.predict(face_resized)

                # 寬鬆合理的信心閥值 (conf < 110)，確保資料庫內的人絕對能精準辨識
                if p_id in label_map and conf < 110:
                  person_info = label_map.get(p_id)
                  custom_name = person_info["name"]
                  custom_code = person_info.get("code", "")
                  sim_score = max(0, int(100 - conf / 1.5))
                  label_text = f"{custom_name} ({sim_score}%)"

                  detection_records.append(
                      {"name": custom_name, "code": custom_code}
                  )
                else:
                  label_text = "未知人物"
              except Exception:
                pass

          display_image = display_image.astype(np.uint8)
          cv2.rectangle(display_image, (x0, y0), (x1, y1), (18, 127, 15), 2)

          font = cv2.FONT_HERSHEY_SIMPLEX
          font_scale = 0.5
          ((text_w, text_h), _) = cv2.getTextSize(
              label_text, font, font_scale, 1
          )
          back_tl = x0, y0 - int(1.4 * text_h)
          back_br = x0 + text_w + 6, y0
          cv2.rectangle(display_image, back_tl, back_br, (18, 127, 15), -1)
          cv2.putText(
              display_image,
              label_text,
              (x0 + 3, y0 - int(0.4 * text_h)),
              font,
              font_scale,
              (255, 255, 255),
              1,
              cv2.LINE_AA,
          )
          detected_count += 1

        if detected_count == 0:
          h_img, w_img, _ = display_image.shape
          display_image = display_image.astype(np.uint8)
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

        display_image = display_image.astype(np.uint8)
        st.image(
            display_image, caption="AI 智慧辨識結果", use_container_width=True
        )

        if has_custom_model and len(detection_records) > 0:
          total_count = len(detection_records)
          fen_bie_str = "分別" if total_count > 1 else ""

          result_lines = [
              f"成功偵測並分析 {total_count} 個臉部目標！",
              f"目標{fen_bie_str}為：",
          ]

          for rec in detection_records:
            p_name = rec["name"]
            p_code = rec["code"]
            if p_code and p_code.strip():
              result_lines.append(f"{p_name} 編號:{p_code}")
            else:
              result_lines.append(f"{p_name}")

          st.success("\n\n".join(result_lines))

# ------------------ 功能三：管理與修改已儲存模型 ------------------
elif menu == "3. 管理與修改已儲存模型":
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

    for idx, (p_id, p_info) in enumerate(label_map.items()):
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
                import shutil
                shutil.rmtree(new_folder)
              os.rename(old_folder, new_folder)

          label_map[selected_id] = {"name": new_name, "code": new_code}

          with open(LABELS_PATH, "wb") as f:
            pickle.dump(label_map, f)

          st.success(f"已成功更新「{selected_name}」的模型資料！")
          st.rerun()

    with col2:
      if st.button("🗑️ 刪除此人物模型", type="secondary"):
        import shutil

        target_folder = os.path.join(MODEL_DIR, "dataset", selected_name)
        if os.path.exists(target_folder):
          shutil.rmtree(target_folder)

        del label_map[selected_id]

        if len(label_map) > 0:
          all_faces = []
          all_ids = []
          dataset_path = os.path.join(MODEL_DIR, "dataset")
          new_label_map = {}

          name_list = sorted(
              [
                  d
                  for d in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, d))
              ]
          )
          for current_id, name in enumerate(name_list):
            existing_code = ""
            for k, v in label_map.items():
              if v.get("name") == name:
                existing_code = v.get("code", "")
                break

            new_label_map[current_id] = {"name": name, "code": existing_code}
            person_dir = os.path.join(dataset_path, name)
            for img_name in os.listdir(person_dir):
              if img_name.lower().endswith(
                  (".png", ".jpg", ".jpeg", ".webp")
              ):
                img_path = os.path.join(person_dir, img_name)
                try:
                  pil_img = Image.open(img_path).convert("L")
                  img_np = np.array(pil_img)
                  all_faces.append(img_np)
                  all_ids.append(current_id)
                except Exception:
                  pass

          label_map = new_label_map

          if len(all_faces) == 1:
            all_faces.append(cv2.flip(all_faces[0], 1))
            all_ids.append(all_ids[0])

          if len(all_faces) > 0:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(all_faces, np.array(all_ids))
            recognizer.save(MODEL_PATH)
            with open(LABELS_PATH, "wb") as f:
              pickle.dump(label_map, f)
          else:
            if os.path.exists(MODEL_PATH):
              os.remove(MODEL_PATH)
            if os.path.exists(LABELS_PATH):
              os.remove(LABELS_PATH)
        else:
          if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
          if os.path.exists(LABELS_PATH):
            os.remove(LABELS_PATH)

        st.success(f"已成功刪除「{selected_name}」的模型資料！")
        st.rerun()
