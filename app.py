from flask import Flask, request, render_template, redirect, url_for
from linebot import LineBotApi
from linebot.models import TextSendMessage, ImageSendMessage
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv
import uuid
from datetime import datetime

# 載入 .env 檔案
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 讀取 LINE Bot 憑證
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
USER_ID = os.getenv("USER_ID")
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)

# 載入模型
model = load_model('model_inception.h5')

# 類別對應資訊
class_info = {
    'healthy': {'status': '健康', 'cause': '無病害', 'solution': '維持良好環境即可。'},
    'bacterial_spot': {'status': '細菌性斑點病', 'cause': '潮濕環境感染。', 'solution': '移除病葉，用銅劑處理。'},
    'early_blight': {'status': '早疫病', 'cause': '老葉感染 Alternaria。', 'solution': '清除病葉，噴藥。'},
    'late_blight': {'status': '晚疫病', 'cause': '疫黴菌感染。', 'solution': '噴甲霜靈，移除感染區。'},
    'leaf_mold': {'status': '葉黴病', 'cause': '高濕黴菌感染。', 'solution': '通風、殺菌劑。'}
}
class_names = list(class_info.keys())

# 圖片轉換與壓縮（最大 1MB）
def convert_and_compress_image(input_path, output_path, max_size_kb=1000):
    with Image.open(input_path) as img:
        img = img.convert('RGB')
        quality = 95
        while True:
            img.save(output_path, format='JPEG', quality=quality)
            size_kb = os.path.getsize(output_path) // 1024
            if size_kb <= max_size_kb or quality <= 30:
                break
            quality -= 5

# 預測圖片
def predict_image(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    idx = np.argmax(preds)
    label = class_names[idx]
    confidence = float(np.max(preds))
    return label, confidence

# 首頁與上傳處理
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return redirect(request.url)

        # 取得當前時間
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        filename_time = now.strftime("%Y%m%d_%H%M%S")

        # 儲存圖片（原圖暫存）
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"original_{uuid.uuid4().hex}.tmp")
        file.save(original_path)

        # 轉換為 JPEG 並壓縮
        filename = f"{filename_time}_{uuid.uuid4().hex}.jpg"
        final_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        convert_and_compress_image(original_path, final_path)

        # 刪除暫存檔
        os.remove(original_path)

        # 預測圖片結果
        label, confidence = predict_image(final_path)
        info = class_info[label]

        result_text = (
            f"🕒 上傳時間：{timestamp}\n"
            f"🌿 狀況：{info['status']}\n"
            f"📌 原因：{info['cause']}\n"
            f"🛠️ 建議：{info['solution']}\n"
        )

        BASE_URL = os.getenv("BASE_URL")
        image_url = f"{BASE_URL}/static/uploads/{filename}"

        # 傳送 LINE 訊息
        try:
            if not USER_ID:
                print("❌ USER_ID 尚未設定，請確認 .env 檔案內容")
            else:
                line_bot_api.push_message(USER_ID, [
                    TextSendMessage(text=result_text),
                    ImageSendMessage(original_content_url=image_url, preview_image_url=image_url)
                ])
        except Exception as e:
            print("❌ LINE 傳送失敗：", e)

        return render_template('result.html', result=result_text, image_path=image_url)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
