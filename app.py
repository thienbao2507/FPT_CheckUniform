from flask import Flask, request, render_template, send_file
from process import run_inference
import os

app = Flask(__name__)
UPLOAD_FOLDER = "test"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    if not file:
        return "❌ Không có file nào được tải lên.", 400

    # Lưu ảnh tạm vào thư mục test/
    upload_path = os.path.join(UPLOAD_FOLDER, "saiDongPhuc.jpg")
    file.save(upload_path)

    # Gọi xử lý
    output_path, results = run_inference(upload_path)

    # Trả kết quả cho giao diện
    return render_template("result.html", img_path="/result", results=results)

@app.route('/result')
def result_image():
    result_path = "check/test_result.jpg"
    if not os.path.exists(result_path):
        return "❌ Ảnh kết quả chưa được tạo!", 404
    return send_file(result_path, mimetype='image/jpeg')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render sẽ gán biến PORT
    app.run(host='0.0.0.0', port=port, debug=False)

