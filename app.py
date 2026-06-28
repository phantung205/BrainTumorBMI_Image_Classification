from flask import render_template,request,Flask,send_from_directory
from src import config
import os
from datetime import datetime
from deploy.load_model import load_model
from deploy.inference import predict_image

#load model
model,device = load_model(config.checkpoint_path)

# tạo thư mục lưu file người dùng upload
upload = config.upload_folder
os.makedirs(upload, exist_ok=True)


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html",result=None,image_name=None,error=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        image = request.files["image"]

        # kiểm tra xem người dugnf đã up ảnh chưa
        if image.filename == "":
            raise ValueError("Bạn chưa chọn ảnh.")

        # lấy ra tên đuôi file người dùng up
        name, ext = os.path.splitext(image.filename)

        # kiểm tra đuôi file có đúng ko
        ext = ext.lower()
        allowed_extensions = {".jpg",".jpeg",".png"}
        if ext not in allowed_extensions:
            raise ValueError("Chỉ chấp nhận các định dạng ảnh: .jpg, .jpeg, .png")

        # lấy ra thời gian hiện tại tên mới
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{name}_{timestamp}{ext}"

        # lưu ảnh vào thư mục upload
        input_path = os.path.join(upload,new_filename)
        image.save(input_path)

        result = predict_image(model,device,input_path)

        return render_template(
            "index.html",
            result=result,
            image_name=new_filename,
            error=None
        )

    except Exception as e:

        return render_template(
            "index.html",
            result=None,
            image_name=None,
            error=str(e)
        )

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(upload,filename)


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )