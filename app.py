import os
import io
import base64
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from models import db, InformTable
from imgaug import augmenters as iaa
#from pred import preprocess_img, predict_result
from werkzeug.utils import secure_filename
from keras.models import load_model
import matplotlib.pyplot as plt
import sqlite3
import tensorflow as tf 
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

model_path = './fingerprintTest_original.h5'
model = load_model(model_path)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
basedir = os.path.abspath(os.path.dirname(__file__))
dbfile = os.path.join(basedir, 'db.sqlite')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + dbfile
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'jqiowejrojzxcovnklqnweiorjqwoijroi'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db.init_app(app)

with app.app_context():
    db.create_all()



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    return render_template('hello.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        personid = request.form['name']
        gender = request.form['gender']
        lr = request.form['lr']

        new_entry = InformTable(personid, gender, lr)
        db.session.add(new_entry)
        db.session.commit()
        flash('Registration successful!')
        return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        data = file.read()

        try:
            # 데이터베이스에 연결
            conn = sqlite3.connect(dbfile)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO fingerprint (name, data) VALUES (?, ?)", (filename, data))
            conn.commit()
            cursor.close()
            conn.close()
            flash('File successfully uploaded and saved to database')
        except sqlite3.Error as e:
            flash(f"Error: {str(e)}")
        return redirect(url_for('index'))
    else:
        flash('Allowed file types are bmp, png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/prediction', methods=['POST'])
def predict_image_file():
    if 'file' not in request.files:
        return "파일이 없습니다."
    
    f = request.files['file']
    if f.filename == '':
        return "파일 이름이 없습니다."
    
    filename = secure_filename(f.filename)
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(upload_dir)

    name = filename[:-4]
    num, etc = name.split('__')
    gender, lr, finger, _ = etc.split('_')

    gender = 0 if gender == 'M' else 1
    lr = 0 if lr == 'Left' else 1

    finger_map = {'thumb': 0, 'index': 1, 'middle': 2, 'ring': 3, 'little': 4}
    finger = finger_map.get(finger, -1)
    
    if finger == -1:
        return "잘못된 손가락 정보입니다."

    info = np.array([num, gender, lr, finger], dtype=np.uint16)
    match_key = ''.join(info.astype(str)).zfill(6)

    seq = iaa.Sequential([
    # blur images with a sigma of 0 to 0.5
    iaa.GaussianBlur(sigma=(0, 0.5)),
    iaa.Affine(
        # scale images to 90-110% of their size, individually per axis
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        # translate by -10 to +10 percent (per axis)
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        # rotate by -30 to +30 degrees
        rotate=(-30, 30),
        # use nearest neighbour or bilinear interpolation (fast)
        order=[0, 1],
        # if mode is constant, use a cval between 0 and 255
        cval=255
    )
], random_order=True)
    
    #img = preprocess_img(upload_dir)
    
    #pred, num, result_image = predict_result(img, model, match_key)
    x_real = np.load('./dataset/x_real.npz')['data']
    y_real = np.load('./dataset/y_real.npy')
    label_real_dict = {''.join(y.astype(str)).zfill(6): i for i, y in enumerate(y_real)}

    if match_key not in label_real_dict:
        return render_template("result.html", predictions=str(0), image_path=upload_dir, img_str='None', label=name)

    else:
        random_img = x_real[label_real_dict[match_key]]
        img = seq.augment_image(random_img).reshape((1, 90, 90, 1)).astype(np.float32) / 255.

        real_x = x_real[label_real_dict[match_key]].reshape((1, 90, 90, 1)).astype(np.float32) / 255.0
        prediction = model.predict([img, real_x])

        #pred = prediction[0]
        result_image = real_x
        pred = prediction[0][0]
        #pred = round(pred, 3)  # 소수점 셋째 자리까지만

        # 터미널에 pred 값 출력
        print(f"Prediction: {pred}")
        #print(f"Prediction: {pred}")
        #result_index = label_real_dict[match_key]


        if pred is None:
            return "일치하는 결과가 없습니다."

        # Matplotlib 작업
        buffer = io.BytesIO()
        plt.figure(figsize=(2, 2))
        plt.imshow(result_image.squeeze(), cmap='gray')
        plt.axis('off')
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()

        return render_template("result.html", predictions=str(pred), image_path=upload_dir, img_str=img_str, label=name)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port=5000)