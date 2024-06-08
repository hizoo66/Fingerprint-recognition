import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
from keras.models import load_model


def SequentialImage(image):
    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 0.5)),
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-30, 30),
            order=[0, 1],
            cval=255
        )
    ], random_order=True)
    return seq.augment_image(image)

def preprocess_img(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((90, 90))
    img_array = np.array(image)
    
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    
    img_array = img_array / 255.0
    img_array = SequentialImage(img_array)
    img_array = np.reshape(img_array, (1, 90, 90, 1))
    img_array = img_array.astype(np.float32) / 255.0
    return img_array

def predict_result(img,model, match_key):
    x_real = np.load('x_real.npz')['data']
    y_real = np.load('y_real.npy')
    label_real_dict = {''.join(y.astype(str)).zfill(6): i for i, y in enumerate(y_real)}

    if match_key not in label_real_dict:
        return None, None, None

    real_x = x_real[label_real_dict[match_key]].reshape((90, 90, 1)).astype(np.float32) / 255.0
    prediction = model.predict([img, real_x.reshape((1, 90, 90, 1))])

    result = prediction[0] * 100
    result_img = real_x
    result_index = label_real_dict[match_key]

    return result, result_index, result_img