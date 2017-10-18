from flask import Flask, render_template, redirect, url_for, send_from_directory, request
from flask_bootstrap import Bootstrap
from PIL import Image
from werkzeug.utils import secure_filename
import os

import numpy as np
import cntk
from cntk import load_model
import sys
sys.path.append("..")
from FasterRCNN_train import prepare, train_faster_rcnn, store_eval_model_with_native_udf
from FasterRCNN_eval import compute_test_set_aps, FasterRCNN_Evaluator
sys.path.append("../utils")
from utils.config_helpers import merge_configs
from utils.plot_helpers import plot_test_set_results, plot_test_file_results

def get_configuration():
    from FasterRCNN_config import cfg as detector_cfg
    from utils.configs.VGG16_config import cfg as network_cfg
    from utils.configs.Building100_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg])

app = Flask(__name__)
Bootstrap(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
images_directory = os.path.join(APP_ROOT, 'images')
thumbnails_directory = os.path.join(APP_ROOT, 'thumbnails')
if not os.path.isdir(images_directory):
    os.mkdir(images_directory)
if not os.path.isdir(thumbnails_directory):
    os.mkdir(thumbnails_directory)

@app.route('/')
def index():

    path = os.path.join(os.path.abspath(__file__), 'thumbnails')
    print(path)
    thumbnail_names = os.listdir(thumbnails_directory)
    #return render_template('gallery.html', )
    return render_template('index.html',thumbnail_names=thumbnail_names)

@app.route('/result/<filename>')
def result(filename):

    path = os.path.join(os.path.abspath(__file__), 'thumbnails')
    print(path)
    thumbnail_names = os.listdir(thumbnails_directory)
    #return render_template('gallery.html', )
    return render_template('result.html',thumbnail_names=thumbnail_names, image_result= filename)

@app.route('/thumbnails/<filename>')
def thumbnails(filename):
    return send_from_directory('thumbnails', filename)

@app.route('/images/<filename>')
def images(filename):
    return send_from_directory('images', filename)

@app.route('/public/<path:filename>')
def static_files(filename):
    return send_from_directory('./public', filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        for upload in request.files.getlist('images'):
            filename = upload.filename
            # Always a good idea to secure a filename before storing it
            filename = secure_filename(filename)
            # This is to verify files are supported
            ext = os.path.splitext(filename)[1][1:].strip().lower()
            if ext in set(['jpg', 'jpeg', 'png']):
                print('File supported moving on...')
            else:
                return render_template('error.html', message='Uploaded files are not supported...')
            destination = '/'.join([images_directory, filename])
            # Save original image

            upload.save(destination)
            print('start detection')
            img_path = plot_test_file_results(evaluator, destination, images_directory, cfg)
            print('done detection')
            # Save a copy of the thumbnail image

            print(img_path)
            #image = Image.open(destination)
            image = Image.open(img_path)

            if image.mode != "RGB":
                image = image.convert("RGB")
            
            print('image_opened')
            image.thumbnail((300, 170))

            result_filename = img_path.split('/')[-1]
            image.save('/'.join([thumbnails_directory, result_filename]))
        #return redirect(url_for('index'))
        return redirect(url_for('result', filename = result_filename))
    return render_template('index.html')

if __name__ == '__main__':

    cfg = get_configuration()
    prepare(cfg, False)
    #cntk.device.try_set_default_device(cntk.device.gpu(cfg.GPU_ID))
    cntk.device.try_set_default_device(cntk.device.cpu())
    model_path = cfg['MODEL_PATH']
    print(model_path)

    if os.path.exists(model_path) and cfg["CNTK"].MAKE_MODE:
        print("Loading existing model from %s" % model_path)
        eval_model = load_model(model_path)
    else:
        print("No trained model found.")
        exit()

    # Plot results on test set images
    results_folder = os.path.join(cfg.OUTPUT_PATH, cfg["DATA"].DATASET)
    evaluator = FasterRCNN_Evaluator(eval_model, cfg)

    app.run(host='0.0.0.0', port=os.environ.get('PORT', 3000))
