import os
import uuid
import requests
from whitenoise import WhiteNoise

from flask import (Flask, flash, redirect, render_template, request,
                   send_from_directory, url_for)

from src import sample


UPLOAD_FOLDER = './static/images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
YANDEX_API_KEY = 'YOUR API KEY HERE'
SECRET_KEY = 'YOUR SECRET KEY FOR FLASK HERE'

app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY

@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)

# check if file extension is right
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# force browser to hold no cache. Otherwise old result returns.
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# main directory of programme
@app.route('/caption-it/', methods=['GET', 'POST'])
def upload_file():
    try:
        # remove files created more than 5 minute ago
        os.system("find static/images/ -maxdepth 1 -mmin +5 -type f -delete")
    except OSError:
        pass
    if request.method == 'POST':
        # check if the post request has the file part
        if 'content-file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        content_file = request.files['content-file']
        files = [content_file]
        # give unique name to each image
        content_name = str(uuid.uuid4()) + ".png"
        file_names = [content_name]
        for i, file in enumerate(files):
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_names[i]))
        args={
            'image' : "static/images/" + file_names[0],
            'model_path': 'src/model/deploy_model_cpu.pth.tar', 
            'vocab_path': 'src/vocab/vocab.json', 
            'embed_size': 256,
            'hidden_size': 512,
            'num_layers': 1,
        }        
        # returns created caption
        caption = sample.main(args)
        try:
            r = requests.post('https://translate.yandex.net/api/v1.5/tr.json/translate',
                data = {'key': YANDEX_API_KEY, 
                'text':str(caption), 'lang':'en-tr'})
            tr_caption = r.json()['text'][0]
        except:
            tr_caption = ''
        params={
            'content': "static/images/" + file_names[0],
            'caption': caption,
            'tr_caption': tr_caption,
        }
        return render_template('success.html', **params)
    return render_template('upload.html')


@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404


if __name__ == "__main__":
    app.run(host='0.0.0.0')
