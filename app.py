from flask import Flask, request, render_template, send_file
from io import BytesIO
import snowcode
import pdb

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/snowcode', methods=['GET', 'POST'])
def render_snowcode():
    if request.method == 'POST':
        if request.form['type'] == "payload":
            img = snowcode.text_to_snowcode(request.form['payload'])
            if img:
                return serve_pil_image(img)
            else: render_template('snowcode.html', result="error")
        elif request.form['type'] == "user_img":
            text = snowcode.snowcode_to_text(request.files['user_img'])
            text = text if text else "error"
            return render_template('snowcode.html', result=text)
    else:
        return render_template('snowcode.html')

# serving images without temp files - https://stackoverflow.com/questions/7877282/how-to-send-image-generated-by-pil-to-browser

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'PNG', quality=100)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')