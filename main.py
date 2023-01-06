from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import numpy
from PIL import Image
from imageColorExtractor import find_colors


app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/image-uploaded', methods=['GET', 'POST'])
def image_uploaded():
    if request.method == 'POST':

        # Processing the Image
        raw_uploaded_image = request.files['image_input']
        uploaded_image = Image.open(raw_uploaded_image)
        array_uploaded_image = numpy.asarray(uploaded_image)

        # Get user selected number of colours from the frontend
        input_number_of_colours = int(request.form['number_of_colors'])

        color_hex_codes, color_rgb_codes = find_colors(image=array_uploaded_image,
                                                       number_of_colors=input_number_of_colours,
                                                       display_color_chart=False)

        return render_template('show-colours.html',
                               color_hex_codes_in_html=color_hex_codes,
                               number_of_colours_in_html=input_number_of_colours)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
