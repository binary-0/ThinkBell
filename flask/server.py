import object_detection_api
import VoiceActivityDetection
from multiprocessing import Process, Value
import threading
import os
from PIL import Image
from flask import Flask, request, Response


app = Flask(__name__)

global cvResult
global vadResult
# import ssl
# ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
# ssl_context.load_cert_chain(certfile='server.crt', keyfile='server.key', password='samzzang18')

# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response


@app.route('/')
def index():
    global vadResult
    vadResult =Value('i', 0)

    # vadResult=threading.Thread(target=VoiceActivityDetection.start_recording).start()
    # gen_frame_thread.start()
    # VoiceActivityDetection.start_recording()
    p=Process(target=VoiceActivityDetection.start_recording, args=(vadResult, ))
    p.start()

    return Response('Voice detection On')


# @app.route('/local')
# def local():
#     return Response(open('./static/local.html').read(), mimetype="text/html")


@app.route('/image', methods=['POST'])
def image():
    try:
        image_file = request.files['image']  # get the image

        # Set an image confidence threshold value to limit returned data
        threshold = request.form.get('threshold')
        if threshold is None:
            threshold = 0.5
        else:
            threshold = float(threshold)

        # finally run the image through tensor flow object detection`
        image_object = Image.open(image_file)
        objects = object_detection_api.get_objects(image_object, threshold)
        print(image_file)

        global vadResult
        print(vadResult.value)


        global cvResult
        cvResult=objects
        return objects

    except Exception as e:
        print('POST /image error: %e' % e)
        return e



@app.route('/teacher', methods=['GET'])
def send2teacher():
    global cvResult
    return cvResult

if __name__ == '__main__':
	# without SSL

    app.run(debug=True, host='0.0.0.0')

	# with SSL
    #app.run(debug=True, host='0.0.0.0', ssl_context=('ssl/server.crt', 'ssl/server.key'))
