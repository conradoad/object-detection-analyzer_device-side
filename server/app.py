from flask import Flask, request, jsonify
import flask
from flask_cors import CORS
import os
import argparse
import cv2 as cv
import numpy as np
import time
import base64

from classes import Image
from utils.decorators import duration


UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = {'bmp', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app)

image = Image()

global cascade
cascade=None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/check_image_loaded', methods=['POST'])
def check_image_loaded():
    image_uuid = request.json['image_uuid']
    if image.uuid == None:
        image.uuid = image_uuid
        return 'false'
    elif image.uuid != image_uuid:
        image.uuid = image_uuid
        return 'false'
    elif image.uuid == image_uuid:
        return 'true'


@app.route('/upload_image', methods=['POST'])
def upload_image():
    image_data = request.files['image_data']

    if image_data and allowed_file(image_data.filename):
        image.imageLocation = os.path.join(
            app.config['UPLOAD_FOLDER'], image_data.filename)
        image_data.save(image.imageLocation)

        return {'status': 'ok', 'msg': 'Image has uploaded'}

    else:
        return {'status': 'nok', 'msg': 'Some problem has ocurred. The image was not uploaded.'}


@app.route('/detection_analysis', methods=['POST'])
def run_detection_analysis():
    params = request.json['params']

    cv_image = getImage()

    global cascade
    if cascade == None:
        cascade = loadCascade(params["model"])

    if not cascade:
        return {'isOk': 'false'}

    result = objectDetectionAnalysis(cv_image, cascade, params)

    res = flask.json.dumps(result)

    return res


def loadCascade(model):
    modelFile = None
    if model == '1':
        modelUrl = "./server/assets/pretrainedModels/haarcascade_frontalface_alt.xml"
    else:
        return None

    # CRIA OBJETOS FACE CASCADE
    parser = argparse.ArgumentParser(
        description='Code for Cascade Classifier.')
    parser.add_argument('--model_cascade', help='Path to model cascade.',
                        default=modelUrl)
    args, unknown = parser.parse_known_args()
    cascade_name = args.model_cascade
    cascade = cv.CascadeClassifier()
    # Load the cascades
    if not cascade.load(cascade_name):
        print('--(!)Error loading face cascade')
        exit(0)

    return cascade

def getImage():
    cv_image = cv.imread(
        app.config['UPLOAD_FOLDER'] + '/' + image.imageLocation)
    return cv_image


def objectDetectionAnalysis(image, cascade, params):
    IMG_WIDTH = image.shape[1]
    IMG_HEIGHT = image.shape[0]

    minNeighbors = params["minNeighbors"]
    numberOfSamples = params["numberOfSamples"]

    scaleFactorFrom = params["scaleFactor_x_axis"]["from"]
    scaleFactorTo = params["scaleFactor_x_axis"]["to"]
    scaleFactorSteps = params["scaleFactor_x_axis"]["steps"]
    scaleFactors = np.linspace(scaleFactorFrom, scaleFactorTo, scaleFactorSteps)
    
    for i, x in enumerate(scaleFactors):
        scaleFactors[i] = round(x, 3)
        
    minSizeFrom = params["minSize_y_axis"]["from"]
    minSizeTo = params["minSize_y_axis"]["to"]
    minSizeSteps = params["minSize_y_axis"]["steps"]
    minSizes = np.linspace(minSizeFrom, minSizeTo, minSizeSteps)
    
    for i, x in enumerate(minSizes):
        minSizes[i] = round(x, 4)

    imgGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #detectionResultMatrix = [[{}] * minSizeSteps] * scaleFactorSteps
    detectionResultMatrix = [
        [{} for i in range(minSizeSteps)] for j in range(scaleFactorSteps)]

    for x, scaleFactor in enumerate(scaleFactors):
        for y, minSize in enumerate(minSizes):
            minSizeAbs = int(IMG_WIDTH * minSize)
            deltaTimeAcc = 0
            detectedObjects = []
            params_cmd= {"scaleFactor":scaleFactor, "minNeighbors":minNeighbors, "minSizeAbs":minSizeAbs}
            
            for _ in range(numberOfSamples):
                (detections, rejectLevels, levelWeight ), dt = duration(objectDetection)(imgGray, cascade, params_cmd)
                deltaTimeAcc += dt


            for idx, detection in enumerate(detections):
                detectedObj = {
                    # "levelWeigth": levelWeight[idx][0].round(5),
                    "perc_x": np.round(detection[0]/IMG_WIDTH, 7),
                    "perc_y": np.round(detection[1]/IMG_HEIGHT, 7),
                    "perc_w": np.round(detection[2]/IMG_WIDTH, 7),
                    "perc_h": np.round(detection[3]/IMG_HEIGHT, 7)
                }
                detectedObjects.append(detectedObj)

            meanTime = deltaTimeAcc/numberOfSamples
            meanLevelWeight = np.average(levelWeight).round(5) \
                if len(levelWeight) > 0 \
                else 0

            singleDetectionResult = {
                "scaleFactor_x_axis": scaleFactor,
                "minSize_y_axis": minSize,
                "meanDetectionTime": np.round(meanTime, 3),
                "meanLevelWeight": meanLevelWeight,
                "detectedFaces": detectedObjects
            }

            detectionResultMatrix[x][y] = singleDetectionResult

    print('Analysis complete')
    return {"xAxis": scaleFactors.tolist(), "yAxis": minSizes.tolist(), "resultMatrix": detectionResultMatrix}


def objectDetection(image, cascade, params):
    detections, rejectLevels, levelWeight = \
                    cascade.detectMultiScale3(image,
                                              scaleFactor=params["scaleFactor"],
                                              minNeighbors=int(params["minNeighbors"]),
                                              minSize=(params["minSizeAbs"], params["minSizeAbs"]),
                                              outputRejectLevels=True)
    return (detections, rejectLevels, levelWeight)


def buildBase64OriginalImage(image):    
    rawOriginalImageSize = image.size
    retval, bf = cv.imencode('.bmp', image)
    originalImageEncoded = base64.b64encode(bf).decode('utf-8')
    encodedOriginalImageSize = len(originalImageEncoded)
    
    return originalImageEncoded, rawOriginalImageSize, encodedOriginalImageSize
    

def buildBase64ImageArray(image, detectedFaces):
    encodedImages = []
    totalRawImagesSizes = 0
    groupedEncodedImagesSize = 0
    for detectedFace in detectedFaces:
        x = detectedFace[0]
        y = detectedFace[1]
        w = detectedFace[2]
        h = detectedFace[3]

        faceImg = image[y:(y+h), x:(x+w)]
        totalRawImagesSizes += faceImg.size

        retval, bf = cv.imencode('.bmp', faceImg)
        imgEncoded = base64.b64encode(bf).decode('utf-8')
        groupedEncodedImagesSize += len(imgEncoded)
        encodedImages.append(imgEncoded)
    return encodedImages, totalRawImagesSizes, groupedEncodedImagesSize


@app.route('/metrics', methods=['POST'])
def metrics():
    params = request.json['params']
    numberOfSamples = params['numberOfSamples']
    metrics = {}

    global cascade
    if cascade == None:
        cascade = loadCascade('1')

    metrics['2.1 - Loading Image (s)'] = 0
    for _ in range(numberOfSamples):
        image, dt = duration(getImage)()
        metrics['2.1 - Loading Image (s)'] += dt
    metrics['2.1 - Loading Image (s)'] = round(metrics['2.1 - Loading Image (s)']/numberOfSamples, 3)
    
    IMG_WIDTH = image.shape[1]
    IMG_HEIGHT = image.shape[0]

    params['minSizeAbs'] = int(IMG_WIDTH * params["minSize"])

    metrics['1.0 - Params'] = "Min. Size Face: " + str(round(params["minSize"]*100, 3)) + " / Scale Factor: " + str(round(params["scaleFactor"], 3)) + " / Min. Neighbors: " + str(params["minNeighbors"])
    metrics['1.1 - Full Image Resolution'] = str(IMG_HEIGHT) + " x " + str(IMG_WIDTH)
    metrics['1.2 - Full Image Size (bytes)'] = '{0:,}'.format(image.size)

    metrics['2.2 - Convert Image to Gray (s)'] = 0
    metrics['2.3 - Detection (s)'] = 0
    metrics['2.4 - Build Encoded Faces Images (s)'] = 0
    
    for _ in range(numberOfSamples):
        imgGray, dt = duration(cv.cvtColor)(image, cv.COLOR_BGR2GRAY)
        metrics['2.2 - Convert Image to Gray (s)'] += dt
    
        (detectedFaces, _, _), dt = duration(objectDetection)(imgGray, cascade, params)
        metrics['2.3 - Detection (s)'] += dt
    
        (encodedImagesArray, totalRawImagesSizes, groupedEncodedImagesSize), dt = duration(buildBase64ImageArray)(image, detectedFaces)
        metrics['2.4 - Build Encoded Faces Images (s)'] += dt

        
    metrics['2.2 - Convert Image to Gray (s)'] = round(metrics['2.2 - Convert Image to Gray (s)'] / numberOfSamples, 3)
    metrics['2.3 - Detection (s)'] = round(metrics['2.3 - Detection (s)'] / numberOfSamples, 3)
    metrics['2.4 - Build Encoded Faces Images (s)'] = round(metrics['2.4 - Build Encoded Faces Images (s)'] / numberOfSamples, 3)
    
    metrics['1.4 - Number of detected faces'] = detectedFaces.shape[0]
    
    
    (encodedOriginalImage, rawOriginalImageSize, encodedOriginalImageSize) = buildBase64OriginalImage(image)
    metrics['1.3 - Full Image Encoded Size (bytes)'] = '{0:,}'.format(encodedOriginalImageSize)
    
    metrics['2.5 - Total execution time (s)'] = round(metrics['2.1 - Loading Image (s)'] + metrics['2.2 - Convert Image to Gray (s)'] + metrics['2.3 - Detection (s)'] + metrics['2.4 - Build Encoded Faces Images (s)'],3)

    metrics['1.5 - Cropped Faces Images Total Size (bytes) / (% from full image size)'] = '{0:,}'.format(totalRawImagesSizes) + " / " + str(round((totalRawImagesSizes/image.size)*100, 1)) + "%"
    metrics['1.6 - Encoded Faces Images Total Size (bytes) / (% from full image encoded size)'] = '{0:,}'.format(groupedEncodedImagesSize) + " / " + str(round((groupedEncodedImagesSize/encodedOriginalImageSize)*100, 1)) + "%"
    metrics['3.1 - Faces images'] = encodedImagesArray

    print('Metrics was sent')

    return metrics
