import sys
import argparse
from yolo import YOLO
from PIL import Image
import cv2
import numpy as np
from timeit import default_timer as timer

def output_video(vid):
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if FLAGS.output:
        print("!!! TYPE:", type(FLAGS.output), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(FLAGS.output, video_FourCC, video_fps, video_size)
    return out 

def process_frames(cam, output_video = None):
    yolo = YOLO()
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        ret_val, image = cam.read()
        image = Image.fromarray(image)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(13, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.00, color=(255, 100, 100), thickness=2)
        #cv2.namedWindow("Live YoloV3", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if output_video:
            output_video.write(result)
    yolo.close_session()
    
    
def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False, default="",
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    parser.add_argument(
        "--webcam", action="store_true", default=False,
        help = "Detect direct from webcam"
    )

    parser.add_argument(
        "--streaming", action="store_true", default=False,
        help = "Detect direct from streaming, need --url parameter"
    )

    parser.add_argument(
        "--url", nargs='?', type=str, default="",
        help = "Input url of streaming"
    )

    FLAGS = parser.parse_args()

    if FLAGS.webcam:
        cam = cv2.VideoCapture(0)
        process_frames(cam)
    
    if FLAGS.streaming:
        cam = cv2.VideoCapture()
        cam.open(FLAGS.url) 
        if not FLAGS.url:
            print("Must specify url to use streaming feature.")
        #cam.open("http://192.168.1.4:8000/stream.mjpeg") my ip, ignore it, i'm lazy
        process_frames(cam)
 
    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        out = None
        vid = cv2.VideoCapture(FLAGS.input)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        if FLAGS.output:
            out = output_video(vid)
        process_frames(vid, out) 
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
