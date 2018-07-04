import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/rockstar/workspace/caffe/'
#caffe_root = '/opt/movidius/caffe/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  


model_dir = 'trained_net/'
#model_dir = 'orgnl-model/'
net_file= model_dir + 'MobileNetSSD_deploy.prototxt'
caffe_model= model_dir + 'MobileNetSSD_deploy.caffemodel'
test_dir = "../data/ofo_test"
imgout_dir = "images-output/"

if not os.path.exists(caffe_model):
    print('caffe_model path:', caffe_model)
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

'''
CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('background',
           'ofo', 'mobike', 'kuqi', 'bluegogo',
           'youon', 'hellobike', 'reddy go', 'xiaoqiang', 'xiaoming',
           'other', 'UniBike')
'''
CLASSES = ('background', 'ofo', 'mobike')

def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)


cap = cv2.VideoCapture('mobike.mov')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
v_out = cv2.VideoWriter('mobike-out.avi', fourcc, 25.0, (1280, 720))

idx = 0
while cap.isOpened():
    ret, origimg = cap.read()

    img = preprocess(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
        if CLASSES[int(cls[i])] != 'mobike' or conf[i] <= 0.98:
            continue
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        if int(p2[0]) - int(p1[0]) >= 1240 and int(p2[1]) - int(p1[1]) >= 700:
            continue
        if idx < 25:
            idx += 1
            continue
        cv2.rectangle(origimg, p1, p2, (0,255,0), 2)
        #cv2.rectangle(origimg, p1, p2, (0,255,0))
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        print(title, p1, p2)
        #cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 3, (0, 255, 0), 3)
        #cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 1, (0, 255, 0), 1)
    #cv2.imwrite(imgout_dir + imgfile.split('/')[1], origimg)
    #h, w = origimg.shape[:2]
    #cv2.namedWindow('mobike', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('mobike', w, h)
    #cv2.imshow('mobike', origimg)
    v_out.write(origimg)

    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
v_out.release()
cv2.destroyAllWindows()
