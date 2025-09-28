import torch
import torch.backends.cudnn as cudnn
import numpy as np
from utils import align_face
from Retina.Retinadata import cfg_mnet, cfg_re50
from Retina.Retinalayers.functions.prior_box import PriorBox
from Retina.Retinautils.nms.py_cpu_nms import py_cpu_nms
import cv2
from Retina.Retinamodels.retinaface import RetinaFace
from Retina.Retinautils.box_utils import decode, decode_landm
import time
from utils import resize_with_padding

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class FaceFeatures:
    def __init__(self):
        cpu = False
        modelPath = "Retina/weights/mobilenet0.25_Final.pth"
        
        self.treshold = 0.2
        self.top_K = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = 750
        
        self.cfg = cfg_mnet
        self.net = RetinaFace(cfg=self.cfg, phase = 'test')
        self.net = load_model(self.net, modelPath, cpu)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = self.net.to(self.device)
        
        self.resize=1
        
    def faceAlign(self, image, bbox, landmarks):
        x1, y1, x2, y2 = np.array(bbox, dtype=np.int32)
        facetoalign = image[y1:y2, x1:x2]
        keypoints = np.array(landmarks).reshape((5, 2))      
        facial_landmarks = np.zeros((5, 2))  # five keypoints (x, y)
        for i in range(5):
            facial_landmarks[i] = [keypoints[i][0] - x1, keypoints[i][1] - y1]
            
        aligned_face = align_face(facetoalign, facial_landmarks)
        
        return aligned_face
        
    def preprocess(self, images):
        #verify if images is a single image or a list of images
        torchimages = []
        scales = []
        widths = []
        heights = []
        
        if type(images) == list:
            for index, image in enumerate(images):
                image = resize_with_padding(image, (256,256))
                scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                scales.append(scale)
                heights.append(image.shape[0])
                widths.append(image.shape[1])
                images[index] = image
                image = np.array(image, dtype=np.int64)
                image -= (104, 117, 123)
                image = image.transpose(2, 0, 1)
                torchimages.append(image)
                
            torchimages = torch.tensor(np.array(torchimages),dtype=torch.float32).to(self.device)
        
        else:
            scale = torch.Tensor([images.shape[1], images.shape[0], images.shape[1], images.shape[0]])
            scales.append(scale)
            heights.append(images.shape[0])
            widths.append(images.shape[1])
            images -= (104, 117, 123)
            images = images.transpose(2, 0, 1)
            torchimages = torch.from_numpy(images).unsqueeze(0).to(self.device)
        
        
        return torchimages, scales
    
    def postprocess(self, boxes, scores, landms):
        results = []
        
        for batch, score in enumerate(scores):
            # ignore low scores
            inds = np.where(score > self.treshold)[0]
            bboxes = boxes[batch][inds]
            land = landms[batch][inds]
            score = scores[batch][inds]

            # keep top-K before NMS
            order = score.argsort()[::-1][:self.top_K]
            bboxes = bboxes[order]
            land = land[order]
            score = score[order]

            # do NMS
            dets = np.hstack((bboxes, score[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            land = land[keep]

            # keep top-K faster NMS
            dets = dets[:self.keep_top_k, :]
            land = land[:self.keep_top_k, :]

            dets = np.concatenate((dets, land), axis=1)
            results.append(dets)
        return results
    
    def getHiscoreface(self, dets):
        HiScore = -1.0
        for detection in dets:
            confidence = detection[4]
            if (confidence > self.treshold) and (confidence > HiScore):
                HiScore = confidence
                HiScoredet =  detection
        return HiScoredet
        
        
    def run_faceDet(self, crop):
        try:
            preprocessed, scale = self.preprocess(crop)

            loc, conf, landms = self.net(preprocessed)
            #print detection time in ms
            
            priorbox = PriorBox(self.cfg, image_size=(preprocessed.shape[2], preprocessed.shape[3]))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data.expand(loc.size(0),-1,-1)
            boxes = decode(loc.data, prior_data, self.cfg['variance'])
            boxes = boxes.cpu()
            boxes = (boxes * scale[0] / self.resize).numpy()
            scores = conf.data.cpu().numpy()[:, :, 1]
            landms = decode_landm(landms.data, prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([preprocessed.shape[3], preprocessed.shape[2], preprocessed.shape[3], preprocessed.shape[2],
                                preprocessed.shape[3], preprocessed.shape[2], preprocessed.shape[3], preprocessed.shape[2],
                                preprocessed.shape[3], preprocessed.shape[2]])
            scale1 = scale1.to(self.device)
            landms = (landms * scale1 / self.resize).cpu().numpy()
            toreturn = []
            results = self.postprocess(boxes, scores, landms)
            for index, dets in enumerate(results):
                if len(dets) == 0:
                    toreturn.append([])
                    continue
                hiScoreface = self.getHiscoreface(dets)
            
                bbox = np.array(hiScoreface[:4], dtype=np.int32).tolist()
                conf = hiScoreface[4]
                landmarks = hiScoreface[5:]
                face = crop[index][bbox[1]:bbox[3], bbox[0]:bbox[2]]
                try:
                    aligned_face = self.faceAlign(crop[index], bbox, landmarks)
                except:
                    aligned_face = face
            
                toreturn.append([aligned_face, bbox, conf])
            return toreturn
        except:
            return None    
        