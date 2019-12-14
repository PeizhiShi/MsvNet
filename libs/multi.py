import numpy as np
import libs.binvox_rw
from skimage import measure
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import torch
import selectivesearch
import cupy as cp
import glob
import os
import random
from scipy.special import softmax
from torchvision import transforms, utils
import tensorflow as tf
import libs.featurenetmodel as lfmd
from pathlib import Path
from PIL import Image
import torch.nn as nn
import libs.vis as vis


def load_3dmodel(fidx,sidx):
    modelfilename = 'data/set' +str(sidx)+ '/'+str(fidx) + '.binvox'
            
    with open(modelfilename, 'rb') as f:
        sample = libs.binvox_rw.read_as_3d_array(f).data
        
    
    return sample



    
def featurenet_segmentation(sample):
    
    blobs = ~sample

    final_labels = np.zeros(blobs.shape)
    
    all_labels = measure.label(blobs)
    
    for i in range(1,np.max(all_labels)+1):
        mk = (all_labels==i)
        distance = ndi.distance_transform_edt(mk)
        
        labels = watershed(-distance)
        
        max_val = np.max(final_labels)+1
        idx = np.where(mk)
        
        
        
        final_labels[idx] += (labels[idx] + max_val)
    
    results = get_seg_samples(final_labels)

    results = 2*(results - 0.5)
    
    return results

def get_seg_samples(labels):
    
    samples = np.zeros((0,labels.shape[0],labels.shape[1],labels.shape[2]))
    
    for i in range(1,np.max(labels.astype(int))+1):
        idx = np.where(labels == i)

        
        if len(idx[0]) == 0:
            continue

        
        cursample = np.ones(labels.shape)
        cursample[idx] = 0
        cursample = np.expand_dims(cursample,axis=0)
        samples = np.append(samples,cursample,axis=0)
    
    
    return samples

def soft_nms_pytorch(samples, box_scores, sigma=0.028):

    
    N = samples.shape[0]
    
    dets = np.zeros((N, 6))
    
    for i in range(N):
        idx = np.where(samples[i,:,:,:] == 0)
        #print(idx)
        dets[i,0] = idx[2].min()
        dets[i,1] = idx[1].min()
        dets[i,2] = idx[0].min()
        dets[i,3] = idx[2].max()
        dets[i,4] = idx[1].max()
        dets[i,5] = idx[0].max()
    
    
    
    indexes = torch.arange(0, N, dtype=torch.double).view(N, 1)
    dets = torch.from_numpy(dets).double()
    box_scores = torch.from_numpy(box_scores).double()
        
    dets = torch.cat((dets, indexes), dim=1)
    

    z1 = dets[:, 0]
    y1 = dets[:, 1]
    x1 = dets[:, 2]
    z2 = dets[:, 3]
    y2 = dets[:, 4]
    x2 = dets[:, 5]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)* (z2 - z1 + 1)

    for i in range(N):
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()


        # IoU calculate
        zz1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        yy1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        zz2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 4].to("cpu").numpy(), dets[pos:, 4].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 5].to("cpu").numpy(), dets[pos:, 5].to("cpu").numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        l = np.maximum(0.0, zz2 - zz1 + 1)
        inter = torch.tensor(w * h * l)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        
        
        
        scores[pos:] = weight * scores[pos:]


    
    max_margin = 0
    for i in range(scores.shape[0]-1):
        if scores[i] - scores[i + 1] > max_margin:
            max_margin = scores[i] - scores[i + 1]
            thresh = (scores[i] + scores[i+1])/2
    
    
    keep = dets[:, 6][scores > thresh].int()
    
    

    return keep.to("cpu").numpy()

def test_msvnet(sidx):
    
    
    num_cuts = 12
    
    random.seed(213)
    

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    net = torch.load('models/msvnet.pt')
     
    
    net.eval()
    
    predictions = np.zeros(24)
    truelabels = np.zeros(24)
    truepositives = np.zeros(24)
    
    with torch.no_grad():
        with open(os.devnull, 'w') as devnull:
            for filename in Path('data/set' + str(sidx)+ '/').glob('*.STL'):
                
                filename = os.path.basename(filename)
                idx = int(os.path.splitext(filename)[0])
                
                
                sample = load_3dmodel(idx,sidx)
                
                segs = msvnet_segmentation(sample) 
                
                 
                inputs = get_msv_samples(segs,num_cuts)
                
                inputs = inputs.to(device)
                
                
                outputs = net(inputs)
                m = nn.Softmax(dim=1)
                
                outputs = m(outputs)
                
                val, predicted = outputs.max(1)
                val = val.cpu().numpy()
                predicted = predicted.cpu().numpy()
                

                
                keepidx = soft_nms_pytorch(segs, val)
                predicted = predicted[keepidx]
                

                
                pred = get_lvec(predicted).astype(int)           
                trul = get_lvec(vis.get_label('data/set'+str(sidx)+'/'+str(idx)+'.csv',0)[:,6]).astype(int)
                

                
                print(idx)
                print('Predicted labels:\t',pred)
                print('True labels:\t\t',trul)
                    
    #                        
                tp = np.minimum(pred,trul)
                
                predictions += pred
                truelabels += trul
                truepositives += tp
    

    precision, recall = eval_metric(predictions,truelabels,truepositives)
    
    
    return precision.mean(), recall.mean()

def cal_segs(img_org, rotation):
    
    img = 1-img_org

    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.8, min_size=10)
    
    samples = np.zeros((0,64,64,64))
    
    candidates = set()
    for r in regions:
     
        
        x, y, w, h = r['rect']
        w += 1
        h += 1
       
       
        
        labels = r['labels']
        idx = np.where(img_lbl[y:y+h, x:x+w,3] == labels[0])
        
        for i in range(1, len(labels)):
            idx2 = np.where(img_lbl[y:y+h, x:x+w,3] == labels[i])
            idx = (np.append(idx[0],idx2[0], axis=0),np.append(idx[1],idx2[1], axis=0))
        

        
        tmpimg = np.zeros(img_lbl[y:y+h, x:x+w,0].shape)
        tmpimg[idx] = 1
        idx = np.where(img_lbl[y:y+h, x:x+w,0] == 1)
        tmpimg[idx] = 0
        
        selval = tmpimg.sum()
        
        maskval = h*w-idx[0].shape[0]
        
        if maskval <= 0 or selval/maskval < 0.5:
            continue
        
        idx = np.where(tmpimg == 1)
        minx = idx[1].min()
        maxx = idx[1].max()
        miny = idx[0].min()
        maxy = idx[0].max()
        w = maxx - minx + 1
        h = maxy - miny + 1
        x += minx
        y += miny
#        
        r['rect'] = (x,y,w,h)
        
        if r['rect'] in candidates:
            continue
        
        if w <= 0 or h <= 0 or w/h <= 0.1 or h/w <= 0.1 or w < 6 or h < 6:
            continue
        
        tmpimg = tmpimg[miny:miny+h,minx:minx+w]
        
        all_labels = measure.label(tmpimg)
        if all_labels.max() >= 2:
            continue
        
        cursample = np.ones((64,64,64))
        
        for i in range(y, y+h):
            for j in range(x, x+w):
                if tmpimg[i-y, j-x] ==1:
                    depth = int(64*img_org[i,j,0])
                    cursample[0:depth,i,j] = 0
        
        cursample = rotate_sample(cursample,rotation,True)
        

        
        cursample = np.expand_dims(cursample,axis=0)
        samples = np.append(samples,cursample,axis=0)
        

    
        candidates.add(r['rect'])
        

     
    return samples 

  
    
def create_img(obj3d):
    img = np.zeros((obj3d.shape[1],obj3d.shape[2]))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for d in range(obj3d.shape[0]):
                
                if obj3d[d,i,j] == True:
                    img[i,j] = d/obj3d.shape[0]
                    break
                
                if d == obj3d.shape[0] - 1:
                    img[i,j] = 1
                    break
                
    
    img = np.stack((img,)*3, axis=-1)
    
    
    return img

def rotate_sample(sample,rotation, reverse = False):

    if reverse:
        if rotation == 1:
            sample = cp.rot90(sample, -2, (0,1)).copy()  
        elif rotation == 2:
            sample = cp.rot90(sample, -1, (0,1)).copy()  
        elif rotation == 3:
            sample = cp.rot90(sample, -1, (1,0)).copy()  
        elif rotation == 4:
            sample = cp.rot90(sample, -1, (2,0)).copy()  
        elif rotation == 5:
            sample = cp.rot90(sample, -1, (0,2)).copy() 
    else:
        if rotation == 1:
            sample = cp.rot90(sample, 2, (0,1)).copy()  
        elif rotation == 2:
            sample = cp.rot90(sample, 1, (0,1)).copy()  
        elif rotation == 3:
            sample = cp.rot90(sample, 1, (1,0)).copy()  
        elif rotation == 4:
            sample = cp.rot90(sample, 1, (2,0)).copy()  
        elif rotation == 5:
            sample = cp.rot90(sample, 1, (0,2)).copy() 
        
    
    return sample

    
def msvnet_segmentation(sample): 
    
    allsamples = np.zeros((0,sample.shape[0],sample.shape[1],sample.shape[2]))
    

    
    for i in range(0,6):
        cursample = sample.copy()
        cursample = rotate_sample(cursample,i)
        
        
        img = create_img(cursample)
        cursample = cal_segs(img,i)
        allsamples = np.append(allsamples,cursample,axis=0)
        
        
    return allsamples

def test_featurenet(sidx):
    
    
    resolution = 64
    
    x=tf.placeholder(tf.float32,shape=[None,resolution,resolution,resolution,1])
    output_layer = lfmd.inference2(x)
    
    
    saver = tf.train.Saver()
    
    predictions = np.zeros(24)
    truelabels = np.zeros(24)
    truepositives = np.zeros(24)
    
    
    with tf.Session() as sess:
        
        
        saver.restore(sess, "models/featurenet.ckpt")
        print("Model restored.")
        
        
        with open(os.devnull, 'w') as devnull:
            for filename in Path('data/set' + str(sidx)+ '/').glob('*.STL'):
                
                filename = os.path.basename(filename)
                idx = int(os.path.splitext(filename)[0])
                
                sample = load_3dmodel(idx,sidx)
                segs = featurenet_segmentation(sample)
                    
                inputs = segs
                temp = output_layer.eval({x: inputs.reshape(-1, resolution,resolution,resolution,1)})
                       
                outputs = softmax(temp,axis=1)
                    
                    
                pred = get_lvec(outputs.argmax(1)).astype(int)
                    
                trul = get_lvec(vis.get_label('data/set'+str(sidx)+'/'+str(idx)+'.csv',0)[:,6]).astype(int)
                    
    
                    
                print(idx)
                print('Predicted labels:\t',pred)
                print('True labels:\t\t',trul)
                    
                tp = np.minimum(pred,trul)
                    
                predictions += pred
                truelabels += trul
                truepositives += tp
    


    precision, recall = eval_metric(predictions,truelabels,truepositives)

    
    return precision.mean(), recall.mean()

def create_sectional_view(obj3d):
    minres = min(obj3d.shape[0],obj3d.shape[1],obj3d.shape[2]) 
    
    
    proj_dir = random.randint(0,1)
    sel_axis = random.randint(0,2)
    sel_idx = random.randint(1, minres - 2)
    
    if sel_axis == 0:
        if proj_dir == 0:
            img = cp.mean(obj3d[sel_idx:,:,:], sel_axis)
        else:
            img = cp.mean(obj3d[:sel_idx,:,:], sel_axis)
    elif sel_axis == 1:
        if proj_dir == 0:
            img = cp.mean(obj3d[:,sel_idx:,:], sel_axis)
        else:
            img = cp.mean(obj3d[:,:sel_idx,:], sel_axis)
    elif sel_axis == 2:
        if proj_dir == 0:
            img = cp.mean(obj3d[:,:,sel_idx:], sel_axis)
        else:
            img = cp.mean(obj3d[:,:,:sel_idx], sel_axis)


    img = torch.from_numpy(img).float()
    img = img.expand(3,img.shape[0],img.shape[1]) #convert to rgb chanels
    
    
    trans = transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize((64,64), interpolation = Image.NEAREST),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])
    
    img = trans(img)
    
    
    return img

def create_sectional_views(obj3d, img_num):
    imgs = []
    for i in range(img_num):
        imgs.append(create_sectional_view(obj3d))
    
    return torch.stack(imgs)


#input: n*64*64*64
#output: n*12*3*64*64
def get_msv_samples(samples, num_cuts):
    batch_size = samples.shape[0]
    results = torch.zeros((batch_size,num_cuts,3,64,64))
    
    for i in range(batch_size):
        results[i] = create_sectional_views(samples[i],num_cuts)
    
    return results
        
        
def dense_to_one_hot(labels_dense, num_classes=24):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    
    return labels_one_hot
      

def get_lvec(labels):
    results = np.zeros(24)
    
    for i in labels:
        results[int(i)] += 1
    
    return results
        
def eval_metric(pre,trul,tp):
    precision = tp/pre
    
    recall = tp/trul
    
    return precision, recall
