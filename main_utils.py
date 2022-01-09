import cv2
import numpy as np
import pickle
import json


def draw_all_dist(imgs, bboxes, paired_list, measurements, colors):
    for i, pair in enumerate(paired_list):
        point_3D_1 = measurements[i][0]
        for j, mes in enumerate(measurements):
            point_3D_2 = mes[0]
            dist = np.linalg.norm(point_3D_1-point_3D_2)
            if(np.linalg.norm(point_3D_1) != np.linalg.norm(point_3D_2)):
                draw_line(imgs[0], bboxes[0][pair[0]], bboxes[0][paired_list[j][0]], colors[i], colors[j], label=f'{round(dist,1)}cm', res=100)
                draw_line(imgs[1], bboxes[1][pair[1]], bboxes[1][paired_list[j][1]], colors[i], colors[j], label=f'{round(dist,1)}cm', res=100)


def draw_least_dist(imgs, bboxes, paired_list, measurements, colors):
    for i, pair in enumerate(paired_list):
        distances = np.ones((len(paired_list)))*1000
        point_3D_1 = measurements[i][0]
        for j, mes in enumerate(measurements):
            point_3D_2 = mes[0]
            dist = np.linalg.norm(point_3D_1-point_3D_2)
            if(np.linalg.norm(point_3D_1) != np.linalg.norm(point_3D_2)):
                distances[j] = dist
        arg = np.argmin(distances)
        if(distances[arg] < 1000 and distances[arg] > 3):
            draw_line(imgs[0], bboxes[0][pair[0]], bboxes[0][paired_list[arg][0]], colors[i], colors[arg], label=f'{round(distances[arg],1)}cm', res=100)
            draw_line(imgs[1], bboxes[1][pair[1]], bboxes[1][paired_list[arg][1]], colors[i], colors[arg], label=f'{round(distances[arg],1)}cm', res=100)


def draw_line(img, bbx1, bbx2, color1, color2, label='', res=100):
    center1 = (int((bbx1[0] + bbx1[2])/2), int((bbx1[1] + bbx1[3])/2))
    center2 = (int((bbx2[0] + bbx2[2])/2), int((bbx2[1] + bbx2[3])/2))
    cv2.circle(img, center1, 5, color1, -1)
    cv2.circle(img, center2, 5, color2, -1)
    step = abs(center1[0]-center2[0])/(res+1)
    if(center2[0]>center1[0]):
        ydelta = center2[1]-center1[1]
        xdelta = center2[0]-center1[0]
    else:
        ydelta = center1[1]-center2[1]
        xdelta = center1[0]-center2[0]
        temp = center1
        center1 = center2
        center2 = temp
        temp = color1
        color1 = color2
        color2 = temp
    m = ydelta/(xdelta + 1e-6)
    b = center1[1] - (center1[0]*m)
    for i in range(res+1):
        alpha = i/res
        seg_color = (int(((1-alpha)*color1[0]+(alpha)*color2[0])), int(((1-alpha)*color1[1]+(alpha)*color2[1])), int(((1-alpha)*color1[2]+(alpha)*color2[2])))
        x0 = center1[0] + (i)*step
        y0 = x0*m+b
        x1 = center1[0] + (i+1)*step
        y1 = x1*m+b
        seg_pt1 = (int(x0), int(y0))
        seg_pt2 = (int(x1), int(y1))
        cv2.line(img, seg_pt1, seg_pt2, seg_color, 2)
        if(i == int(res/2)):
            pos = seg_pt1
            color = seg_color
    tl = 2 or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  
    c1 = pos
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def get_flipped_txt(img, bbx, label, color, oppacity):
    tl = 2 or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    max_t = max(t_size[1], t_size[0])
    min_t = min(t_size[1], t_size[0])
    temp = img[    bbx[1]  :   bbx[1]+max_t,     bbx[0]-int(max_t/2)-int(min_t/2)-4   :   bbx[0]+int(max_t/2)-int(min_t/2)-4   ]
    text_height = t_size[1] + 3
    delta = (max_t - text_height) / 2
    c1 = (0, int(max_t-delta))
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    M = cv2.getRotationMatrix2D((int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2)), -90, 1)
    pt1 = np.zeros((2,1), dtype=np.uint8)
    pt1[0] = c1[0]
    pt1[1] = c1[1]
    pt2 = np.zeros((2,1), dtype=np.uint8)
    pt2[0] = c2[0]
    pt2[1] = c2[1]
    A = M[:,0:2]
    B = np.zeros((2,1), dtype=np.uint8)
    B[:,0] = M[0:2,2]
    new_c1 = (np.matmul(A, pt1) + B).astype(int)
    new_c2 = (np.matmul(A, pt2) + B).astype(int)
    temp = cv2.warpAffine(temp, M, (temp.shape[1], temp.shape[0]))
    temp_opa = temp.copy()
    cv2.rectangle(temp, c1, c2, color , -1)  # filled
    res = cv2.addWeighted(temp_opa, 1-oppacity, temp, oppacity, 1.0)
    cv2.putText(res, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2)), 90, 1)
    res = cv2.warpAffine(res, M, (temp.shape[1], temp.shape[0]))
    res = res[1:,new_c1[0][0]:new_c2[0][0]]
    c2 = (bbx[0]-2, bbx[1])
    c1 = (bbx[0]-res.shape[1]-2, bbx[1]+res.shape[0])
    return res, c1, c2


def plot_one_box(x, im, color=None, label=None, line_thickness=3, oppacity=1.):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    sub_img = im.copy()
    if(oppacity<1):
        cv2.rectangle(sub_img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    else:
        cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        if(oppacity<1):
            cv2.rectangle(sub_img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        else:
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
    if(oppacity<1):
        new_c1 = min(max(c1[0], 0), im.shape[1]), min(max(c1[1], 0), im.shape[0])
        new_c2 = min(max(c2[0], 0), im.shape[1]), min(max(c2[1], 0), im.shape[0])
        res = cv2.addWeighted(sub_img, oppacity, im, 1-oppacity, 1.0)
        im[new_c2[1]:new_c1[1], new_c1[0]:new_c2[0]] = res[new_c2[1]:new_c1[1], new_c1[0]:new_c2[0]]
        x0 = max(0, int(x[0]-tl/2))
        x1 = min(im.shape[1], int(x[2]+tl/2)+1)
        y0 = max(0, int(x[1]-tl/2))
        y1 = min(im.shape[0], int(x[3]+tl/2)+1)
        im[y0:y1:, x0:x1] = res[y0:y1:, x0:x1]
    if label:
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        

def draw_results(opt, img, detections):
    for i, detection in enumerate(detections):
        bbx = detection.xyxy
        label = f'id:{detection.id} {detection.classification.name}'
        label = detection.classification.name
        conf = detection.classification.conf
        color = detection.classification.color
        label = None if opt.hide_labels else (f'{label}' if opt.hide_conf else f'{label} {conf:.2f}')
        plot_one_box(bbx, img, label=label, color=color, line_thickness=opt.line_thickness, oppacity=opt.oppacity)


def draw_segmentation(opt, img, detections):
    if(opt.draw_segmentation):
        try:
            cop = img.copy()
            for i, detection in enumerate(detections):
                if(detection.is_stereo) or True:
                    xyxy = detection.xyxy
                    out = detection.segmentation
                    mask = np.where(out > 0)
                    new_mask_y = [x+xyxy[1] for x in mask[0]]
                    new_mask_x = [x+xyxy[0] for x in mask[1]]
                    new_mask = (new_mask_y, new_mask_x)
                    cop[new_mask] = [0, 120, 255]
            img = cv2.addWeighted(cop, 0.5, img, 0.5, 1.0)
        except:
            """
                cop[new_mask] = [0, 120, 255]
                IndexError: index 1080 is out of bounds for axis 0 with size 1080
            """
            pass
    return img


def draw_length(opt, img, detections):
    if(opt.draw_length):
        for detection in detections:
            if(detection.is_stereo):
                if(detection.p1_width_bary is not None) and (detection.p2_width_bary is not None):
                    p1 = detection.p1_width_bary.astype(int)
                    p2 = detection.p2_width_bary.astype(int)
                    p1 = (p1[0], p1[1])
                    p2 = (p2[0], p2[1])
                    img = cv2.line(img, p1, p2, (0, 0, 255), 2)
                    cv2.circle(img, p1, 2, (255, 0, 0), -1)
                    cv2.circle(img, p2, 2, (255, 0, 0), -1)
                '''
                if(detection.p1_height_bary is not None) and False:
                    p1 = detection.p1_height_bary.astype(int)
                    p2 = detection.p2_height_bary.astype(int)
                    p1 = (p1[0], p1[1])
                    p2 = (p2[0], p2[1])
                    cv2.circle(img0s[0], p1, 2, (255, 0, 0), -1)
                    cv2.circle(img0s[0], p2, 2, (255, 0, 0), -1)
                '''
    return img


def draw_trajectories(opt, img, tracker):
    if(opt.draw_trajectories):
        for track in tracker.get_active():
            centers = []
            for xyxy in track.tracked_2D_pos:
                center = (int((xyxy[0] + xyxy[2])/2), int((xyxy[1] + xyxy[3])/2))
                centers.append(center)
            for i in range(len(centers)-1):
                center_n0 = centers[i]
                center_n1 = centers[i+1]
                img = cv2.line(img, center_n0, center_n1, track.color, 3)
    return img


def draw_dimensions(opt, img, detection):
    oppacity = opt.oppacity
    if(detection.is_stereo):
        color = detection.stereo_color
        sub_img = img.copy()
        bbx = detection.xyxy
        # Height is not correct so it doesn't display
        """
        if(detection.measurement.height != -1):
            label = f'{round(detection.measurement.height,1)}cm'
        else: label = ''
        
        try:
            flipped_txt, c1_h, c2_h = get_flipped_txt(img, bbx=bbx, label=label, color=color, oppacity=oppacity)
        except:
            # TODO: Fix issue vertical text is out of the image
            pass
        """
        if(detection.measurement.width != -1):
            label = f'{round(detection.measurement.width,1)}cm'
        else: label = ''

        tl = 2 or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        max_t = max(t_size[1], t_size[0])
        temp = np.zeros((max_t,max_t,3), dtype=np.uint8)
        text_height = t_size[1] + 3
        c1 = (bbx[0]-tl, bbx[3]+text_height)
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(sub_img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        res = cv2.addWeighted(img, 1-oppacity, sub_img, oppacity, 1.0)
        h, w = res.shape[:2]
        try: 
            img[bbx[1]:bbx[1]+h, bbx[0]-w:bbx[0]] = res[bbx[1]:bbx[1]+h, bbx[0]-w:bbx[0]]
            img[c2[1]+2:c1[1]+2, c1[0]:c2[0]] = res[c2[1]+2:c1[1]+2, c1[0]:c2[0]]
        except:
            # TODO: Fix issue horizontal text is out of the image
            pass
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        try:
            img[c2_h[1]+1:c1_h[1]+1, c1_h[0]:c2_h[0]] = flipped_txt
        except:
            # TODO: Fix issue vertical text is out of the image
            pass


def draw_stereo_results(opt, imgs, detections):
    # Draw all boxes in red
    for i in [0,1]:
        for detection in detections[i]:
            bbx = detection.xyxy
            label = f'id:{detection.id} {detection.classification.name}'
            label = detection.classification.name
            conf = detection.classification.conf
            label = None if opt.hide_labels else (f'{label}' if opt.hide_conf else f'{label} {conf:.2f}')
            if(detection.is_stereo):
                color = detection.stereo_color
                plot_one_box(bbx, imgs[i], label=label, color=color, line_thickness=opt.line_thickness, oppacity=opt.oppacity)
                if(opt.draw_dim):
                    draw_dimensions(opt, imgs[i], detection)
            else:
                color = (0, 0, 255)
                #color = detection.stereo_color
                plot_one_box(bbx, imgs[i], label=label, color=color, line_thickness=opt.line_thickness, oppacity=1)


def check_json_files(dir_path):
    assert os.path.isdir(dir_path), f"\nERROR: '{os.path.abspath(opt.left_path)}' is not a directory."
    json_paths = glob.glob(dir_path+'/*.json')
    assert len(json_paths)>0, f"\nERROR: No json file found in '{os.path.abspath(dir_path)}'."
    data_list = []
    left_paths = []
    right_paths = []
    for path in json_paths:
        data = read_json(path)
        data_list.append((data, path))
        left_paths.append(data['left_path'])
        right_paths.append(data['right_path'])
    # Check occurence
    check_ok = True
    for i, (data, path) in enumerate(data_list):
        for data_, path_ in data_list[i+1:]:
            if(data['left_path'] == data_['left_path']) and (path != path_):
                check_ok = False
                print(f"WARNING: Conflict between '{path}' and '{path_}'. Same left path shared.")
        for data_, path_ in data_list[i+1:]:
            if(data['right_path'] == data_['right_path']) and (path != path_):
                check_ok = False
                print(f"WARNING: Conflict between '{path}' and '{path_}'. Same right path shared.")
    if(not(check_ok)): raise Exception('Conflicts detected in json files. Please check warnings.')


def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def read_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(filepath, data):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)