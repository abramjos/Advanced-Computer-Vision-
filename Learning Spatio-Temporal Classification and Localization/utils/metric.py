import numpy as np

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1/100.0) * max(0, yB - yA + 1/100.0)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1/100.0) * (boxA[3] - boxA[1] + 1/100.0)
	boxBArea = (boxB[2] - boxB[0] + 1/100.0) * (boxB[3] - boxB[1] + 1/100.0)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def IoU(boxes1,boxes2,format_w_h=True):
	iou = 0.0 
	c = 0 
	for b1,b2 in zip(boxes1.cpu().detach().numpy(),boxes2.cpu().detach().numpy()):
		if b1.sum() == 0: continue
		boxA = np.array(b1)+np.array([0,0,float(b1[2]),float(b1[3])])
		boxB = np.array(b2)+np.array([0,0,float(b2[2]),float(b2[3])])
		if format_w_h == True:
			iou += bb_intersection_over_union(boxA,boxB)
		else:
			iou += bb_intersection_over_union(b1,b2)
		c+=1
	if c == 0: 
		return(iou)
	else:
		return (iou/float(c))
