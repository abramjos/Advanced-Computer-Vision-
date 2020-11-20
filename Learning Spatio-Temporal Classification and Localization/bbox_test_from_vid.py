import pickle
import cv2
import os 

#os.mkdir('tests')

test = 'Biking/v_Biking_g08_c06'
#video input 
cap = cv2.VideoCapture('train/'+test+'.avi')

with open('pyannot.pkl','rb') as f:
    x = pickle.load(f)

anno = x[test]['annotations'][0]
boxes = anno['boxes']

sf = anno['sf']
ef = anno['ef']

print('starting frame %d, ending frame %d'%(sf,ef))

c = 0 
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if c>=ef:
       break
    elif c<sf:
       c+=1
       continue
    else:
       box = boxes[c]
       im_box = cv2.rectangle(frame, (box[0],box[1]),(box[0]+box[2],box[1]+box[3]), (255,0,0), 2)
       cv2.imwrite('tests/im_{}_f-{}.jpg'.format(c,sf+c),im_box)
       c+=1

