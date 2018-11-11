# -*- coding: UTF-8 -*-
import cv2
import selectivesearch
 
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)
height = 180
#height = 270
width  = 320
#width  = 480
 
def main():
    while(True):
 
        # 動画ストリームからフレームを取得
        ret, frame = cap.read()
        
        # カメラ画像をリサイズ
        img = cv2.resize(frame,(width,height))
 
        # perform selective search
        img_lbl, regions = selectivesearch.selective_search(
            img, scale=500, sigma=0.9, min_size=10)
 
        candidates = set()
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 2000 pixels
            if r['size'] < 2000:
                continue
            # distorted rects
            x, y, w, h = r['rect']
            if w / h > 1.2 or h / w > 1.2:
                continue
            candidates.add(r['rect'])
        
        #画像への枠作成
        for region in candidates:
            x,y,w,h = region
            color = (100, 200, 100)
            cv2.rectangle(img, (x,y), (x+w, y+h), color, thickness=2)
            
        cv2.imshow("camera window", img) 
 
        # escを押したら終了。
        if cv2.waitKey(1) == 27:
            break
    
    #終了
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    main()
