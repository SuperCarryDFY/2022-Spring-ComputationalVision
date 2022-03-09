import cv2 as cv
cap = cv.VideoCapture('Waymo.mp4')
fps = cap.get(cv.CAP_PROP_FPS)
while cap.isOpened():
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv.imshow('frame',frame)
    if cv.waitKey(int(fps)) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()