import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    cv2.imshow('Camera Test', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.getWindowProperty('Camera Test', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()