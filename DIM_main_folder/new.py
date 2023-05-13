import cv2
import matplotlib as plt


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(1)
    while True:
        print(1)
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    plt.imshow()
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


main()
