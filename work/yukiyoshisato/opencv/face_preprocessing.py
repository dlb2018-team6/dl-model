import sys
import cv2
from PIL import Image


def preprocessing(source, destination):

    img = cv2.imread(source)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        w_adj = int(w / 2)
        # cv2.rectangle(img, (x - w_adj, y), (x + w + w_adj, y + w_adj * 4), (255, 0, 0), 2)
        cv2.imwrite(destination, img[y - w_adj:y + w_adj * 3, x - w_adj:x + w_adj * 3])
        break

    print("Image is processed.")


def resize(source, resized, size):

    img = Image.open(source)

    img_resize_lanczos = img.resize((size, size), Image.LANCZOS)
    img_resize_lanczos.save(resized)

    print("image is resized.")


if __name__ == '__main__':

    def main(args):

        source = "input/" + args[1]
        destination = "output/" + args[2]
        resized = "resized/" + args[2]
        size = int(args[3])

        print(source, destination, size)

        preprocessing(source=source, destination=destination)
        resize(source=destination, resized=resized, size=size)

    main(sys.argv)
