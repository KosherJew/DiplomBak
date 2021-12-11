import os, io
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, BatchNormalization
from tensorflow.keras.layers import Activation
import cv2
from sklearn.model_selection import train_test_split
import time
from PIL import Image


def load_train_data(way_x, way_y):
    train_x, train_y = [], []
    for j in sorted(os.listdir(way_x)):
        flore_x = os.path.join(way_x, j)
        flore_y = os.path.join(way_y, j)
        for i in sorted(os.listdir(flore_x)):
            image = cv2.imread(os.path.join(flore_x, i))
            image = cv2.resize(image, (455, 256))
            image = np.asarray(image)
            if 'lost_image' in locals():
                frame_to_frame = np.concatenate([lost_image, image], axis=2)
                lost_image = image
                train_x.append(frame_to_frame)
            else:
                lost_image = image
            if os.path.isfile(os.path.join(flore_y, i)):
                print(i)
                image = cv2.imread(os.path.join(flore_y, i))
                image = cv2.resize(image, (455, 256))
                image = np.asarray(image)
                train_y.append(image)
        del lost_image
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    train_x = train_x / 255
    train_y = train_y / 255
    return train_x, train_y


def model_init(input_shape):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization(batch_size=16))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization(batch_size=16))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(UpSampling2D(size=(2, 2)))

    model.add(BatchNormalization(batch_size=16))
    model.add(Conv2D(32, (2, 2), padding='same'))
    model.add(Activation('relu'))

    model.add(BatchNormalization(batch_size=16))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(ZeroPadding2D(padding=((0, 0), (1, 2))))  # Zero padding for fitting output layers shape
    model.add(BatchNormalization(batch_size=16))
    model.add(Conv2D(16, (2, 2), padding='same'))
    model.add(Activation('relu'))

    model.add(BatchNormalization(batch_size=16))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(BatchNormalization(batch_size=16))
    model.add(Conv2D(3, (1, 1)))
    model.add(Activation('sigmoid'))

    model.compile(
        optimizer="adam",
        loss=tensorflow.losses.binary_crossentropy,
        metrics=["accuracy"])
    return model


def train(model, train_x, train_y, test_x, test_y, epochs=20, batch_size=16):
    model.fit(
        train_x, train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_x, test_y)
    )
    model.save_weights('1.h5')
    return model


def write_video(file_path):
    model.load_weights(input("Путь к файлу весов: "))
    cap = cv2.VideoCapture(file_path)
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    buf = []
    images = []
    lost_time = time.time()
    all_time = time.time()
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    count = 0
    agr_time = 0
    avr_img = 0
    step_x = 10
    step_y = 10
    ret, control_image = cap.read()
    if ret:
        control_image = cv2.resize(control_image, (455, 256))
        control_image = np.asarray(control_image)
        control_image = control_image / 255
        file = open(input("Путь к запакованному видеофайлу: "), "wb")
        file.write(w.to_bytes(2, "little"))
        file.write(h.to_bytes(2, "little"))
        file.write(fps.to_bytes(2, "little"))
        file.write(step_x.to_bytes(2, "little"))
        file.write(step_y.to_bytes(2, "little"))
    while 1:
        ret, image = cap.read()
        if ret:
            images.append(image)
            image = cv2.resize(image, (455, 256))
            image = np.asarray(image)
            image = image / 255
            frame_to_frame = np.concatenate([image, control_image], axis=2)
            buf.append(frame_to_frame)
            control_image = image
            if len(buf) == 12:
                buf = np.asarray(buf)
                frame = detect(model, buf)
                count += len(buf)
                buf = []
                for iter in range(len(frame)):
                    buf.append(1 - cv2.resize(np.array(frame[iter]), (w, h)))
                    buf[iter] = np.max(buf[iter], axis=2)
                    buf[iter] = buf[iter] / np.max(buf[iter])
                frame = buf
                buf = []
                c = 0
                images = np.array(images)
                for i in range(len(frame)):
                    frame[i] = frame[i] - (np.sum(frame, axis=0) / np.max(np.sum(frame, axis=0)))
                    frame[i] = np.where(frame[i] <= 0.1, 0, 1)
                    image = frame[i]
                    count_part = 0
                    for x in range(step_x):
                        for y in range(step_y):
                            xratio1 = x / step_x
                            xratio2 = (x + 1) / step_x
                            yratio1 = y / step_y
                            yratio2 = (y + 1) / step_y
                            count_part += 1
                            if np.sum(image[int(yratio1 * h):int(yratio2 * h), int(xratio1 * w):int(xratio2 * w)]) > 0:
                                part = images[i, int(yratio1 * h):int(yratio2 * h), int(xratio1 * w):int(xratio2 * w),
                                       ::]
                                with io.BytesIO() as buffer:
                                    part = Image.fromarray(part)
                                    part.save(buffer, "JPEG")
                                    contents = buffer.getvalue()
                                    file.write(len(contents).to_bytes(2, "little"))
                                    file.write((count + i).to_bytes(2, "little"))
                                    file.write(count_part.to_bytes(1, "little"))
                                    file.write(contents)
                images = []
                agr_time = (time.time() - all_time) / count
                print('обработано: ', count, ' из: ', length, ' кадров, осталось:', round(agr_time * (length - count)),
                      'с')
                ost_time = time.time()
        else:
            if len(buf) > 0:
                buf = np.asarray(buf)
                count += len(buf)
                frame = detect(model, buf)
                buf = []
                for iter in range(len(frame)):
                    buf.append(1 - cv2.resize(np.array(frame[iter]), (w, h)))
                    buf[iter] = np.max(buf[iter], axis=2)
                    buf[iter] = buf[iter] / np.max(buf[iter])
                frame = buf
                buf = []
                c = 0
                images = np.array(images)
                for i in range(len(frame)):
                    frame[i] = frame[i] - (np.sum(frame, axis=0) / np.max(np.sum(frame, axis=0)))
                    frame[i] = np.where(frame[i] <= 0.1, 0, 1)
                    image = frame[i]
                    count_part = 0
                    for x in range(step_x):
                        for y in range(step_y):
                            xratio1 = x / step_x
                            xratio2 = (x + 1) / step_x
                            yratio1 = y / step_y
                            yratio2 = (y + 1) / step_y
                            count_part += 1
                            if np.sum(image[int(yratio1 * h):int(yratio2 * h), int(xratio1 * w):int(xratio2 * w)]) > 0:
                                part = images[i, int(yratio1 * h):int(yratio2 * h), int(xratio1 * w):int(xratio2 * w),
                                       ::]
                                with io.BytesIO() as buffer:
                                    part = Image.fromarray(part)
                                    part.save(buffer, "JPEG")
                                    contents = buffer.getvalue()
                                    file.write(len(contents).to_bytes(2, "little"))
                                    file.write((count + i).to_bytes(2, "little"))
                                    file.write(count_part.to_bytes(1, "little"))
                                    file.write(contents)
                images = []
            print('обработано: ', count, ' из: ', length, ' кадров, прошло:', time.time() - all_time, 'с')
            cap.release()
            cv2.destroyAllWindows()
            break


def detect(model, frame):
    y = model.predict(frame, batch_size=4)

    return y


while 1:
    model = model_init((256, 455, 6))
    chech = int(input('Тренировка = 1\nОптический поток пары изображений = 2\nЗапаковать видео файл = 3\nРаспаковать видео файл = 2: '))
    if chech == 1:
        x, y = load_train_data(input("Путь к входным данным обучения: "), input("Путь к выходным данным обучения: "))
        print(len(x), len(y))
        (trainX, testX, trainY, testY) = train_test_split(x, y, test_size=0.20, random_state=42)
        del x, y
        model = train(model, trainX, trainY, testX, testY, 50)

    elif chech == 2:
        model.load_weights(input("Путь к файлу весов: "))
        image_1 = cv2.imread(input("Кадр 1: "))
        image_2 = cv2.imread(input("Кадр 1: "))
        image_1 = cv2.resize(image_1, (455, 256))
        image_1 = np.asarray(image_1)
        image_2 = cv2.resize(image_2, (455, 256))
        image_2 = np.asarray(image_2)
        frame_to_frame = np.concatenate([image_1, image_2], axis=2)
        frame_to_frame = np.expand_dims(frame_to_frame, axis=0)
        frame_to_frame = frame_to_frame / 255
        image = detect(model, frame_to_frame)
        print(type(image[0]))
        image = np.array(image[0], dtype=float)

        cv2.imshow('or', image)
        cv2.waitKey(0)
    elif chech == 3:
        write_video(input("Путь к видео файлу: "))
    elif chech == 4:
        losttime = time.time()
        f = open(input("Путь к запакованному файлу: "), 'rb')
        a = []
        w = int.from_bytes(f.read(2), "little")
        h = int.from_bytes(f.read(2), "little")
        fps = int.from_bytes(f.read(2), "little")
        step_x = int.from_bytes(f.read(2), "little")
        step_y = int.from_bytes(f.read(2), "little")
        count_part = 0
        parts = {}
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        writer = cv2.VideoWriter(input("Путь к распакованному файлу: "), fourcc, fps, (w, h))
        print(w, h, fps, step_x, step_y)
        for y in range(step_y):
            for x in range(step_x):
                xratio1 = x / step_x
                xratio2 = (x + 1) / step_x
                yratio1 = y / step_y
                yratio2 = (y + 1) / step_y
                count_part += 1
                parts[count_part] = (int(xratio1 * h), int(xratio2 * h), int(yratio1 * w), int(yratio2 * w))
        corrent_frame = np.zeros((h, w, 3))
        number_frame = 0
        while 1:
            len_c = int.from_bytes(f.read(2), "little")
            frame = int.from_bytes(f.read(2), "little")
            part = int.from_bytes(f.read(1), "little")
            with io.BytesIO() as buffer:
                for i in range(len_c):
                    byte = f.read(1)
                    buffer.write(byte)
                    a.append(byte)
                if a == []:
                    print("конец файла")
                    break
                stream = io.BytesIO(buffer.getvalue())
                image = Image.open(stream).convert("RGB")
                stream.close()
                if frame == number_frame:
                    corrent_frame[parts[part][0]:parts[part][1], parts[part][2]:parts[part][3], ::] = image
                else:
                    corrent_frame[parts[part][0]:parts[part][1], parts[part][2]:parts[part][3], ::] = image
                    corrent_frame = cv2.cvtColor(corrent_frame.astype("uint8"), cv2.COLOR_RGB2BGR)
                    for i in range(number_frame, frame):
                        writer.write(corrent_frame)
                        print('кадр ', i)
                    number_frame = frame

            if a != []:
                a = []
            else:
                writer.release()
                cv2.destroyAllWindows()
                print("конец файла")
                break
        print(time.time() - losttime)