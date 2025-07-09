import cv2 as cv

# 1. โหลดรูปภาพ
img = cv.imread('1.jpg')

# 2. โหลดโมเดล Haar Cascade สำหรับการตรวจจับใบหน้า
face_model = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# 3. แปลงรูปภาพเป็นโทนสีเทา
gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 4. ตรวจจับใบหน้า
faces = face_model.detectMultiScale(gray_scale)

# 5. วาดสี่เหลี่ยมรอบใบหน้าที่ตรวจพบ
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

# 6. ปรับขนาดรูปภาพที่แสดงผล
scale_percent = 30  # เปอร์เซ็นต์การปรับขนาด
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
resized_img = cv.resize(img, (width, height))

# 7. แสดงรูปภาพและรอการกดปุ่ม
cv.imshow('image', resized_img)
cv.waitKey(0)
cv.destroyAllWindows()