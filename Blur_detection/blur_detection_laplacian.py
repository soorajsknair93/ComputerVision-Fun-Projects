import cv2


def calculate_focal_measure(img):
    # convert RGB image to Gray scale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Measure focal measure score (laplacian approach)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


imagePath = "images/blurry.jpeg"

image = cv2.imread(imagePath)
focal_measure = calculate_focal_measure(image)
print("focal-measure-score",focal_measure)

if focal_measure > 150:
    status = "Non Blurry"
    color = (255, 0, 0)
else:
    status = "Blurry"
    color = (0, 0, 255)


image = cv2.putText(image, "{}-{}".format(status,int(focal_measure)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale= 1, color=color, thickness = 2, lineType=cv2.LINE_AA)
cv2.imshow("output",image)
cv2.waitKey(0)
