import cv2
#是一个用于构建媒体处理流水线的框架，这里用于手部检测。
import mediapipe as mp
#库用于时间相关的操作，如计算帧率。
import time
import math
import numpy as np

class handDetector():
    #用于初始化类的实例(自动调用):初始化了手部检测的相关参数和对象
    #静态模式(表示视频模式)，最多检测的手部数量2，检测手部的置信度阈值0.5，跟踪手部的置信度阈值0.5
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        #将传入的参数赋值给类的属性
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        #初始化 MediaPipe Hands 模块   并创建手部检测对象，传入初始化参数
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        #初始化用于绘制检测结果的工具
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    #方法用于检测手部并绘制关键点
    def findHands(self, img, draw=True):
        #将输入图像从 BGR 格式转换为 RGB 格式
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #使用手部检测模型处理图像，获取检测结果
        self.results = self.hands.process(imgRGB)
        #如果检测到手部关键点，遍历每只手的关键点，并在图像上绘制关键点和连接线
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        #返回绘制后的图像
        return img

    #找到手部关键点位置  handNo表示手部编号默认为0  draw是否在图像上绘制关键点和边界框（默认True）
    def findPosition(self, img, handNo=0, draw=True):
        #临时存储所有关键点的x/y坐标
        xList = []
        yList = []
        #存储手部边界框的坐标
        bbox = []
        #类属性，存储关键点信息
        self.lmList = []
        if self.results.multi_hand_landmarks:
            #获取指定手部的关键点
            myHand = self.results.multi_hand_landmarks[handNo]
            #遍历手部的关键点，id 是关键点的编号，lm 是关键点的坐标信息
            for id, lm in enumerate(myHand.landmark):
                #图像的高度和宽度
                h, w, c = img.shape
                #将归一化坐标转换为实际像素坐标
                cx, cy = int(lm.x * w), int(lm.y * h)
                #存入像素坐标
                xList.append(cx)
                yList.append(cy)
                #存入关键点ID与坐标
                self.lmList.append([id, cx, cy])
                #如果需要绘制，在图像上绘制圆圈标记关键点
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            #通过最小/最大x/y值确定手部区域边界框
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = [xmin, ymin, xmax, ymax]
            #判断是否绘制绿色矩形框，边界扩展20像素以留出视觉边距
            if draw:
                cv2.rectangle(img, (xmin-20,ymin-20), (xmax+20,ymax+20), (0,255,0), 2)
        #返回包含关键点坐标列表和边界框坐标
        return self.lmList, bbox

    #检测每根手指的状态，1表示伸直，0表示弯曲
    def fingersUp(self):
        fingers = []
        # 通过x坐标检测拇指是否伸展(图像要反转，所以这里是小于号)
        if self.lmList[self.tipIds[0]][1]<self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 通过y坐标检测其他四根手指（食指、中指、无名指、小指）是否伸展
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2]<self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #返回五根手指的状态数组
        return fingers

    #计算两个手部关键点之间的距离 draw是否在图像上绘制图形 r绘制圆形的半径 t连接线的粗细
    def findDistance(self,p1,p2,img,draw=True,r=15,t=3):
        #从类属性 lmList 中提取两个关键点的像素坐标  并计算两个关键点连线的中点坐标
        x1,y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        cx,cy = (x1+x2)//2,(y1+y2)//2

        if draw:
            # 绘制连接线
            cv2.line(img,(x1,y1),(x2,y2),(255,0,255),t)
            # 绘制端点圆
            cv2.circle(img,(x1,y1),r,(255,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),r,(255,0,255),cv2.FILLED)
            # 绘制中点圆
            cv2.circle(img,(cx,cy),r,(0,0,255),cv2.FILLED)
        #计算欧氏距离
        length = math.hypot(x2-x1, y2-y1)
        #返回两点之间的实际像素距离，绘制后的图像，两个端点及中点的坐标列表
        return length, img, [x1,y1,x2,y2,cx,cy]

def main():
    pTime = 0
    cTime = 0
    #打开默认的摄像头
    cap = cv2.VideoCapture(0)
    #创建 handDetector 对象
    detector = handDetector()
    #进入无限循环，实时处理视频流
    while True:
        #读取摄像头的一帧图像
        success, img = cap.read()
        # 调用 findHands 方法检测手部并绘制关键点
        img = detector.findHands(img)
        #调用 findPosition 方法获取手部关键点坐标
        lmList, bbox= detector.findPosition(img)
        #如果检测到关键点，打印第 4 个关键点的坐标（食指指尖）
        if len(lmList) != 0:
            print(lmList[4])
        # fps 是帧率，计算公式为 1 / (当前时间 - 上一帧时间)，表示每秒可以处理多少帧图像。
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # 在图像上绘制文本
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

#这段代码的作用是确保 main() 函数只有在脚本被直接运行时才会被调用，而不是在脚本被导入时运行
if __name__ == "__main__":
    main()
