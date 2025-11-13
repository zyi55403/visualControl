import cv2
import numpy as np
import time
import os
import handtrackingmodule as htm
from blinkdetectionmodule import BlinkDetector
import tkinter as tk
from PIL import Image, ImageTk


class AIPaintingGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("AI绘画模块")
        self.master.geometry("1280x720")

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        # 初始化画布
        self.folderPath = "penimg"
        self.overlayList = self.load_pen_images()
        self.header = cv2.resize(self.overlayList[0], (1280, 125))
        self.drawColor = (255, 0, 255)
        self.brushThickness = 15
        self.eraserThickness = 80
        self.xp, self.yp = 0, 0
        self.imgCanvas = np.zeros((720, 1280, 3), np.uint8)
        self.lastCanvas = np.zeros((720, 1280, 3), np.uint8)

        # 初始化检测模块
        self.detector = htm.handDetector(detectionCon=0.80)
        self.blink_detector = BlinkDetector(reset_frames=5)
        self.blink_count = 0
        self.prev_blink_time = 0
        self.smoothening = 5
        self.prev_draw_x = 0
        self.prev_draw_y = 0

        # 定义工具区域 (这些是基于原始1280px宽度的逻辑区域)
        self.UNDO_AREA = (0, 400)
        self.COLOR1_AREA = (400, 600)
        self.COLOR2_AREA = (600, 840)
        self.COLOR3_AREA = (840, 1040)
        self.ERASER_AREA = (1040, 1280)

        # 定义界面布局常量
        self.TABLE_WIDTH = 480
        self.HEADER_HEIGHT = 125
        self.SCREEN_WIDTH = 1280

        # 创建GUI组件
        self.create_widgets()
        self.running = True
        self.update_frame()

        # 窗口关闭处理
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    # 加载画笔图像资源
    def load_pen_images(self):
        overlayList = []
        myList = os.listdir(self.folderPath)
        for imPath in myList:
            image = cv2.imread(f'{self.folderPath}/{imPath}')
            overlayList.append(image)
        return overlayList

    # 创建GUI界面组件
    def create_widgets(self):
        self.video_label = tk.Label(self.master)
        self.video_label.pack()

    def process_frame(self, img):
        img = cv2.flip(img, 1)
        # 眨眼检测
        img_blink = cv2.resize(img.copy(), (640, 360))
        processed_img, is_blinking = self.blink_detector.process_frame(img_blink, draw=True)
        current_blinks = self.blink_detector.get_blink_count()

        # 双眨眼检测
        if current_blinks > self.blink_count:
            current_time = time.time()
            if self.prev_blink_time != 0 and (current_time - self.prev_blink_time) <= 1:
                self.imgCanvas = self.lastCanvas.copy()
                cv2.putText(img, "Undo by Double Blink", (400, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.prev_blink_time = 0
            else:
                self.prev_blink_time = current_time
            self.blink_count = current_blinks

        # 手部检测
        img = self.detector.findHands(img)
        lmList, bbox = self.detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            fingers = self.detector.fingersUp()

            # 选择模式
            if fingers[1] and fingers[2]:
                self.xp, self.yp = 0, 0
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), self.drawColor, cv2.FILLED)

                if y1 < self.HEADER_HEIGHT:
                    # 因为工具栏被挤压显示在 [480, 1280] 的区域，
                    # 我们需要将手在该区域的物理x坐标(x1)映射回原始的 [0, 1280] 逻辑坐标(logical_x1)
                    logical_x1 = -1  # 默认值，表示不在有效区域
                    if x1 > self.TABLE_WIDTH:
                        # 计算手在被挤压的工具栏中的相对位置 (0 to 800)
                        pos_in_squeezed_header = x1 - self.TABLE_WIDTH
                        # 计算被挤压后的工具栏宽度
                        squeezed_width = self.SCREEN_WIDTH - self.TABLE_WIDTH
                        # 按比例换算回原始1280宽度下的坐标
                        logical_x1 = (pos_in_squeezed_header / squeezed_width) * self.SCREEN_WIDTH

                    # 使用转换后的 logical_x1 进行判断
                    if self.UNDO_AREA[0] < logical_x1 < self.UNDO_AREA[1]:
                        self.imgCanvas = self.lastCanvas.copy()
                        cv2.putText(img, "Undo Last Stroke", (50, 360),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    elif self.COLOR1_AREA[0] < logical_x1 < self.COLOR1_AREA[1]:
                        self.header = cv2.resize(self.overlayList[0], (1280, 125))
                        self.drawColor = (255, 0, 255)
                    elif self.COLOR2_AREA[0] < logical_x1 < self.COLOR2_AREA[1]:
                        self.header = cv2.resize(self.overlayList[1], (1280, 125))
                        self.drawColor = (255, 50, 50)
                    elif self.COLOR3_AREA[0] < logical_x1 < self.COLOR3_AREA[1]:
                        self.header = cv2.resize(self.overlayList[2], (1280, 125))
                        self.drawColor = (0, 255, 0)
                    elif self.ERASER_AREA[0] < logical_x1 < self.ERASER_AREA[1]:
                        self.header = cv2.resize(self.overlayList[3], (1280, 125))
                        self.drawColor = (0, 0, 0)

            # 绘画模式
            if fingers[1] and not fingers[2]:
                # ... (绘画模式代码无需修改)
                if self.xp == 0 and self.yp == 0:
                    self.lastCanvas = self.imgCanvas.copy()
                    self.prev_draw_x = x1
                    self.prev_draw_y = y1

                clocX = self.prev_draw_x + (x1 - self.prev_draw_x) / self.smoothening
                clocY = self.prev_draw_y + (y1 - self.prev_draw_y) / self.smoothening
                self.prev_draw_x, self.prev_draw_y = clocX, clocY
                cv2.circle(img, (int(clocX), int(clocY)), 15, self.drawColor, cv2.FILLED)

                if self.xp == 0 and self.yp == 0:
                    self.xp, self.yp = int(clocX), int(clocY)

                if self.drawColor == (0, 0, 0):
                    cv2.line(img, (self.xp, self.yp), (int(clocX), int(clocY)),
                             self.drawColor, self.eraserThickness)
                    cv2.line(self.imgCanvas, (self.xp, self.yp), (int(clocX), int(clocY)),
                             self.drawColor, self.eraserThickness)
                else:
                    cv2.line(img, (self.xp, self.yp), (int(clocX), int(clocY)),
                             self.drawColor, self.brushThickness)
                    cv2.line(self.imgCanvas, (self.xp, self.yp), (int(clocX), int(clocY)),
                             self.drawColor, self.brushThickness)

                self.xp, self.yp = int(clocX), int(clocY)
            else:
                self.xp, self.yp = 0, 0
                self.prev_draw_x = 0
                self.prev_draw_y = 0

        # 合成图像
        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, self.imgCanvas)

        # --- 界面元素绘制 ---
        # 1. 优先绘制左上角的波形图（表格）
        table_height = 360
        if processed_img is not None:
            plot_img = processed_img[:, 640:1280, :]
            plot_img = cv2.resize(plot_img, (self.TABLE_WIDTH, table_height))
            img[0:table_height, 0:self.TABLE_WIDTH] = plot_img

        # 2. 将原始 header 图像“挤压”到新的宽度
        new_header_width = self.SCREEN_WIDTH - self.TABLE_WIDTH  # 1280 - 480 = 800
        squeezed_header = cv2.resize(self.header, (new_header_width, self.HEADER_HEIGHT))

        # 3. 将“挤压”后的工具栏图像放置在表格右侧
        img[0:self.HEADER_HEIGHT, self.TABLE_WIDTH:self.SCREEN_WIDTH] = squeezed_header

        # 在其他位置添加文字信息
        cv2.putText(img, f'Blinks: {current_blinks}', (500, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        current_threshold = self.blink_detector.blink_threshold
        cv2.putText(img, f'Threshold: {current_threshold:.1f}', (500, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return img

    def update_frame(self):
        if not self.running:
            return

        success, frame = self.cap.read()
        if success:
            processed_frame = self.process_frame(frame)
            img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.master.after(15, self.update_frame)

    def on_close(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.master.destroy()


# 如果你想直接运行这个文件进行测试，可以添加以下代码
if __name__ == "__main__":
    root = tk.Tk()
    app = AIPaintingGUI(root)
    root.mainloop()