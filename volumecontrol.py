import cv2
import numpy as np
import time
import handtrackingmodule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class VolumeController:
    def __init__(self, cap=None):
        # 摄像头初始化
        self._init_camera(cap)
        # 手部检测器
        self.detector = htm.handDetector(detectionCon=0.7)
        # 音频设备
        self._init_audio()
        # 状态参数
        self.volBar = 400# 音量条初始位置（像素Y坐标）
        self.volPer = 0# 音量百分比初始值
        self.exit_delay = 1.0# 退出手势需要保持的时间（秒）
        self.exit_start_time = 0# 退出手势开始时间戳
    #摄像头初始化逻辑：如果未传入摄像头对象，创建新的摄像头实例
    def _init_camera(self, cap):
        if cap is None:  # 独立运行模式
            self.cap = cv2.VideoCapture(0)
            self._setup_camera()# 设置摄像头参数
            self.own_camera = True# 标记为自有摄像头
        else:  # 共享摄像头模式
            self.cap = cap
            self.own_camera = False
        self.wCam = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.hCam = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #摄像头参数设置
    def _setup_camera(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
    #音频设备初始化
    def _init_audio(self):
        devices = AudioUtilities.GetSpeakers()# 获取默认音频输出设备
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
    #退出手势检测
    def _check_exit_gesture(self, fingers):
        return fingers[1] == 1 and sum(fingers[0:1] + fingers[2:5]) == 0
    #统一的手势处理逻辑
    def _process_gestures(self, img, lmList):
        if not lmList:
            self.exit_start_time = 0
            return img, False
        fingers = self.detector.fingersUp()
        # 退出手势处理
        if self._check_exit_gesture(fingers):
            current_time = time.time()
            if self.exit_start_time == 0:
                self.exit_start_time = current_time
            elif current_time - self.exit_start_time >= self.exit_delay:
                return img, True
        else:
            self.exit_start_time = 0# 重置计时
            img = self._update_volume(img, lmList)# 更新音量
        return img, False
    #音量更新
    def _update_volume(self, img, lmList):
        if len(lmList) < 5:# 确保有足够的关节点
            return img
        bar_top = int(self.hCam * 0.25)
        bar_bottom = int(self.hCam * 0.75)
        thumb_y = lmList[4][2]
        volPer = np.interp(thumb_y, [bar_top, bar_bottom], [100, 0])
        volPer = np.clip(volPer, 0, 100)
        self.volume.SetMasterVolumeLevelScalar(volPer / 100, None)
        self.volBar = np.interp(volPer, [0, 100], [bar_bottom, bar_top])  # 音量条位置映射
        # 绘制指示线和指尖圆圈
        cv2.circle(img, (lmList[4][1], thumb_y), 15, (0, 255, 0), cv2.FILLED)
        cv2.line(img, (lmList[4][1], bar_top), (lmList[4][1], bar_bottom), (0, 255, 0), 2)
        return img
    #统一界面绘制
    def _draw_interface(self, img):
        # 动态计算音量条位置
        bar_top = int(self.hCam * 0.25)  # 顶部位于25%高度
        bar_bottom = int(self.hCam * 0.75)  # 底部位于75%高度
        # 音量条外框和填充部分
        cv2.rectangle(img, (50, bar_top), (85, bar_bottom), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(self.volBar)), (85, bar_bottom), (0, 255, 0), cv2.FILLED)
        # 音量百分比
        actual_vol = round(self.volume.GetMasterVolumeLevelScalar() * 100, 1)# 获取实际音量
        text_x = 100  # 移动到音量条右侧
        text_y = bar_top - 30  # 显示在音量条上方
        bg_color = img[text_y - 20:text_y + 20, text_x - 20:text_x + 20].mean()
        text_color = (0, 0, 0) if bg_color > 127 else (255, 255, 255)
        cv2.putText(img, f'{actual_vol:.0f}%',
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        # 添加音量刻度线
        for per in [0, 25, 50, 75, 100]:
            y_pos = int(np.interp(per, [0, 100], [bar_bottom, bar_top]))
            cv2.line(img, (35, y_pos), (50, y_pos), (0, 255, 0), 2)
        # 退出提示
        if self.exit_start_time != 0:
            remain = self.exit_delay - (time.time() - self.exit_start_time)
            cv2.putText(img, f"EXIT IN: {remain:.1f}s",
                        (self.wCam // 2 - 100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img
    #供外部调用的帧处理接口
    def process_frame(self, img):
        # img = cv2.flip(img, 1)
        img = self.detector.findHands(img)
        lmList, _ = self.detector.findPosition(img, draw=False)
        img, should_exit = self._process_gestures(img, lmList)# 处理手势
        # 返回处理后的帧和退出标志
        return self._draw_interface(img), should_exit
    #独立运行模式入口
    def run(self):
        try:
            while True:
                success, img = self.cap.read()# 读取摄像头帧
                if not success:
                    self._handle_camera_error()
                    continue
                img, should_exit = self.process_frame(img)# 处理当前帧
                cv2.imshow("Volume Control", img)# 显示画面
                # 退出条件检测
                if should_exit or cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.release()
    #摄像头异常处理
    def _handle_camera_error(self):
        if self.own_camera:# 仅处理自有摄像头
            print("Camera error, reinitializing...")
            self.cap.release()
            self.cap = cv2.VideoCapture(0)
            self._setup_camera()
    #安全释放资源
    def release(self):
        if self.own_camera and self.cap.isOpened():# 安全释放自有摄像头
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 正确初始化摄像头实例
    camera = cv2.VideoCapture(0)# 创建摄像头对象
    if not camera.isOpened():
        print("Error: Could not open camera")
        exit()
    try:
        vc = VolumeController(camera)# 创建控制器（共享摄像头模式）
        vc.run()
    finally:
        if camera.isOpened():
            camera.release()
