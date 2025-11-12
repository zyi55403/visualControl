import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot


class BlinkDetector:
    def __init__(self, max_faces=1, ratio_window=5, blink_threshold=24, reset_frames=10):
        self.detector = FaceMeshDetector(maxFaces=max_faces)
        # 定义双眼关键点索引（上、下、内、外）
        self.left_eye_ids = [159, 23, 130, 243]
        self.right_eye_ids = [386, 374, 362, 263]
        self.id_list = self.left_eye_ids + self.right_eye_ids
        self.plot = LivePlot(640, 360, [15, 55], invert=True)

        # 眨眼检测参数
        self.ratio_list = []
        self.blink_counter = 0
        self.frame_counter = 0
        self.color = (255, 0, 255)
        self.ratio_window = ratio_window
        self.blink_threshold = blink_threshold
        self.initial_threshold = blink_threshold  # 保存初始阈值
        self.reset_frames = reset_frames

        # 状态标志
        self.is_blinking = False
        self.previous_ratio_above_threshold = True

        # 自适应阈值参数
        self.peaks_and_valleys = []  # 存储有效的波峰波谷对
        self.current_peak = None  # 当前未配对的波峰
        self.current_valley = None  # 当前未配对的波谷
        self.last_ratio = None  # 上一帧的EAR值
        self.rising = None  # 上一帧是否处于上升趋势
    #实现自适应阈值
    def update_threshold(self):
        if self.peaks_and_valleys:
            total = 0
            for peak, valley in self.peaks_and_valleys:
                total += (peak + valley) / 2
            self.blink_threshold = total / len(self.peaks_and_valleys)

    def process_frame(self, img, draw=True):
        img, faces = self.detector.findFaceMesh(img, draw=False)
        if faces:
            face = faces[0]
            if draw:
                for id in self.id_list:
                    cv2.circle(img, face[id], 5, self.color, cv2.FILLED)

            left_up = face[self.left_eye_ids[0]]
            left_down = face[self.left_eye_ids[1]]
            left_in = face[self.left_eye_ids[2]]
            left_out = face[self.left_eye_ids[3]]

            right_up = face[self.right_eye_ids[0]]
            right_down = face[self.right_eye_ids[1]]
            right_in = face[self.right_eye_ids[2]]
            right_out = face[self.right_eye_ids[3]]

            # 计算左右眼EAR
            length_ver_left, _ = self.detector.findDistance(left_up, left_down)
            length_hor_left, _ = self.detector.findDistance(left_in, left_out)
            ear_left = (length_ver_left / length_hor_left) * 100

            length_ver_right, _ = self.detector.findDistance(right_up, right_down)
            length_hor_right, _ = self.detector.findDistance(right_in, right_out)
            ear_right = (length_ver_right / length_hor_right) * 100

            current_ratio = (ear_left + ear_right) / 2
            #current_ratio = min(ear_left,ear_right)
            self.ratio_list.append(current_ratio)

            # 滑动窗口处理
            if len(self.ratio_list) > self.ratio_window:
                self.ratio_list.pop(0)
            ratio_avg = sum(self.ratio_list) / len(self.ratio_list)

            # 波峰波谷检测（更新阈值）
            if self.last_ratio is not None:
                # 判断当前EAR值是否比上一帧大（上升趋势）
                current_rising = current_ratio > self.last_ratio

                if self.rising is not None and current_rising != self.rising:
                    # 之前是上升趋势，现在变为下降趋势
                    if self.rising:  # 之前是上升趋势
                        # 记录波峰（使用前一个值作为极值）
                        new_peak = self.last_ratio
                        if self.current_valley is not None:
                            # 检查是否为有效波动
                            if new_peak - self.current_valley >= 5:
                                self.peaks_and_valleys.append((new_peak, self.current_valley))
                                # 保持最多5个有效对
                                if len(self.peaks_and_valleys) > 3:
                                    self.peaks_and_valleys.pop(0)
                                # 更新阈值
                                self.update_threshold()
                            self.current_valley = None
                        else:
                            # 暂时存储当前波峰
                            self.current_peak = new_peak
                    else:  # 之前是下降趋势，现在变为上升趋势
                        # 记录波谷
                        new_valley = self.last_ratio
                        if self.current_peak is not None:
                            # 检查有效波动
                            if self.current_peak - new_valley >= 5:
                                # 将有效的波峰波谷对存入列表
                                self.peaks_and_valleys.append((self.current_peak, new_valley))
                                # 保持最多5个有效对
                                if len(self.peaks_and_valleys) > 3:
                                    self.peaks_and_valleys.pop(0)
                                # 更新阈值
                                self.update_threshold()
                            self.current_peak = None
                        else:
                            #暂时存储当前波谷
                            self.current_valley = new_valley
                    # 更新趋势状态
                    self.rising = current_rising
                else:
                    # 首次检测时设置趋势
                    self.rising = current_rising
                # 更新最后比率值
                self.last_ratio = current_ratio
            else:
                # 初始化第一个值
                self.last_ratio = current_ratio

            # 当前帧EAR是否低于阈值
            current_below = ratio_avg < self.blink_threshold
            # 当前帧EAR是否高于阈值
            current_above = not current_below

            if current_below and self.previous_ratio_above_threshold and self.frame_counter == 0:
                # 触发条件：①当前低于阈值 ②上一帧高于阈值 ③防抖计数器未激活(避免重复计数眨眼)
                self.blink_counter += 1 # 眨眼次数+1
                self.color = (0, 255, 0) # 标记颜色改为绿色（检测到眨眼）
                self.frame_counter = 1 # 启动防抖计数器
                self.is_blinking = True # 设置眨眼状态
            else:
                self.is_blinking = False # 未检测到有效眨眼
            # 保存当前阈值状态用于下一帧
            self.previous_ratio_above_threshold = current_above
            # 防抖计数器运行中
            if self.frame_counter != 0:
                self.frame_counter += 1
                if self.frame_counter > self.reset_frames: # 超过防抖帧数（10帧）
                    self.frame_counter = 0 # 重置计数器
                    self.color = (255, 0, 255)  #恢复默认颜色

            if draw:
                # 绘制眼部连线
                cv2.line(img, left_up, left_down, (0, 255, 0), 3)
                cv2.line(img, left_in, left_out, (0, 255, 0), 3)
                cv2.line(img, right_up, right_down, (0, 255, 0), 3)
                cv2.line(img, right_in, right_out, (0, 255, 0), 3)

                # 显示当前阈值
                cvzone.putTextRect(img, f'Threshold: {self.blink_threshold:.1f}', (50, 150),
                                   colorR=self.color)
                cvzone.putTextRect(img, f'Blinks: {self.blink_counter}', (50, 100), colorR=self.color)

                plot_img = self.plot.update(ratio_avg, self.color)
                img = cv2.resize(img, (640, 360))
                return cvzone.stackImages([img, plot_img], 2, 1), self.is_blinking

            return img, self.is_blinking
        else:
            # 重置检测状态
            self.last_ratio = None
            self.rising = None
            self.current_peak = None
            self.current_valley = None

            img = cv2.resize(img, (640, 360))
            return cvzone.stackImages([img, img], 2, 1), False

    def get_blink_count(self):
        return self.blink_counter

    def reset_counter(self):
        self.blink_counter = 0
        self.frame_counter = 0
        self.color = (255, 0, 255)
        self.ratio_list = []
        self.previous_ratio_above_threshold = True
        self.blink_threshold = self.initial_threshold  # 重置为初始阈值
        self.peaks_and_valleys = []


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    blink_detector = BlinkDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        processed_img, is_blinking = blink_detector.process_frame(img)
        cv2.imshow("Smart Blink Detection", processed_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
