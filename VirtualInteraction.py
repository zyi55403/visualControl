import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import PIL.Image, PIL.ImageTk
import cv2 #计算机视觉库，用于摄像头捕获、图像处理
import numpy as np
import time
import pandas as pd #用于数据分析和处理，主要操作CSV格式的数据集。
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib
import os #处理文件和目录路径
import handtrackingmodule as htm
from volumecontrol import VolumeController
from voiceinput import VoiceInputModule
import autopy as ap
import pyautogui as pg
from threading import Thread
from AiPaintingmodule import AIPaintingGUI

class GestureControlGUI:
    def __init__(self, root):
        self.root = root
        try:
            if isinstance(root, tk.Toplevel):
                self.root.transient(root.master)
        except Exception:
            pass
        self.root.title("手势控制系统")
        self.wCam, self.hCam = 640, 360
        self.pTime = 0
        self.smoothening = 10
        self.plocX, self.plocY = 0, 0
        self.wScr, self.hScr = ap.screen.size()
        # 各种状态标志
        self.running = True  # 运行状态标志
        self.interaction_active = False  # 交互模式是否激活
        self.volume_control_mode = False  # 音量控制模式
        self.input_active = False  # 防止多次开始采集
        self.collecting = False  # 是否正在采集数据
        self.pred_mode = False  # 预测模式是否激活
        self.show_collect_timer = False  # 是否显示采集计时器
        self.countdown_active = False  # 倒计时是否激活
        # 时间相关变量
        self.start_time = time.time()
        self.current_mode = None
        self.last_print_time = 0
        self.countdown_start = 0
        self.collection_start_time = 0
        # 数据采集参数
        self.COLLECT_INTERVAL = 5  # 采集间隔
        self.COLLECT_DURATION = 10  # 采集持续时间
        self.samples = []  # 存储采集的样本
        self.gesture_name = ""  # 手势名称
        # 预测/交互参数
        self.PREDICTION_THRESHOLD = 0.9
        self.frame1 = 160  # 检测区域边界
        self.frame2 = 100
        self.PREDICT_GESTURES = ['up', 'down', 'left', 'right', 'volume', 'voice', 'space']
        self.DIRECT_GESTURES = ['move', 'lclick', 'rclick']
        self.GESTURE_HOLD_DURATION = 1  # 手势保持时间
        self.current_gesture = {'name': None, 'start_time': None}  # 当前手势
        self.last_lclick_time = 0  # 上次左键点击时间
        self.last_rclick_time = 0  # 上次右键点击时间
        self.CLICK_COOLDOWN = 0.3  # 点击冷却时间
        # 初始化模块
        self.detector = htm.handDetector(maxHands=1)
        self.vim = VoiceInputModule()
        self.scaler = None
        self.knn = None
        self._load_model()  # 加载现有模型
        # GUI设置
        self.gesture_combo_var = tk.StringVar()
        self.create_widgets()
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("摄像头错误", "无法打开摄像头", parent=self.root)
            self.root.destroy()
            return
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        # 初始化音量控制器
        self.vc = VolumeController(cap=self.cap)
        # 开始更新循环
        self.update_frame()
        # 窗口关闭处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.threads = []  # 跟踪所有后台线程

    # 模型加载方法
    def _load_model(self):
        model_path = 'knn_model.pkl'
        scaler_path = 'scaler.pkl'
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.knn = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.pred_mode = True
                print("模型加载成功。")  # 使用 print 输出到控制台日志
                self.log_message("模型加载成功。")  # 同时记录到 GUI 日志
            except Exception as e:
                print(f"加载模型时出错: {e}")
                self.log_message(f"加载模型时出错: {e}")
                self.knn = None
                self.scaler = None
                self.pred_mode = False
        else:
            self.pred_mode = False
            print("未找到模型文件。请先训练模型。")
            self.log_message("未找到模型文件。请先训练模型。")

    # 创建GUI控件方法
    def create_widgets(self):
        try:
            # 主容器使用 grid 布局，确保扩展性
            main_container = tk.Frame(self.root)
            main_container.pack(fill=tk.BOTH, expand=True)
            self.root.grid_columnconfigure(0, weight=1)
            self.root.grid_rowconfigure(0, weight=1)

            # 配置左右框架比例 2:1
            main_container.grid_columnconfigure(0, weight=2)
            main_container.grid_columnconfigure(1, weight=1)
            main_container.grid_rowconfigure(0, weight=1)

            # --- 左框架 (手势控制区域) ---
            left_frame = tk.Frame(main_container, bd=2, relief=tk.SUNKEN)
            left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

            # 配置左框架布局权重
            left_frame.grid_columnconfigure(0, weight=1)
            left_frame.grid_rowconfigure(2, weight=1)  # 摄像头区域可扩展

            # 标题
            left_title = tk.Label(
                left_frame,
                text="自定义手势",
                font=("Arial", 14, "bold"),
                bg="#f0f0f0"
            )
            left_title.grid(row=0, column=0, pady=(10, 5), sticky="ew")

            # 手势选择组合框
            gesture_subframe = tk.Frame(left_frame)
            gesture_subframe.grid(row=1, column=0, pady=5, sticky="ew")
            gesture_subframe.grid_columnconfigure(1, weight=1)  # 组合框扩展

            lbl_gesture = tk.Label(gesture_subframe, text="手势名称:")
            lbl_gesture.grid(row=0, column=0, padx=(5, 0), sticky='w')

            self.gesture_combo = ttk.Combobox(
                gesture_subframe,
                textvariable=self.gesture_combo_var,
                values=self.PREDICT_GESTURES,
                state="readonly"
            )
            self.gesture_combo.set("选择手势")
            self.gesture_combo.grid(row=0, column=1, padx=5, sticky='ew')

            self.train_button = tk.Button(
                gesture_subframe,
                text="开始训练",
                command=self.start_collection_thread
            )
            self.train_button.grid(row=0, column=2, padx=(0, 5), sticky='e')

            # 摄像头显示区域
            self.camera_label_left = tk.Label(left_frame, bg="black")
            self.camera_label_left.grid(row=2, column=0, pady=(0, 5), sticky="nsew")

            # --- 右框架 (交互控制区域) ---
            right_frame = tk.Frame(main_container, bd=2, relief=tk.SUNKEN, width=300)
            right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
            right_frame.grid_propagate(False)  # 固定宽度

            # 配置右框架布局权重
            right_frame.grid_columnconfigure(0, weight=1)
            right_frame.grid_rowconfigure(3, weight=1)  # 日志区域可扩展

            right_title = tk.Label(
                right_frame,
                text="电脑交互",
                font=("Arial", 14, "bold"),
                bg="#f0f0f0"
            )
            right_title.grid(row=0, column=0, pady=(10, 5), sticky="ew")

            self.interact_button = tk.Button(
                right_frame,
                text="开始交互",
                command=self.toggle_interaction
            )
            self.interact_button.grid(row=1, column=0, pady=10, padx=20, sticky="ew")

            # 日志区域
            self.status_log = scrolledtext.ScrolledText(
                right_frame,
                wrap=tk.WORD,
                font=("Consolas", 9),
                state=tk.DISABLED
            )
            self.status_log.grid(row=3, column=0, pady=5, padx=10, sticky="nsew")

        except Exception as e:
            messagebox.showerror(
                "GUI初始化错误",
                f"界面创建失败: {str(e)}",
                parent=self.root
            )
            if self.root.winfo_exists():
                self.root.destroy()

    # 日志记录方法
    def log_message(self, message):
        # 确保 GUI 更新在主线程中进行
        def update_gui():
            if self.root.winfo_exists():  # 检查窗口是否仍然存在
                self.status_log.config(state=tk.NORMAL)  # 允许写入
                self.status_log.insert(tk.END, message + "\n")
                self.status_log.see(tk.END)  # 滚动到底部
                self.status_log.config(state=tk.DISABLED)  # 禁止写入

        # 仅当根窗口仍然存在时才使用 after()
        if self.running and self.root.winfo_exists():
            self.root.after(0, update_gui)
        else:
            print(f"日志 (窗口已关闭): {message}")  # 如果 GUI 不存在，则记录到控制台

    # 切换交互模式方法
    def toggle_interaction(self):
        if self.interaction_active:
            self.interaction_active = False
            self.interact_button.config(text="开始交互")
            self.current_mode = None
            self.log_message("交互已停止.")
            print("交互已停止.")
        else:
            if not self.pred_mode or self.knn is None or self.scaler is None:
                messagebox.showwarning("模型未就绪", "请先训练模型.", parent=self.root)
                return
            self.interaction_active = True
            self.interact_button.config(text="停止交互")
            self.log_message("交互已开始.")
            print("交互已开始.")
            # 清除任何采集/训练状态
            self.collecting = False
            self.countdown_active = False
            self.show_collect_timer = False
            self.samples = []

    # 特征提取方法
    def extract_features(self, lmList, bbox):
        if not lmList or len(lmList) < 21:  # 确保21个手部关键点存在
            return None
        features = []
        try:
            fingers = self.detector.fingersUp()
            if fingers is None:
                return None
            features.extend(fingers)
            base_x, base_y = lmList[0][1], lmList[0][2]
            # 计算关键点相对于手掌基点的偏移量
            for i in [4, 8, 12, 16, 20, 5, 9, 13]:  # 8个关键点
                dx = lmList[i][1] - base_x
                dy = lmList[i][2] - base_y
                features.extend([dx, dy])
            # 边界框特征（宽高）
            if bbox and len(bbox) == 4:
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                features.extend([bbox_w, bbox_h])
            else:
                features.extend([0, 0])
            if len(features) == 23:
                return features
            else:
                return None
        except IndexError:
            return None
        except Exception as e:
            return None

    # 训练模型方法
    def train_model(self):
        self.log_message("开始训练模型...")
        success_flag = False  # 标记训练是否成功
        # 配置前检查按钮是否存在
        if hasattr(self, 'train_button') and self.train_button.winfo_exists():
            self.train_button.config(state=tk.DISABLED)  # 开始训练，禁用训练按钮
        try:
            data_path = 'dataset/data.csv'
            if not os.path.exists(data_path):
                self.log_message(f"错误: 数据集未找到 {data_path}")
                messagebox.showerror("训练错误", f"数据集未找到:\n{data_path}", parent=self.root)
                return False
            df = pd.read_csv(data_path)
            self.log_message(f"加载数据集 {data_path}，包含 {df.shape[0]} 条记录。")
            if df.isnull().values.any():
                self.log_message("警告: 数据集包含 NaN 值，尝试删除包含NaN的行。")
                initial_rows = df.shape[0]
                df.dropna(inplace=True)
                self.log_message(f"删除 NaN 后剩余 {df.shape[0]} 行 (原 {initial_rows} 行)。")
                if df.empty:
                    self.log_message("错误: 删除 NaN 值后数据集为空。")
                    messagebox.showerror("训练错误", "数据集在删除无效条目后为空。", parent=self.root)
                    return False
            min_samples_per_gesture = 15
            if 'label' not in df.columns:
                self.log_message("错误: 数据集中缺少 'label' 列。")
                messagebox.showerror("训练错误", "数据集缺少 'label' 列。", parent=self.root)
                return False
            gesture_counts = df['label'].value_counts()
            self.log_message(f"各手势样本数量:\n{gesture_counts}")
            if not gesture_counts.empty and gesture_counts.min() < min_samples_per_gesture:
                low_sample_gestures = gesture_counts[gesture_counts < min_samples_per_gesture].index.tolist()
                warn_msg = f"警告: 以下手势样本数量不足 {min_samples_per_gesture} 个: {', '.join(low_sample_gestures)}。\n模型效果可能不佳。"
                self.log_message(warn_msg)
                messagebox.showwarning("训练警告", warn_msg + "\n建议为这些手势采集更多数据。", parent=self.root)
            elif gesture_counts.empty:
                self.log_message("警告: 数据集中没有标签或标签计数为空。")
            if df.shape[0] < 50:
                self.log_message(f"警告: 训练数据可能不足 ({df.shape[0]} 个样本)，建议至少 50 个。")
            expected_cols = 24
            if df.shape[1] != expected_cols:
                self.log_message(f"错误: 数据集列数不正确 ({df.shape[1]} 列)。期望 {expected_cols} 列。")
                messagebox.showerror("训练错误", f"数据集格式错误。期望 {expected_cols} 列，实际为 {df.shape[1]} 列。",
                                     parent=self.root)
                return False
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            if X.size == 0 or y.size == 0:
                self.log_message("错误: 未能从数据集中提取特征或标签。")
                messagebox.showerror("训练错误", "无法从 CSV 文件中提取数据。", parent=self.root)
                return False
            # 检查是否有足够的样本用于带分层的 train_test_split
            unique_labels, counts = np.unique(y, return_counts=True)
            if np.any(counts < 2):  # 每个类别至少需要 2 个样本才能进行分层
                problematic_labels = unique_labels[counts < 2]
                self.log_message(f"错误: 以下标签样本数不足2个，无法进行分层分割: {problematic_labels}")
                messagebox.showerror("训练错误",
                                     f"以下手势样本太少无法训练: {', '.join(map(str, problematic_labels))}\n请采集更多数据。",
                                     parent=self.root)
                return False
            # 带分层的数据分割
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                self.log_message(f"数据分割完成：训练集 {X_train.shape[0]} 条，测试集 {X_test.shape[0]} 条。")
            except ValueError as e:
                self.log_message(f"数据分割错误（可能由于样本不足）: {e}")
                messagebox.showerror("训练错误", f"数据分割失败，请确保每个手势至少有2个样本。\n错误: {e}",
                                     parent=self.root)
                return False
            self.scaler = MinMaxScaler().fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            self.log_message("数据缩放完成 (MinMaxScaler)。")
            X_train_scaled[:, 0:5] *= 2
            X_test_scaled[:, 0:5] *= 2
            self.log_message("特征加权完成 (手指状态特征 * 2)。")
            # 安全地确定 n_neighbors
            n_unique_classes = len(np.unique(y_train))
            n_samples_train = X_train_scaled.shape[0]
            # k 不应大于分割后最小类别中的样本数，理想情况下小于总样本数
            k_max = min(n_samples_train, 11)  # 限制 k 以提高性能和稳定性
            n_neighbors = min(k_max, n_unique_classes if n_unique_classes > 0 else 1)
            n_neighbors = max(1, n_neighbors)  # k 必须至少为 1
            self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='manhattan',
                                            algorithm='auto')
            self.log_message(f"开始训练 KNN 模型 (k={n_neighbors}, metric=manhattan, weights=distance)...")
            self.knn.fit(X_train_scaled, y_train)
            self.log_message("KNN 模型训练完成。")
            y_pred = self.knn.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred) * 100
            self.log_message(f"模型在测试集上的准确率: {accuracy:.1f}%")
            joblib.dump(self.knn, 'knn_model.pkl')
            joblib.dump(self.scaler, 'scaler.pkl')
            self.pred_mode = True
            self.log_message("模型训练完成并已保存 (knn_model.pkl, scaler.pkl)。")
            messagebox.showinfo("训练完成", f"模型训练成功!\n准确率: {accuracy:.1f}%", parent=self.root)
            success_flag = True
            return True
        except FileNotFoundError:
            error_msg = f"错误: 数据集文件 '{data_path}' 未找到。"
            self.log_message(error_msg)
            messagebox.showerror("训练错误", f"数据集文件未找到:\n{data_path}", parent=self.root)
            return False
        except ValueError as e:
            error_msg = f"训练值错误: {e}"
            self.log_message(error_msg)
            messagebox.showerror("训练错误", f"训练期间数据错误: {e}", parent=self.root)
            return False
        except Exception as e:
            error_msg = f"训练失败: {e}"
            self.log_message(error_msg)
            messagebox.showerror("训练错误", f"训练期间发生意外错误:\n{type(e).__name__}: {e}", parent=self.root)
            return False
        finally:
            # 仅当按钮存在且窗口未关闭时才重新启用按钮
            if hasattr(self, 'train_button') and self.train_button.winfo_exists():
                self.train_button.config(state=tk.NORMAL)

    # 执行动作方法
    def perform_action(self, action):
        current_time = time.time()
        # 音量控制模式入口
        if action == "volume":
            if not self.volume_control_mode:
                self.volume_control_mode = True
                self.current_mode = "volume"
                print("进入音量控制模式")
                self.log_message("进入音量控制模式")
            return
        # 语音输入模式入口
        elif action == "voice":
            if hasattr(self, 'vim') and self.vim is not None:
                if not self.vim.running:
                    self.log_message("启动语音输入...")
                    print("启动语音输入...")
                    # 确保在启动线程前 vim 已初始化
                    if not hasattr(self.vim,
                                   'is_listening') or not self.vim.is_listening():  # Add check if vim has state
                        Thread(target=self.vim.start_listening, daemon=True).start()
                    self.current_mode = "voice"
                else:
                    print("语音输入已在运行。")
                    self.log_message("语音输入已在运行。")
            else:
                print("语音输入模块未初始化.")
                self.log_message("错误：语音输入模块未初始化。")
            return
        # 如果在音量或语音模式，不处理其他手势
        if self.volume_control_mode or (hasattr(self, 'vim') and self.vim and self.vim.running):
            return
        # 执行各种动作（滚动、按键、点击等）
        # 仅在非音量/语音模式下设置模式
        if not self.volume_control_mode and not (hasattr(self, 'vim') and self.vim and self.vim.running):
            self.current_mode = action
        self.log_message(f"触发动作: {action}")
        print(f"触发动作: {action}")
        if action == "up":
            pg.scroll(-300)
            self.log_message("动作: 上一页 ")
            print("动作: 上一页 ")
        elif action == "down":
            pg.scroll(300)
            self.log_message("动作: 下一页 ")
            print("动作: 下一页 ")
        elif action == "left":
            pg.press('left')
            self.log_message("动作: 上一首 ")
            print("动作: 上一首 ")
        elif action == "right":
            pg.press('right')
            self.log_message("动作: 下一首 ")
            print("动作: 下一首 ")
        elif action == "space":
            pg.press('space')
            self.log_message("动作: 播放/暂停 ")
            print("动作: 播放/暂停 ")

    # 鼠标控制逻辑
    def mouse_control_logic(self, img, lmList):
        # 检查语音输入是否激活 - 如果是，则禁用鼠标控制
        if hasattr(self, 'vim') and self.vim and self.vim.running:
            return False
        fingers = None
        try:
            fingers = self.detector.fingersUp()
        except Exception as e:
            print(f"获取手指状态时出错: {e}")
            self.log_message(f"获取手指状态时出错: {e}")
            return False
        if fingers is None:
            return False
        # 模式 1: 移动鼠标 (食指竖起，其他手指放下)
        if fingers[1] == 1 and sum(fingers[2:]) == 0:
            x1, y1 = lmList[8][1:]
            # 将坐标映射到屏幕
            effective_wCam = max(1, self.wCam - 2 * self.frame1)
            effective_hCam = max(1, self.hCam - 2 * self.frame2)
            x3 = np.interp(x1, (self.frame1, self.wCam - self.frame1), (0, self.wScr))
            y3 = np.interp(y1, (self.frame2, self.hCam - self.frame2), (0, self.hScr))
            # 平滑移动
            clocX = self.plocX + (x3 - self.plocX) / self.smoothening
            clocY = self.plocY + (y3 - self.plocY) / self.smoothening
            try:
                # 将坐标限制在屏幕边界内，以防插值略微超出
                move_x = int(np.clip(clocX, 0, self.wScr - 1))
                move_y = int(np.clip(clocY, 0, self.hScr - 1))
                ap.mouse.move(move_x, move_y)
                self.plocX, self.plocY = clocX, clocY  # 使用平滑值更新先前位置
            except ValueError as e:
                # 如果坐标是 NaN 或无穷大，可能会发生这种情况
                print(f"移动鼠标时出错: {e}. 坐标: ({clocX}, {clocY})")
                pass
            except Exception as e:  # 捕获潜在的 autopy 错误
                print(f"Autopy 鼠标移动错误: {e}")
                pass
            # 视觉反馈：食指指尖上的圆圈
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            if not self.volume_control_mode and not (hasattr(self, 'vim') and self.vim and self.vim.running):
                self.current_mode = "move"
            return True  # 鼠标被控制了
        # 模式 2: 点击 (食指和中指竖起)
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            length, img, lineInfo = self.detector.findDistance(8, 12, img)
            if length is not None and length < 40:
                current_time = time.time()
                # 用于视觉反馈圆圈的中心点
                cx, cy = lmList[8][1], lmList[8][2]  # 默认为食指指尖
                if lineInfo and len(lineInfo) >= 4:  # 如果线存在，则使用中点
                    cx = (lineInfo[0] + lineInfo[2]) // 2
                    cy = (lineInfo[1] + lineInfo[3]) // 2
                # 左键点击: 拇指放下 (fingers[0] == 0)
                if fingers[0] == 0:
                    if current_time - self.last_lclick_time > self.CLICK_COOLDOWN:
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                        try:
                            ap.mouse.click(ap.mouse.Button.LEFT)
                            self.log_message("动作: 左键点击 ")
                            print("动作: 左键点击 ")
                            self.last_lclick_time = current_time
                            if not self.volume_control_mode and not (
                                    hasattr(self, 'vim') and self.vim and self.vim.running):
                                self.current_mode = "lclick"
                            return True  # 鼠标被控制了 (点击)
                        except Exception as e:
                            print(f"Autopy 左键点击错误: {e}")
                            pass
                # 右键点击: 拇指竖起 (fingers[0] == 1)
                elif fingers[0] == 1:
                    if current_time - self.last_rclick_time > self.CLICK_COOLDOWN:
                        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                        try:
                            ap.mouse.click(ap.mouse.Button.RIGHT)
                            print("动作: 右键点击 ")
                            self.log_message("动作: 右键点击 ")
                            self.last_rclick_time = current_time
                            if not self.volume_control_mode and not (
                                    hasattr(self, 'vim') and self.vim and self.vim.running):
                                self.current_mode = "rclick"
                            return True  # 鼠标被控制了 (点击)
                        except Exception as e:
                            print(f"Autopy 右键点击错误: {e}")
                            pass
            # 即使未点击 (length >= 40)，视觉上仍将其视为鼠标控制模式
            if lineInfo and len(lineInfo) >= 4 and length is not None:
                cv2.line(img, (lineInfo[0], lineInfo[1]), (lineInfo[2], lineInfo[3]), (0, 255, 0), 3)
            # 除非发生点击，否则此处不设置模式，但返回 True，因为此手势优先
            return True
        return False

    # 添加方法统一处理训练按钮状态
    def _toggle_train_button(self, state):
        if hasattr(self, 'train_button') and self.train_button.winfo_exists():
            self.train_button.config(state=state)
    # 数据采集方法
    def _start_collect_data(self):
        try:
            selected_gesture = self.gesture_combo_var.get()
            if not selected_gesture or selected_gesture == "选择手势":
                self.log_message("错误: 未选择手势进行采集。")
                # 从主线程安全地重新启用按钮
                self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL) if hasattr(self,
                                                                                                'train_button') and self.train_button.winfo_exists() else None)
                return
            self.gesture_name = selected_gesture
            self.log_message(f"准备数据采集: {self.gesture_name}")
            # 开始 5 秒倒计时
            self.countdown_start = time.time()
            self.countdown_active = True
            self.collecting = False  # 确保倒计时期间不采集
            self.show_collect_timer = False
            self.samples = []  # 清除以前的样本
            self.log_message("开始 5 秒倒计时...")
            # 倒计时循环 (检查 running 标志)
            while time.time() - self.countdown_start < 5 and self.running and self.countdown_active:
                time.sleep(0.1)  # 短暂睡眠以避免忙等待
            # 检查是否在倒计时期间取消或应用关闭
            if not self.running or not self.countdown_active:
                self.log_message("倒计时期间取消或应用关闭。")
                self.countdown_active = False
                # 从主线程安全地重新启用按钮
                self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL) if hasattr(self,
                                                                                                'train_button') and self.train_button.winfo_exists() else None)
                return
            # 开始 10 秒数据采集
            self.countdown_active = False
            self.collecting = True
            self.collection_start_time = time.time()
            self.show_collect_timer = True
            self.log_message(f"开始10秒数据采集 '{self.gesture_name}'...")
            # 采集循环 (检查 running 标志)
            while time.time() - self.collection_start_time < self.COLLECT_DURATION and self.running and self.collecting:
                time.sleep(0.1)
            # 检查是否在采集期间取消或应用关闭
            if not self.running or not self.collecting:
                self.log_message("采集期间取消或应用关闭。")
                self.collecting = False
                self.show_collect_timer = False
                # 从主线程安全地重新启用按钮
                self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL) if hasattr(self,'train_button') and self.train_button.winfo_exists() else None)
                return
            # 正常完成采集
            self.collecting = False
            self.show_collect_timer = False
            self.log_message(f"完成采集 {len(self.samples)} 个样本 '{self.gesture_name}'.")
            # 保存数据 (确保样本存在)
            if self.samples:
                # 筛选有效特征 (长度正确)
                valid_samples = [s for s in self.samples if s is not None and len(s) == 23]
                if len(valid_samples) != len(self.samples):
                    self.log_message(f"警告: 丢弃 {len(self.samples) - len(valid_samples)} 个无效或长度不正确的样本。")
                if not valid_samples:
                    self.log_message("错误: 未采集到有效样本。无法保存数据。")
                    # 在主线程中使用 root.after 显示 messagebox
                    self.root.after(0, lambda: messagebox.showerror("采集错误", "未采集到有效数据点。", parent=self.root))
                    # 安全地重新启用按钮
                    self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL) if hasattr(self,
                                                                                                    'train_button') and self.train_button.winfo_exists() else None)
                    return
                # 为新数据创建 DataFrame
                df_new = pd.DataFrame(valid_samples,
                                      columns=[f'f{i}' for i in range(5)] +
                                              [f'p{i}_{axis}' for i in [4, 8, 12, 16, 20, 5, 9, 13] for axis in
                                               ['dx', 'dy']] +
                                              ['bbox_w', 'bbox_h'])
                df_new['label'] = self.gesture_name
                # 确保数据集目录存在
                data_dir = 'dataset'
                data_path = os.path.join(data_dir, 'data.csv')
                try:
                    if not os.path.exists(data_dir):
                        os.makedirs(data_dir)
                        self.log_message(f"创建目录: {data_dir}")
                except OSError as e:
                    self.log_message(f"创建目录 {data_dir} 时出错: {e}")
                    self.root.after(0, lambda: messagebox.showerror("文件错误", f"无法创建目录:\n{data_dir}\n错误: {e}",
                                                                    parent=self.root))
                    self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL) if hasattr(self,
                                                                                                    'train_button') and self.train_button.winfo_exists() else None)
                    return
                # 与现有数据合并 (处理潜在错误)
                df_combined = None
                if os.path.exists(data_path):
                    try:
                        df_old = pd.read_csv(data_path)
                        # 基本检查列是否大致匹配 (不包括标签)
                        if list(df_old.columns[:-1]) == list(df_new.columns[:-1]):
                            df_combined = pd.concat([df_old, df_new], ignore_index=True)
                            self.log_message("新数据已追加到现有 data.csv。")
                        else:
                            self.log_message("警告: 现有CSV列结构不匹配。将用新数据覆盖旧文件。")
                            df_combined = df_new
                    except pd.errors.EmptyDataError:
                        self.log_message("警告: 现有 data.csv 为空。将保存新数据。")
                        df_combined = df_new
                    except Exception as e:
                        self.log_message(f"读取现有 data.csv 时出错: {e}. 将尝试覆盖。")
                        df_combined = df_new
                else:
                    df_combined = df_new
                    self.log_message("未找到现有 data.csv，将创建新文件。")
                # 保存合并后的数据
                if df_combined is not None:
                    try:
                        df_combined.to_csv(data_path, index=False)
                        self.log_message(f"成功保存数据。总样本数: {len(df_combined)}")
                        # 询问是否训练模型 (使用 root.after 在主线程显示 messagebox)
                        self.root.after(0, self._ask_to_train)  # 调用辅助函数显示 messagebox
                    except Exception as e:
                        self.log_message(f"保存数据到 CSV 时出错: {e}")
                        self.root.after(0,
                                        lambda: messagebox.showerror("保存错误", f"无法保存数据到:\n{data_path}\n错误: {e}",
                                                                     parent=self.root))
                        # 即使保存失败，也要重新启用按钮
                        self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL) if hasattr(self,'train_button') and self.train_button.winfo_exists() else None)
                else:
                    # 如果 valid_samples 存在，这种情况理论上不应该发生
                    self.log_message("错误: 合并数据后 df_combined 为 None。")
                    self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL) if hasattr(self,'train_button') and self.train_button.winfo_exists() else None)

            else:  # 完全没有采集到样本
                self.log_message("未采集到样本。")
                self.root.after(0, lambda: messagebox.showinfo("采集信息", "未采集到有效数据点。", parent=self.root))
                # 安全地重新启用按钮
                self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL) if hasattr(self,'train_button') and self.train_button.winfo_exists() else None)
        finally:
            self.root.after(0, lambda: self._toggle_train_button(tk.NORMAL))

    # 用于询问是否训练的辅助方法 (通过 after 从主线程调用)
    def _ask_to_train(self):
        if messagebox.askyesno("训练提示", f"已采集 '{self.gesture_name}' 的数据。\n是否现在训练模型？",
                               parent=self.root):
            # 如果是，则启动训练线程 (按钮状态由 start_training_thread 处理)
            self.start_training_thread()
        else:
            # 如果否，则仅重新启用按钮
            if hasattr(self, 'train_button') and self.train_button.winfo_exists():
                self.train_button.config(state=tk.NORMAL)

    # 线程启动方法
    def start_collection_thread(self):
        if self.interaction_active:
            messagebox.showwarning("交互模式激活", "请在采集数据前停止交互模式。", parent=self.root)
            return

        selected_gesture = self.gesture_combo_var.get()
        if not selected_gesture or selected_gesture == "选择手势":
            messagebox.showwarning("未选择手势", "请先从下拉列表中选择一个手势名称。", parent=self.root)
            return
        # 禁用按钮并启动采集线程
        if hasattr(self, 'train_button') and self.train_button.winfo_exists():
            self.train_button.config(state=tk.DISABLED)
        # 确保先前的采集没有在运行 (安全检查)
        self.collecting = False
        self.countdown_active = False
        collection_thread = Thread(target=self._start_collect_data, daemon=True)
        self.threads.append(collection_thread)
        collection_thread.start()

    # 在单独线程中启动模型训练
    def start_training_thread(self):
        if self.interaction_active:
            messagebox.showwarning("交互模式激活", "请在训练前停止交互模式。", parent=self.root)
            return
        if self.collecting or self.countdown_active:
            messagebox.showwarning("采集进行中", "请等待数据采集完成后再训练。", parent=self.root)
            return
        data_path = 'dataset/data.csv'
        if not os.path.exists(data_path):
            messagebox.showerror("训练错误", f"未找到数据集文件 '{data_path}'。\n请先采集数据。", parent=self.root)
            # 如果文件未找到，重新启用按钮
            if hasattr(self, 'train_button') and self.train_button.winfo_exists():
                self.train_button.config(state=tk.NORMAL)
            return
        # 禁用按钮并启动训练线程
        # 按钮状态将由 train_model 的 finally 块管理
        if hasattr(self, 'train_button') and self.train_button.winfo_exists():
            self.train_button.config(state=tk.DISABLED)
        training_thread = Thread(target=self.train_model, daemon=True)
        self.threads.append(training_thread)
        training_thread.start()

    # 主交互处理方法
    def main_interaction_process(self, img):
        # 绘制检测边界矩形
        cv2.rectangle(img, (self.frame1, self.frame2),
                      (self.wCam - self.frame1, self.hCam - self.frame2),
                      (255, 0, 255), 2)
        # 检测手，在交互模式下始终绘制手以提供反馈
        try:
            img = self.detector.findHands(img, draw=True)
            lmList, bbox = self.detector.findPosition(img, draw=False)  # 获取数据，不重绘点
        except Exception as e:
            print(f"手部检测/定位错误: {e}")
            self.log_message(f"手部检测错误: {e}")
            lmList, bbox = None, None
        mouse_controlled_this_frame = False
        if lmList:  # 检测到手
            # 优先级 1: 直接鼠标控制逻辑
            mouse_controlled_this_frame = self.mouse_control_logic(img, lmList)
            if mouse_controlled_this_frame:
                # 如果鼠标被直接控制，重置任何手势预测状态
                if self.current_gesture['name'] is not None:
                    self.current_gesture['name'] = None
                    self.current_gesture['start_time'] = None
            else:
                # 优先级 2: 手势预测 (如果鼠标未控制且模型就绪)
                if self.pred_mode and self.knn and self.scaler:
                    features = self.extract_features(lmList, bbox)
                    if features is not None:
                        try:
                            # 缩放和加权特征
                            features_scaled = self.scaler.transform([features])
                            features_scaled[:, 0:5] *= 2
                            # 预测概率
                            proba = self.knn.predict_proba(features_scaled)[0]
                            max_proba = np.max(proba)
                            pred_index = np.argmax(proba)
                            predicted_gesture = self.knn.classes_[pred_index]
                            if max_proba >= self.PREDICTION_THRESHOLD:
                                current_time = time.time()
                                # 检查是否是需要保持持续时间的手势
                                if predicted_gesture in self.PREDICT_GESTURES:
                                    if self.current_gesture['name'] != predicted_gesture:
                                        # 检测到新姿势，启动计时器
                                        self.current_gesture['name'] = predicted_gesture
                                        self.current_gesture['start_time'] = current_time
                                    else:
                                        # 保持相同手势，检查持续时间
                                        duration = current_time - self.current_gesture['start_time']
                                        # 视觉反馈：进度条
                                        progress = min(duration / self.GESTURE_HOLD_DURATION, 1.0)
                                        bar_width = 200
                                        cv2.rectangle(img, (50, self.hCam - 30),
                                                      (50 + bar_width, self.hCam - 10),
                                                      (100, 100, 100), cv2.FILLED)
                                        cv2.rectangle(img, (50, self.hCam - 30),
                                                      (int(50 + bar_width * progress), self.hCam - 10),
                                                      (0, 255, 0), cv2.FILLED)

                                        if duration >= self.GESTURE_HOLD_DURATION:
                                            # 达到持续时间，执行动作！
                                            self.perform_action(predicted_gesture)
                                            # 稍微重置计时器（如果保持则允许重复动作）
                                            # 调整因子（例如 0.8）以控制重新触发的灵敏度
                                            self.current_gesture['start_time'] = current_time - (self.GESTURE_HOLD_DURATION * 0.1)
                                else:
                                    # 预测的手势不需要保持（例如，可能是“中性”姿势？）
                                    # 如果出现不同的、高置信度的手势，则重置保持状态
                                    if self.current_gesture['name'] is not None:
                                        self.current_gesture['name'] = None
                                        self.current_gesture['start_time'] = None
                            else:  # 置信度低于阈值
                                # 如果置信度下降，重置保持的手势状态
                                if self.current_gesture['name'] is not None:
                                    self.current_gesture['name'] = None
                                    self.current_gesture['start_time'] = None
                        except ValueError as e:  # 如果 scaler 期望不同数量的特征，可能会发生
                            print(f"预测错误: {e}。请检查特征提取")
                            self.log_message(f"预测错误: {e}")
                            self.current_gesture['name'] = None
                            self.current_gesture['start_time'] = None
                        except Exception as e:
                            print(f"预测错误: {e}")
                            self.log_message(f"预测错误: {e}")
                            self.current_gesture['name'] = None
                            self.current_gesture['start_time'] = None
                    else:
                        if self.current_gesture['name'] is not None:
                            self.current_gesture['name'] = None
                            self.current_gesture['start_time'] = None
            # 在帧上显示当前模式
            mode_display_text = f"Mode: {self.current_mode}" if self.current_mode else "Mode: -"
            # 活动语音输入的特殊显示
            if hasattr(self, 'vim') and self.vim and self.vim.running:
                # 使语音状态高度可见
                cv2.putText(img, "[VOICE INPUT ACTIVE]", (self.wCam // 2 - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                mode_display_text = "Mode: voice"
                # 在左下角绘制模式文本
            text_size = cv2.getTextSize(mode_display_text, cv2.FONT_HERSHEY_PLAIN, 1.5, 2)[0]
            text_x = self.wCam - text_size[0] - 20  # 右侧留20像素边距
            cv2.putText(img, mode_display_text, (text_x, self.hCam - 40), cv2.FONT_HERSHEY_PLAIN,
                        1.5, (0, 255, 0), 2)
        else:
            # 如果手消失，重置保持的手势
            if self.current_gesture['name'] is not None:
                self.current_gesture['name'] = None
                self.current_gesture['start_time'] = None
            # 如果不在音量或语音控制模式下，则重置模式
            if not self.volume_control_mode and (not hasattr(self, 'vim') or not self.vim or not self.vim.running):
                if self.current_mode not in [None, "Idle"]:  # 仅当它处于活动状态时才重置
                    # print("Hand lost, resetting mode.")
                    self.current_mode = None
            # 如果没有手且不在其他模式下，则显示空闲模式
            if not self.volume_control_mode and not (hasattr(self, 'vim') and self.vim and self.vim.running):
                text = "Mode: Idle"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1.5, 2)[0]
                text_x = self.wCam - text_size[0] - 20
                cv2.putText(img, text, (text_x, self.hCam - 40), cv2.FONT_HERSHEY_PLAIN,
                            1.5, (200, 200, 200), 2)
        return img

    # 主更新循环
    def update_frame(self):
        """读取帧，根据模式处理，并更新 GUI。必须运行。"""
        if not self.running:
            # 如果循环意外停止，确保释放摄像头
            if self.cap and self.cap.isOpened():
                self.cap.release()
                print("摄像头已从 update_frame 退出时释放。")
                self.log_message("摄像头已从 update_frame 退出时释放。")
            return
        success, frame = self.cap.read()
        if not success:
            if self.root.winfo_exists():
                self.root.after(20, self.update_frame)
            else:
                self.running = False
            return
        frame = cv2.flip(frame, 1)
        processed_frame = frame.copy()
        # --- 模式处理逻辑 ---
        try:
            # 1. 音量控制模式 (抓取帧后的最高优先级)
            if self.volume_control_mode:
                if self.vc: #初始化好了音量模块
                    # 处理帧并检查是否退出音量模式
                    processed_frame_vc, should_exit = self.vc.process_frame(processed_frame.copy())
                    processed_frame = processed_frame_vc
                    if should_exit:
                        print("音量控制器请求退出。")
                        self.log_message("音量控制器请求退出。")
                        self.volume_control_mode = False
                        self.current_mode = None
                        # 退出音量模式时重置手势状态
                        self.current_gesture['name'] = None
                        self.current_gesture['start_time'] = None
                else:
                    self.log_message("错误: VolumeController 未初始化，退出音量模式。")
                    self.volume_control_mode = False
                    self.current_mode = None
            # 2. 主交互模式 (如果不在音量模式，则进入鼠标键盘控制模块)
            elif self.interaction_active:
                processed_frame = self.main_interaction_process(processed_frame)
            # 3. 数据采集倒计时 (如果不在音量或交互模式)
            elif self.countdown_active:
                # 在倒计时期间绘制手以提供定位反馈
                processed_frame = self.detector.findHands(processed_frame, draw=True)
                countdown_remain = 5 - (time.time() - self.countdown_start)
                if countdown_remain > 0:
                    # 显着显示倒数计时器
                    text = f"START IN: {int(countdown_remain) + 1}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                    text_x = (self.wCam - text_size[0]) // 2
                    text_y = (self.hCam + text_size[1]) // 2
                    cv2.putText(processed_frame, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
                # 注意：实际转换到采集中发生在 _start_collect_data 线程
            # 4. 数据采集中 (如果不在音量、交互或倒计时模式)
            elif self.collecting:
                # 在采集期间绘制手以提供视觉反馈
                processed_frame = self.detector.findHands(processed_frame, draw=True)
                lmList, bbox = self.detector.findPosition(processed_frame, draw=False)
                collect_remain = self.COLLECT_DURATION - (time.time() - self.collection_start_time)
                if collect_remain > 0:
                    if self.show_collect_timer:
                        # 显示采集计时器
                        text = f"RECORDING {self.gesture_name}: {int(collect_remain)}s"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
                        text_x = (self.wCam - text_size[0]) // 2
                        text_y = (self.hCam + text_size[1]) // 2
                        cv2.putText(processed_frame, text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                    # 为采集线程定期提取特征
                    if lmList:
                        current_time_collect = time.time()
                        # 以受控速率采样 (例如，最大约 20 Hz)
                        if current_time_collect - self.last_print_time > (1.0 / 20.0):
                            features = self.extract_features(lmList, bbox)
                            if features is not None:
                                self.samples.append(features)
                            self.last_print_time = current_time_collect
                # 注意：实际的采集结束/保存在 _start_collect_data 线程中处理
            # 5. 空闲模式 (如果以上都不是，则为默认模式)
            else:
                processed_frame = self.detector.findHands(processed_frame, draw=True)
                text = "Mode: Idle"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1.5, 2)[0]
                text_x = self.wCam - text_size[0] - 20
                cv2.putText(processed_frame, text, (text_x, self.hCam - 40), cv2.FONT_HERSHEY_PLAIN,
                            1.5, (200, 200, 200), 2)
                # 空闲时显式设置模式为 None (除非被手部丢失逻辑覆盖)
                if self.current_mode != None:
                    if self.current_mode != "Idle":
                        self.current_mode = None

            # --- FPS 计算和显示 (所有模式通用) ---
            cTime = time.time()
            fps = 1 / (cTime - self.pTime) if (cTime - self.pTime) > 0 else 0
            self.pTime = cTime
            cv2.putText(processed_frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 0), 2)
            # --- GUI 更新 ---
            # 将处理后的帧 (模式逻辑的最终输出) 转换为 Tkinter 格式
            cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(cv2image)
            imgtk = PIL.ImageTk.PhotoImage(image=img)
            # 更新 GUI 中的相机标签 (检查它是否存在)
            if hasattr(self, 'camera_label_left') and self.camera_label_left.winfo_exists():
                self.camera_label_left.imgtk = imgtk
                self.camera_label_left.configure(image=imgtk)
            else:
                # 如果标签不见了，窗口可能正在关闭，停止循环
                print("未找到相机标签，停止更新循环。")
                self.log_message("未找到相机标签，停止更新循环。")
                self.running = False
        except Exception as e:
            # 记录帧处理期间的任何意外错误
            print(f"update_frame 主循环出错: {e}")
            self.log_message(f"错误 in update_frame: {e}")
            import traceback
            traceback.print_exc()  # 打印堆栈跟踪以进行调试
        # --- 安排下一次更新 ---
        # 仅当正在运行且窗口仍然存在时才安排
        if self.running and self.root.winfo_exists():
            self.root.after(10, self.update_frame)
        elif not self.running:
            print("更新循环因 running 为 False 而完成。")
            self.log_message("更新循环因 running 为 False 而完成。")
            # 在受控停止时确保释放摄像头
            if self.cap and self.cap.isOpened():
                self.cap.release()
                print("在 running 标志设置为 False 后释放摄像头。")
                self.log_message("在 running 标志设置为 False 后释放摄像头。")

    # 窗口关闭处理
    def on_close(self):
        """GestureControlGUI (Toplevel) 窗口的关闭处理程序。"""
        self.log_message("关闭手势控制模块...")
        print("正在关闭手势控制窗口...")
        self.running = False  # 通知 update_frame 和线程停止
        # 停止所有后台线程
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=1)
        # 如果正在运行，则停止语音输入线程
        if hasattr(self, 'vim') and self.vim and self.vim.running:
            print("正在停止语音输入模块...")
            self.log_message("正在停止语音输入模块...")
            self.vim.stop_listening()
        # 释放摄像头资源 - 至关重要
        if self.cap and self.cap.isOpened():
            print("正在释放摄像头捕获...")
            self.cap.release()
            print("摄像头已释放 (on_close)")
        else:
            print("摄像头已被释放或未打开。")
        print("正在销毁手势控制GUI窗口。")
        if self.root and self.root.winfo_exists():
            self.root.destroy()
        print("手势控制模块窗口已关闭。")

# MainApp 类 (新的启动器 GUI)
class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("非接触式人机交互系统")
        # 设置与 GestureControlGUI 相同的初始大小
        self.root.geometry("1200x600")
        self.root.configure(bg="#f0f0f0")  # 浅灰色背景
        self.gesture_control_window = None  # 用于跟踪子窗口
        # 添加AI绘画窗口引用
        self.ai_painting_window = None
        title_label = tk.Label(self.root, text="非接触式人机交互系统",
                               font=("Arial", 24, "bold"), bg="#f0f0f0")
        title_label.pack(pady=40)
        # --- 按钮框架 ---
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=20, padx=50, fill=tk.BOTH, expand=True)
        # 配置按钮框架中的列以居中按钮
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        button_frame.grid_rowconfigure(0, weight=1)
        # 按钮样式 (可选，为了更好看)
        button_font = ("Arial", 14)
        button_width = 20
        button_height = 3
        # --- 电脑交互按钮 ---
        interaction_button = tk.Button(
            button_frame,
            text="电脑交互模块",
            font=button_font,
            width=button_width,
            height=button_height,
            command=self.launch_gesture_control,
            bg="#d0e0f0",  # 浅蓝色背景
            fg="black",  # 黑色文本
            relief=tk.RAISED,
            bd=2
        )
        # 将按钮放在左侧网格单元格中，如果单元格扩展，则在所有侧面粘性
        interaction_button.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        # --- AI 绘画按钮 ---
        painting_button = tk.Button(
            button_frame,
            text="AI绘画模块",
            font=button_font,
            width=button_width,
            height=button_height,
            command=self.launch_ai_painting,
            bg="#d0f0d0",
            fg="black",
            relief=tk.RAISED,
            bd=2
        )
        # 将按钮放在右侧网格单元格中
        painting_button.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        # 处理主窗口关闭
        self.root.protocol("WM_DELETE_WINDOW", self.on_main_close)

    # 在新的 Toplevel 窗口中启动 GestureControlGUI。
    def launch_gesture_control(self):
        # 防止打开多个手势窗口
        if self.gesture_control_window is not None and self.gesture_control_window.winfo_exists():
            print("电脑交互模块已打开。")
            messagebox.showinfo("提示", "电脑交互模块窗口已经打开。", parent=self.root)
            self.gesture_control_window.lift()  # 将现有窗口置于前面
            return
        print("启动电脑交互模块...")
        # 为手势控制 GUI 创建一个新的 Toplevel 窗口
        self.gesture_control_window = tk.Toplevel(self.root)
        self.gesture_control_window.geometry("1200x600")  # 为新窗口设置大小
        # 在新窗口内实例化 GestureControlGUI 类
        # GestureControlGUI 类本身处理其内部设置和主循环
        try:
            gesture_app = GestureControlGUI(self.gesture_control_window)
            # 检查初始化是否失败 (例如，摄像头错误)
            if not gesture_app.running:
                print("GestureControlGUI 初始化失败或立即关闭。")
                self.gesture_control_window = None
        except Exception as e:
            print(f"启动电脑交互模块时出错: {e}")
            messagebox.showerror("启动错误", f"无法启动电脑交互模块:\n{e}", parent=self.root)
            if self.gesture_control_window and self.gesture_control_window.winfo_exists():
                self.gesture_control_window.destroy()
            self.gesture_control_window = None

    # 启动 AI 绘画模块的占位符。
    def launch_ai_painting(self):
        if self.ai_painting_window is not None and self.ai_painting_window.winfo_exists():
            self.ai_painting_window.lift()
            return

        self.ai_painting_window = tk.Toplevel(self.root)
        self.ai_painting_app = AIPaintingGUI(self.ai_painting_window)

    # 处理主启动器窗口的关闭。
    def on_main_close(self):
        print("正在关闭主应用程序...")
        # 可选地，询问用户是否也想关闭子窗口，
        if self.gesture_control_window is not None and self.gesture_control_window.winfo_exists():
            print("请先关闭 '电脑交互模块' 窗口。")
            messagebox.showwarning("关闭提示", "请先手动关闭打开的 '电脑交互模块' 窗口。", parent=self.root)
            return  # 如果子窗口打开，则阻止主窗口关闭
        self.root.destroy()
        print("主应用程序已关闭。")


# 主执行块
if __name__ == "__main__":
    # 在启动任何 GUI 之前确保数据集目录存在
    if not os.path.exists('dataset'):
        try:
            os.makedirs('dataset')
            print("创建 'dataset' 目录。")
        except OSError as e:
            print(f"无法创建 'dataset' 目录: {e}")

    main_root = tk.Tk()

    app = MainApp(main_root)

    main_root.mainloop()
