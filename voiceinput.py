from vosk import Model, KaldiRecognizer
import json
import pyperclip
import pyautogui
import pyaudio
import time
import sys
from threading import Thread

class VoiceInputModule:
    def __init__(self):
        try:
            self.model = Model('vosk-model-small-cn-0.22')
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            print("请确认：1.模型文件是否存在 2.路径是否正确 3.模型文件是否完整")
            sys.exit(1)
        self.timeout = 4  # 无声音超时时间
        self.running = False
        self.last_active = time.time()  # 添加最后活动时间记录
    #音频识别核心逻辑
    def _recognize_audio(self):
        # 初始化PyAudio
        p = pyaudio.PyAudio()
        try:
            # 打开音频流
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000
            )
        except Exception as e:
            print(f"无法打开麦克风: {str(e)}")
            self.running = False
            return
        # 创建VOSK识别器
        recognizer = KaldiRecognizer(self.model, 16000)
        print("\n🎤 开始监听...（说中文即可）")

        try:
            while self.running:
                # 读取音频数据
                data = stream.read(4000, exception_on_overflow=False)
                # 更新最后活动时间
                self.last_active = time.time()
                # 处理完整识别结果
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.FinalResult())
                    if result['text'] != '':
                        # 调用文本输入方法
                        self._input_text(result['text'])
                        return
                else:
                    partial = json.loads(recognizer.PartialResult())['partial']
                    if partial:
                        print(f"\r🎙 识别中: {partial}    ", end='')
                # 全局超时检测
                if time.time() - self.last_active > self.timeout:
                    print("\n⏰ 聆听超时，自动停止")
                    return
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self.running = False
            print("麦克风已释放")
    #文本输入方法
    def _input_text(self, text):
        text = text.replace(' ', '')
        try:
            # 复制文本到剪贴板
            pyperclip.copy(text)
            # 模拟Ctrl+V粘贴操作
            pyautogui.hotkey('ctrl', 'v')
            print(f"\n✅ 已输入: {text}")
        except Exception as e:
            print(f"\n❌ 输入失败: {str(e)}")
    #启动监听
    def start_listening(self):
        if not self.running:
            self.running = True
            self.last_active = time.time()  # 重置计时器
            Thread(target=self._recognize_audio, daemon=True).start()
        else:
            print("已经在监听状态")

    def stop_listening(self):
        """强制停止监听"""
        if self.running:
            self.running = False
            print("手动停止监听")

# 测试用例
if __name__ == "__main__":
    print("=== 语音输入测试 ===")
    print("请确保：")
    print("1. 麦克风已连接")
    print("2. 模型文件存在")
    print("3. 当前窗口可接受文本输入")

    vim = VoiceInputModule()
    vim.start_listening()

    # 主循环等待模块自动停止
    while vim.running:
        time.sleep(0.1)
    print("测试结束")
