import os
import sys
import pandas as pd
import json
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QLineEdit, QProgressBar, QAction, QSlider, QLabel, QMessageBox, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QMutex, QWaitCondition
from PyQt5.QtGui import QIcon
from tqdm import tqdm
import ollama

class Worker(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, df, system_prompt, num_rows, model_name):
        super().__init__()
        self.df = df
        self.system_prompt = system_prompt
        self.num_rows = num_rows
        self.model_name = model_name
        self.running = True
        self.paused = False
        self.mutex = QMutex()
        self.condition = QWaitCondition()

    def run(self):
        if 'instruction' not in self.df.columns or 'input' not in self.df.columns or 'output' not in self.df.columns:
            print("Error: DataFrame does not contain the required columns.")
            return

        for index, row in tqdm(self.df.iterrows(), total=self.num_rows):
            self.mutex.lock()
            while self.paused:
                self.condition.wait(self.mutex)
            self.mutex.unlock()

            if index >= self.num_rows:
                break
            if not self.running:
                break
            response = self.get_ollama_response(row['input'])
            self.df.at[index, 'output'] = response
            self.update_progress.emit(int((index + 1) / self.num_rows * 100))

        # Save the updated DataFrame to a JSON file
        self.df.iloc[:index+1].to_json('filled_qna_dataset.json', orient='records', lines=True)
        print("Dataset processing complete. Updated dataset saved as 'filled_qna_dataset.json'.")
        self.finished.emit()

    def get_ollama_response(self, prompt):
        response = ollama.chat(model=self.model_name, messages=[
            {
                'role': 'user',
                'content': f'{self.system_prompt} {prompt}',
            },
        ])
        return response['message']['content']

    def stop(self):
        self.running = False
        self.paused = False
        self.condition.wakeAll()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        self.condition.wakeAll()

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = self.load_data()
        self.dark_mode = False
        self.init_ui()
        self.init_menu()
        self.set_stylesheet()

    def load_data(self):
        local_file_path = 'alpaca_data_cleaned.json'
        if not os.path.exists(local_file_path):
            url = 'https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json'
            response = requests.get(url)
            with open(local_file_path, 'wb') as file:
                file.write(response.content)
        with open(local_file_path, 'r') as file:
            data = json.load(file)
        return pd.DataFrame(data)

    def init_ui(self):
        self.setGeometry(200, 200, 600, 400)
        self.setWindowTitle('Data Processor GUI')

        layout = QVBoxLayout()

        models = sorted([
            "llama3", "uncensored_llama3", "phi3", "wizardlm2", "mistral", "gemma", "mixtral", "llama2",
            "codegemma", "command-r", "command-r-plus", "llava", "dbrx", "codellama",
            "qwen", "dolphin-mixtral", "llama2-uncensored", "deepseek-coder",
            "mistral-openorca", "nomic-embed-text", "dolphin-mistral", "phi",
            "orca-mini", "nous-hermes2", "zephyr", "llama2-chinese",
            "wizard-vicuna-uncensored", "starcoder2", "vicuna", "tinyllama",
            "openhermes", "openchat", "starcoder", "dolphin-llama3", "yi",
            "tinydolphin", "wizardcoder", "stable-code", "mxbai-embed-large",
            "neural-chat", "phind-codellama", "wizard-math", "starling-lm",
            "falcon", "dolphincoder", "orca2", "nous-hermes", "stablelm2",
            "sqlcoder", "dolphin-phi", "solar", "deepseek-llm", "yarn-llama2",
            "codeqwen", "bakllava", "samantha-mistral", "all-minilm",
            "medllama2", "llama3-gradient", "wizardlm-uncensored", "nous-hermes2-mixtral",
            "xwinlm", "stable-beluga", "codeup", "wizardlm", "yarn-mistral",
            "everythinglm", "meditron", "llama-pro", "magicoder", "stablelm-zephyr",
            "nexusraven", "codebooga", "mistrallite", "wizard-vicuna", "llama3-chatqa",
            "snowflake-arctic-embed", "goliath", "open-orca-platypus2", "llava-llama3",
            "moondream", "notux", "megadolphin", "duckdb-nsql", "notus", "alfred",
            "llava-phi3", "falcon2"
        ])

        self.model_select = QComboBox(self)
        self.model_select.addItems(models)

        # Set the default value to "llama3"
        default_model = "llama3"
        default_index = models.index(default_model)
        self.model_select.setCurrentIndex(default_index)

        layout.addWidget(self.model_select)

        self.prompt_input = QLineEdit(self)
        self.prompt_input.setPlaceholderText('Enter your system prompt here...')
        layout.addWidget(self.prompt_input)

        self.slider_label = QLabel(f"Number of rows to fill: 0 / {len(self.df)}", self)
        layout.addWidget(self.slider_label)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.df))
        self.slider.valueChanged.connect(self.update_slider_label)
        layout.addWidget(self.slider)

        self.generate_button = QPushButton('Generate Dataset')
        self.generate_button.clicked.connect(self.start_processing)
        layout.addWidget(self.generate_button)

        self.pause_button = QPushButton('Pause')
        self.pause_button.clicked.connect(self.pause_processing)
        self.pause_button.setVisible(False)
        layout.addWidget(self.pause_button)

        self.stop_button = QPushButton('Stop Now')
        self.stop_button.clicked.connect(self.stop_processing)
        layout.addWidget(self.stop_button)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def init_menu(self):
        toggle_theme_action = QAction(QIcon(), 'Toggle Theme', self)
        toggle_theme_action.triggered.connect(self.toggle_theme)
        self.toolbar = self.addToolBar('Toggle Theme')
        self.toolbar.addAction(toggle_theme_action)

    def set_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f5f5f5;
                color: #333;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }
            QPushButton {
                background-color: #0078d7;
                color: #fff;
                border: none;
                padding: 10px;
                margin: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005fa3;
            }
            QLineEdit, QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
            }
            QLabel {
                margin: 5px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #f5f5f5;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #0078d7;
                border: 1px solid #777;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #fff;
                border: 1px solid #0078d7;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)

    def toggle_theme(self):
        if self.dark_mode:
            self.set_stylesheet()
            self.dark_mode = False
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #333;
                    color: #eee;
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                }
                QPushButton {
                    background-color: #444;
                    color: #fff;
                    border: none;
                    padding: 10px;
                    margin: 5px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #555;
                }
                QLineEdit, QProgressBar {
                    border: 1px solid #555;
                    border-radius: 5px;
                    padding: 5px;
                }
                QLabel {
                    margin: 5px;
                }
                QSlider::groove:horizontal {
                    border: 1px solid #bbb;
                    background: #333;
                    height: 10px;
                    border-radius: 4px;
                }
                QSlider::sub-page:horizontal {
                    background: #444;
                    border: 1px solid #777;
                    height: 10px;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #fff;
                    border: 1px solid #444;
                    width: 18px;
                    margin: -2px 0;
                    border-radius: 9px;
                }
            """)
            self.dark_mode = True

    def start_processing(self):
        system_prompt = self.prompt_input.text()
        num_rows = self.slider.value()
        model_name = self.model_select.currentText()
        if not system_prompt or num_rows == 0:
            self.show_alert("Please provide a system prompt and select the number of rows to fill.")
            return
        self.worker = Worker(self.df, system_prompt, num_rows, model_name)
        self.worker.update_progress.connect(self.update_progress_bar)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.start()
        self.generate_button.setVisible(False)
        self.pause_button.setVisible(True)
        self.slider.setEnabled(False)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}%")

    def pause_processing(self):
        if self.worker:
            if self.worker.paused:
                self.worker.resume()
                self.pause_button.setText('Pause')
            else:
                self.worker.pause()
                self.pause_button.setText('Resume')

    def stop_processing(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.on_generation_finished()

    def on_generation_finished(self):
        self.pause_button.setVisible(False)
        self.generate_button.setVisible(True)
        self.slider.setEnabled(True)
        self.show_alert("Dataset generation complete. The updated dataset has been saved as 'filled_qna_dataset.json'.")

    def update_slider_label(self, value):
        self.slider_label.setText(f"Number of rows to fill: {value} / {len(self.df)}")

    def show_alert(self, message):
        alert = QMessageBox()
        alert.setText(message)
        alert.exec_()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AppWindow()
    ex.show()
    sys.exit(app.exec_())
