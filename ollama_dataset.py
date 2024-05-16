import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit, QProgressBar, QAction
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QIcon
from tqdm import tqdm
import ollama

class Worker(QThread):
    update_progress = pyqtSignal(int)

    def __init__(self, df, system_prompt):
        super().__init__()
        self.df = df
        self.system_prompt = system_prompt
        self.running = True

    def run(self):
        if 'prompt' not in self.df.columns:
            print("Error: DataFrame does not contain the 'prompt' column.")
            return  # Exit the thread if the required column is missing

        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            if not self.running:
                break
            response = self.get_ollama_response(row['prompt'])
            self.df.at[index, 'output'] = response
            self.update_progress.emit(int((index + 1) / self.df.shape[0] * 100))
        
        self.df.to_csv('filled_qna_dataset.csv', index=False)
        print("Dataset processing complete. Updated dataset saved as 'filled_qna_dataset.csv'.")

    def get_ollama_response(self, prompt):
        response = ollama.chat(model='llama3', messages=[
            {
                'role': 'user',
                'content': f'{self.system_prompt} {prompt}',
            },
        ])
        return response['message']['content']

    def stop(self):
        self.running = False

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.dark_mode = False
        self.init_ui()
        self.init_menu()

    def init_ui(self):
        self.setGeometry(200, 200, 400, 200)
        self.setWindowTitle('Data Processor GUI')
        
        layout = QVBoxLayout()

        self.prompt_input = QLineEdit(self)
        self.prompt_input.setPlaceholderText('Enter your system prompt here...')
        layout.addWidget(self.prompt_input)

        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start_processing)
        layout.addWidget(self.start_button)

        self.pause_button = QPushButton('Pause')
        self.pause_button.clicked.connect(self.pause_processing)
        self.pause_button.setVisible(False)  # Hide the pause button initially
        layout.addWidget(self.pause_button)

        self.stop_button = QPushButton('Stop Now')
        self.stop_button.clicked.connect(self.stop_processing)
        layout.addWidget(self.stop_button)

        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def init_menu(self):
        toggle_theme_action = QAction(QIcon(), 'Toggle Theme', self)
        toggle_theme_action.triggered.connect(self.toggle_theme)
        self.toolbar = self.addToolBar('Toggle Theme')
        self.toolbar.addAction(toggle_theme_action)

    def toggle_theme(self):
        if self.dark_mode:
            self.setStyleSheet("")
            self.dark_mode = False
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget, QPushButton, QLineEdit, QProgressBar, QToolBar {
                    background-color: #333;
                    color: #eee;
                    border: 1px solid #555;
                }
                QLineEdit {
                    border: 2px solid #555;
                }
                QProgressBar::chunk {
                    background-color: #44a;
                }
            """)
            self.dark_mode = True

    def start_processing(self):
        system_prompt = self.prompt_input.text()
        self.df = pd.read_csv('unfilled_qna_dataset.csv')
        self.worker = Worker(self.df, system_prompt)
        self.worker.update_progress.connect(self.progress_bar.setValue)
        self.worker.start()
        self.pause_button.setVisible(True)

    def pause_processing(self):
        if self.worker:
            self.worker.running = not self.worker.running
            self.pause_button.setText('Resume' if not self.worker.running else 'Pause')

    def stop_processing(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.pause_button.setVisible(False)

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
