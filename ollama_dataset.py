import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal
from tqdm import tqdm
import ollama

class Worker(QThread):
    update_progress = pyqtSignal(int)

    def __init__(self, df):
        super().__init__()
        self.df = df
        self.running = True

    def run(self):
        system_prompt = "You are the batman and you will always respond to everything in a dark and gloomy tone, always give a short and simple response"
        
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            if not self.running:
                break
            response = self.get_ollama_response(row['Prompt'], system_prompt)
            self.df.at[index, 'Response'] = response
            self.update_progress.emit(int((index + 1) / self.df.shape[0] * 100))
        
        self.df.to_csv('filled_qna_dataset.csv', index=False)
        print("Dataset processing complete. Updated dataset saved as 'filled_qna_dataset.csv'.")

    def get_ollama_response(self, prompt, system_prompt):
        response = ollama.chat(model='llama3', messages=[
            {
                'role': 'user',
                'content': f'{system_prompt}  {prompt}',
            },
        ])
        return response['message']['content']

    def stop(self):
        self.running = False

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        self.setGeometry(200, 200, 300, 150)
        self.setWindowTitle('Data Processor GUI')
        
        layout = QVBoxLayout()
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start_processing)
        layout.addWidget(self.start_button)

        self.pause_button = QPushButton('Pause')
        self.pause_button.clicked.connect(self.pause_processing)
        layout.addWidget(self.pause_button)

        self.stop_button = QPushButton('Stop Now')
        self.stop_button.clicked.connect(self.stop_processing)
        layout.addWidget(self.stop_button)

        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def start_processing(self):
        self.df = pd.read_csv('unfilled_qna_dataset.csv')
        self.worker = Worker(self.df)
        self.worker.update_progress.connect(self.progress_bar.setValue)
        self.worker.start()

    def pause_processing(self):
        if self.worker:
            self.worker.running = not self.worker.running
            self.pause_button.setText('Resume' if not self.worker.running else 'Pause')

    def stop_processing(self):
        if self.worker:
            self.worker.stop()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AppWindow()
    ex.show()
    sys.exit(app.exec_())
