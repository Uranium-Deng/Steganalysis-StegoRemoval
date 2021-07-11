import pymysql
import sys
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from Main_Window import Ui_MainWindow
from register import Ui_Dialog
from login import Ui_Dialog_1
from stego_main_window import Ui_Dialog_2

from steganalysis_steganography import insert_info, judge_photo

import tensorflow as tf


class Steganography_and_Steganalysis_window(QWidget, Ui_Dialog_2):
    def __init__(self, parent=None):
        super(Steganography_and_Steganalysis_window, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("隐写和隐写分析界面")
        self.setWindowIcon(QIcon('./p1.jpg'))
        self.comboBox.addItems(['WOW', 'S-UNIWARD', 'HUGO'])
        self.comboBox_2.addItems(['0.4bpp', '0.7bpp', '1bpp'])
        self.comboBox_3.addItems(['WOW', 'S-UNIWARD', 'HUGO'])
        self.comboBox_4.addItems(['0.4bpp', '0.7bpp', '1bpp'])

        self.pushButton.clicked.connect(self.getfile_and_insert)
        self.pushButton_2.clicked.connect(self.getfile_and_analysis)

    def getfile_and_insert(self):
        # 向原始图像中插入含密信息
        method = self.comboBox.currentText()
        insert_degree = self.comboBox_2.currentText()

        QMessageBox.about(self, '水印嵌入界面', '您采用的隐写术为: {} 嵌入率为: {}'.format(method, str(insert_degree)))
        source_dir = '/home/dengruizhi/0.paper/4.deng/1.GUI/2.test_photos'
        target_file_type = 'Image files (*.jpg *.png *.bmp *.pgm)'
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open file', source_dir, target_file_type, None,
                                                   QFileDialog.DontUseNativeDialog)

        insert_flag = insert_info(file_path, method, insert_degree)

        if insert_flag:
            QMessageBox.about(self, '水印嵌入界面', '恭喜你，水印信息嵌入成功')
            print('start plt')
            cover_file = plt.imread(file_path)
            stego_path = '/home/dengruizhi/0.paper/4.deng/1.GUI/2.test_photos/stego/' + file_path.split('/')[-1]
            stego_file = plt.imread(stego_path)
            absres = np.abs(cover_file - stego_file)
            plt.subplot(1, 3, 1)
            plt.imshow(cover_file, cmap='gray')
            plt.title('cover image')
            plt.subplot(1, 3, 2)
            plt.imshow(stego_file, cmap='gray')
            plt.title('stego image')
            plt.subplot(1, 3, 3)
            plt.imshow(absres, cmap='gray')
            plt.title('residual image')
            plt.show()
            print('end plt')
        else:
            QMessageBox.about(self, '水印嵌入界面', '很遗憾，水印嵌入失败')

    def getfile_and_analysis(self):
        # 对选中的图片进行隐写分析 给出判断
        method = self.comboBox_3.currentText()
        insert_degree = self.comboBox_4.currentText()

        QMessageBox.about(self, '隐写分析界面', '针对的隐写术为: {} 嵌入率为: {}'.format(method, str(insert_degree)))
        source_dir = '/home/dengruizhi/0.paper/4.deng/1.GUI/2.test_photos'
        target_file_type = 'Image files (*.jpg *.png *.bmp *.pgm)'
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open file', source_dir, target_file_type, None,
                                                   QFileDialog.DontUseNativeDialog)

        # print('judge photo function')
        flag = judge_photo(file_path, method, insert_degree)
        print('flag: ', flag)
        if flag == 1:
            QMessageBox.about(self, '隐写分析界面', '啊哈，这是含密图像，逮住你了, 我厉害吧！')
        elif flag == 0:
            QMessageBox.about(self, '隐写分析界面', '放轻松放轻松，别紧张，这只是一张普通图片！')


class my_login_window(QDialog, Ui_Dialog_1):
    def __init__(self, parent=None):
        super(my_login_window, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("登录界面")
        self.setWindowIcon(QIcon('./p1.jpg'))
        self.radioButton.setCheckable(True)
        # self.radioButton.isChecked(False)
        self.radioButton.toggled.connect(self.to_hide_password)
        self.pushButton.clicked.connect(self.login_success)

    def to_hide_password(self):
        if self.radioButton.isChecked():
            self.lineEdit_3.setEchoMode(QLineEdit.Normal)
        else:
            self.lineEdit_3.setEchoMode(QLineEdit.Password)

    def link_to_database(self):
        self.db = pymysql.connect(host='localhost',
                                  port=3306,
                                  user='root',
                                  password='password',
                                  db='bishe',
                                  charset='utf8'
                                  )
        self.cursor = self.db.cursor()

    def login_success(self):
        key_number = self.lineEdit.text()
        password = self.lineEdit_3.text()

        self.link_to_database()

        try:
            select_sql = "select * from user"
            self.cursor.execute(select_sql)
            result = np.array(self.cursor.fetchall())
            li_result = list(result)
            key_numbers = result[:, 0]

            if key_number not in key_numbers:
                QMessageBox.about(self, '登录界面', '无该用户，请先注册账号')
            else:
                for item in li_result:
                    if item[0] != key_number:
                        continue
                    if item[2] == password:
                        # print('show_personal_window')
                        self.show_personal_window()
                    else:
                        QMessageBox.about(self, '登录界面', '您的密码不正确，请重新输入')
                    break
        except:
            print('login in failed')
        finally:
            self.db.close()
            self.cursor.close()

    def show_personal_window(self):
        self.steganography_and_steganalysis = Steganography_and_Steganalysis_window()
        self.steganography_and_steganalysis.show()
        self.close()


class my_register_window(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(my_register_window, self).__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(QIcon('./p1.jpg'))
        # self.lineEdit_3.setEchoMode(QLineEdit.passwordEchoOnEdit)
        self.pushButton.clicked.connect(self.register_success)
        self.setWindowTitle("注册界面")
        self.radioButton.setCheckable(True)
        self.radioButton.toggled.connect(self.hide_password)

    def hide_password(self):
        if self.radioButton.isChecked():
            self.lineEdit_3.setEchoMode(QLineEdit.Normal)
        else:
            self.lineEdit_3.setEchoMode(QLineEdit.Password)

    def link_to_database(self):
        self.db = pymysql.connect(host='localhost',
                                  port=3306,
                                  user='root',
                                  password='password',
                                  db='bishe',
                                  charset='utf8'
                                  )
        self.cursor = self.db.cursor()

    def register_success(self):
        key_number = self.lineEdit.text()
        username = self.lineEdit_5.text()
        password = self.lineEdit_3.text()

        self.link_to_database()

        try:
            select_sql = "select * from user"
            self.cursor.execute(select_sql)
            result = np.array(self.cursor.fetchall())
            key_numbers = result[:, 0]

            if key_number in key_numbers:
                QMessageBox.about(self, '注册界面', '该注册号已被注册，请更换注册号')
            else:
                insert_sql = "insert into user values (%s, %s, %s)"
                self.cursor.execute(insert_sql, (key_number, username, password))
                QMessageBox.about(self, '注册界面', '恭喜你，注册成功')
                self.db.commit()
        except:
            print('register failed')
        finally:
            self.db.close()
            self.cursor.close()


class my_main_window(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(my_main_window, self).__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(QIcon('./p1.jpg'))
        self.setWindowTitle("演示主界面")

        # self.label_3.setStyleSheet("QLabel{boarder-image:url(./p1.jpg);}")
        # self.label_3.setStyleSheet("QLabel{border-image: url(./xiaohui.jpeg);}")
        # 动态显示时间在label上

        timer = QTimer(self)
        timer.timeout.connect(self.show_time)
        timer.start()

        self.pushButton.clicked.connect(self.show_login)
        self.pushButton_2.clicked.connect(self.show_register)

    def show_time(self):
        datetime = QDateTime.currentDateTime()
        datetime = datetime.toString(Qt.ISODate)
        datetime = datetime.replace('T', ' ')
        self.label_2.setText(datetime)

    def show_login(self):
        self.begin_login = my_login_window()
        self.begin_login.show()
        self.close()

    def show_register(self):
        self.begin_register = my_register_window()
        self.begin_register.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_win = my_main_window()
    my_win.show()
    sys.exit(app.exec_())
