import sys
from PyQt5.QtWidgets import QApplication
from chauncimoti.window import RobotWindow


def main():
    app = QApplication(sys.argv)
    window = RobotWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
