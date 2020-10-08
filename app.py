import sys
from PyQt5.QtWidgets import QApplication
from cobb_angle_dialog import Cobb_Angle_Dialog

if __name__ == "__main__":
    app=QApplication(sys.argv)
    mainform=Cobb_Angle_Dialog()
    mainform.show()
    sys.exit(app.exec_())