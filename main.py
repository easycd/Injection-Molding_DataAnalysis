"""
=====================================================================
main.py
---------------------------------------------------------------------
프로그램 실행 진입점.
tkinter 루트 창을 만들고 ui.py의 ManufacturingAnalyzerApp을 붙여서 실행한다.

실행 방법:
    python main.py

필요 패키지:
    pip install pandas numpy matplotlib scikit-learn
=====================================================================
"""

import tkinter as tk
from ui import ManufacturingAnalyzerApp

if __name__ == "__main__":
    root = tk.Tk()
    app = ManufacturingAnalyzerApp(root)
    root.mainloop()
