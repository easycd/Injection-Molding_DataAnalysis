"""프로그램 실행 파일. UI 클래스는 ui.py에서 불러옵니다."""

from ui import MoldingAnalysisApp


def main() -> None:
    app = MoldingAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()
