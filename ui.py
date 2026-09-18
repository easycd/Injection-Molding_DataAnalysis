"""앞유리 사이드 몰딩 분석 프로그램의 Tkinter UI 클래스.

실행: pip install -r requirements.txt 후 python main.py
화면 순서는 데이터 입력 -> 전처리 -> 시각화 -> 예측입니다.
"""

from __future__ import annotations

import os
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

# Matplotlib 글꼴 캐시는 쓰기 가능한 프로젝트 폴더에 저장합니다.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".matplotlib"))

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from data_analyzer import DataAnalyzer, INDEX_COLUMNS, TARGET_COLUMN

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic")

LABEL_TEXT = {0: "양품(0)", 1: "불량(1)"}
# CSV의 실제 값은 그대로 유지하고 화면에서만 이 자릿수로 반올림합니다.
DISPLAY_DECIMALS = 3

# 데이터의 영문 열 이름을 처음 보는 사람도 이해할 수 있도록 한국어 설명을 붙입니다.
FEATURE_NAMES = {
    "PassOrFail": "품질 판정",
    "Injection_Time": "사출 시간",
    "Filling_Time": "충전 시간",
    "Plasticizing_Time": "가소화 시간",
    "Cycle_Time": "전체 사이클 시간",
    "Clamp_Close_Time": "금형 닫힘 시간",
    "Cushion_Position": "쿠션 위치",
    "Switch_Over_Position": "보압 전환 위치",
    "Plasticizing_Position": "가소화 완료 위치",
    "Clamp_Open_Position": "금형 열림 위치",
    "Max_Injection_Speed": "최대 사출 속도",
    "Max_Screw_RPM": "최대 스크루 회전수",
    "Average_Screw_RPM": "평균 스크루 회전수",
    "Max_Injection_Pressure": "최대 사출 압력",
    "Max_Switch_Over_Pressure": "보압 전환 시 최대 압력",
    "Max_Back_Pressure": "최대 배압",
    "Average_Back_Pressure": "평균 배압",
    "Barrel_Temperature_1": "배럴 온도 1",
    "Barrel_Temperature_2": "배럴 온도 2",
    "Barrel_Temperature_3": "배럴 온도 3",
    "Barrel_Temperature_4": "배럴 온도 4",
    "Barrel_Temperature_5": "배럴 온도 5",
    "Barrel_Temperature_6": "배럴 온도 6",
    "Barrel_Temperature_7": "배럴 온도 7",
    "Hopper_Temperature": "호퍼 온도",
    "Mold_Temperature_3": "금형 온도 3",
    "Mold_Temperature_4": "금형 온도 4",
}


def friendly_name(column: str) -> str:
    """영문 열 이름을 '한국어 (영문)' 형태로 바꿉니다."""
    display_column = column.split("__", 1)[1] if "__" in column else column
    korean = FEATURE_NAMES.get(display_column)
    return f"{korean} ({display_column})" if korean else display_column


class MoldingAnalysisApp(tk.Tk):
    """프로그램의 화면과 분석 기능을 관리하는 메인 클래스입니다."""

    def __init__(self) -> None:
        super().__init__()
        self.title("앞유리 사이드 몰딩 데이터 분석 및 불량 예측")
        self.geometry("1500x900")
        self.minsize(1200, 760)
        self.configure(bg="#f1f5f9")

        # 분석 과정에서 계속 사용하는 데이터와 모델을 저장합니다.
        # UI는 DataAnalyzer 클래스에 분석을 요청하고 결과만 화면에 표시합니다.
        self.analyzer = DataAnalyzer()
        self.raw_df: pd.DataFrame | None = None
        self.analysis_df: pd.DataFrame | None = None
        self.feature_columns: list[str] = []
        self.eda_columns: list[str] = []
        self.model = None
        self.last_model_name = ""
        self.feature_entries: dict[str, ttk.Entry] = {}
        self.canvases: list[FigureCanvasTkAgg] = []

        # StringVar는 Tkinter 위젯에 표시되는 값을 관리합니다.
        self.file_path = tk.StringVar()
        self.target_var = tk.StringVar(value=TARGET_COLUMN)
        self.model_var = tk.StringVar(value="Random Forest")
        self.status_var = tk.StringVar(value="시작 방법: 왼쪽 위의 [열기] 버튼으로 CSV 파일을 선택하세요.")
        self.eda_feature_var = tk.StringVar()
        self.question_var = tk.StringVar(value="불량과 가장 관련이 큰 변수는 무엇인가요?")

        self._make_styles()
        self._build_header()
        self._build_layout()

    def _make_styles(self) -> None:
        """화면의 색상과 글꼴을 통일합니다."""
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f1f5f9")
        style.configure("Card.TFrame", background="#ffffff", relief="flat", borderwidth=0)
        style.configure("Sidebar.TFrame", background="#ffffff", relief="flat", borderwidth=0)
        style.configure("Header.TFrame", background="#0f172a", borderwidth=0)
        style.configure("TLabel", background="#f1f5f9", foreground="#334155", font=("Malgun Gothic", 10))
        style.configure("Card.TLabel", background="#ffffff", foreground="#334155", font=("Malgun Gothic", 10))
        style.configure("Sidebar.TLabel", background="#ffffff", foreground="#475569", font=("Malgun Gothic", 10))
        style.configure("Title.TLabel", background="#0f172a", foreground="#ffffff", font=("Malgun Gothic", 19, "bold"))
        style.configure("Subtitle.TLabel", background="#0f172a", foreground="#94a3b8", font=("Malgun Gothic", 9))
        style.configure("Section.TLabel", background="#f1f5f9", foreground="#1e3a8a", font=("Malgun Gothic", 12, "bold"))
        style.configure("SidebarSection.TLabel", background="#ffffff", foreground="#2563eb", font=("Malgun Gothic", 11, "bold"))
        style.configure("CardSection.TLabel", background="#ffffff", foreground="#1e3a8a", font=("Malgun Gothic", 12, "bold"))
        style.configure("CardValue.TLabel", background="#ffffff", foreground="#2563eb", font=("Malgun Gothic", 14, "bold"))
        style.configure("Blue.TButton", font=("Malgun Gothic", 10, "bold"), foreground="#ffffff", background="#2563eb", borderwidth=0, padding=(12, 9))
        style.map("Blue.TButton", background=[("pressed", "#1d4ed8"), ("active", "#3b82f6")])
        style.configure("Green.TButton", font=("Malgun Gothic", 10, "bold"), foreground="#ffffff", background="#059669", borderwidth=0, padding=(12, 9))
        style.map("Green.TButton", background=[("pressed", "#047857"), ("active", "#10b981")])
        style.configure("TButton", font=("Malgun Gothic", 9), padding=(10, 6), borderwidth=0)
        style.configure("TEntry", padding=7, fieldbackground="#ffffff", bordercolor="#cbd5e1", lightcolor="#cbd5e1", darkcolor="#cbd5e1")
        style.configure("TCombobox", padding=6, fieldbackground="#ffffff", bordercolor="#cbd5e1", arrowsize=14)
        style.configure("TRadiobutton", background="#ffffff", foreground="#334155", font=("Malgun Gothic", 10))
        style.configure("Treeview", background="#ffffff", fieldbackground="#ffffff", foreground="#334155", rowheight=30, borderwidth=0, font=("Malgun Gothic", 9))
        style.configure("Treeview.Heading", background="#e2e8f0", foreground="#1e293b", relief="flat", padding=(8, 8), font=("Malgun Gothic", 9, "bold"))
        style.map("Treeview", background=[("selected", "#dbeafe")], foreground=[("selected", "#1e3a8a")])
        style.map("Treeview.Heading", background=[("active", "#cbd5e1")])

        # 화살표와 입체 테두리를 없앤 얇은 파란색 스크롤바입니다.
        for orientation in ("Vertical", "Horizontal"):
            scrollbar_style = f"Modern.{orientation}.TScrollbar"
            style.configure(
                scrollbar_style,
                background="#60a5fa",
                troughcolor="#e2e8f0",
                bordercolor="#e2e8f0",
                lightcolor="#60a5fa",
                darkcolor="#60a5fa",
                relief="flat",
                borderwidth=0,
                arrowcolor="#2563eb",
                arrowsize=12,
                width=13,
            )
            style.map(
                scrollbar_style,
                background=[("pressed", "#2563eb"), ("active", "#3b82f6")],
            )

        # 운영체제 기본 레이아웃을 유지해야 Windows에서도 손잡이가 정상 표시됩니다.
        # 탭을 눌러도 크기는 바뀌지 않고 배경 명암으로만 선택 상태를 표시합니다.
        style.configure(
            "TNotebook.Tab",
            padding=(20, 10),
            font=("Malgun Gothic", 10, "bold"),
            background="#e2e8f0",
            foreground="#475569",
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#ffffff"), ("active", "#dbeafe")],
            foreground=[("selected", "#2563eb"), ("active", "#1d4ed8")],
            padding=[("selected", (20, 10)), ("active", (20, 10))],
        )
        # 기본 테마의 점선 포커스 요소를 제외하여 클릭 후 점선이 보이지 않게 합니다.
        style.layout(
            "TNotebook.Tab",
            [("Notebook.tab", {"sticky": "nswe", "children": [
                ("Notebook.padding", {"side": "top", "sticky": "nswe", "children": [
                    ("Notebook.label", {"side": "top", "sticky": ""})
                ]})
            ]})],
        )

    def _build_header(self) -> None:
        header = ttk.Frame(self, style="Header.TFrame", height=76)
        header.pack(fill="x")
        header.pack_propagate(False)
        title_box = ttk.Frame(header, style="Header.TFrame")
        title_box.pack(fill="both", expand=True, padx=26, pady=(12, 10))
        ttk.Label(title_box, text="앞유리 사이드 몰딩 품질 분석", style="Title.TLabel", anchor="w").pack(anchor="w")
        ttk.Label(title_box, text="Injection Molding Quality Analytics", style="Subtitle.TLabel", anchor="w").pack(anchor="w", pady=(2, 0))

    def _build_layout(self) -> None:
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=18, pady=16)
        self._build_control_panel(body)
        self._build_notebook(body)
        ttk.Label(self, textvariable=self.status_var, anchor="w", foreground="#64748b").pack(fill="x", padx=22, pady=(0, 10))

    def _build_control_panel(self, parent: ttk.Frame) -> None:
        """왼쪽에 기준 CSV와 예측에 사용할 저장 모델을 배치합니다."""
        panel = ttk.Frame(parent, width=330, style="Sidebar.TFrame", padding=(18, 16))
        panel.pack(side="left", fill="y", padx=(0, 16))
        panel.pack_propagate(False)

        ttk.Label(panel, text="1. 데이터 입력", style="SidebarSection.TLabel").pack(anchor="w", pady=(0, 10))
        ttk.Label(panel, text="CSV 파일 선택", style="Sidebar.TLabel").pack(anchor="w")
        file_row = ttk.Frame(panel, style="Sidebar.TFrame")
        file_row.pack(fill="x", pady=(5, 18))
        ttk.Entry(file_row, textvariable=self.file_path).pack(side="left", fill="x", expand=True)
        ttk.Button(file_row, text="열기", command=self.open_csv).pack(side="left", padx=(6, 0))

        ttk.Label(panel, text="2. 저장 모델 선택", style="SidebarSection.TLabel").pack(anchor="w", pady=(0, 8))
        self.model_combo = ttk.Combobox(
            panel,
            textvariable=self.model_var,
            values=[
                "Gradient Boosting",
                "Random Forest",
            ],
            state="readonly",
        )
        self.model_combo.pack(fill="x", pady=(0, 18))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)

        ttk.Button(panel, text="새 제품 예측하기", style="Green.TButton", command=self.predict_one).pack(fill="x", pady=(10, 0), ipady=9)

        info = ("※ 모델은 add_data.csv로 이미 학습된 PKL을 사용합니다.\n"
                "※ 프로그램에서는 모델을 다시 학습하거나 평가하지 않습니다.\n"
                "※ 입력값은 add_data.csv와 같은 정규화 값 기준입니다.")
        ttk.Label(panel, text=info, style="Sidebar.TLabel", wraplength=285, foreground="#64748b").pack(anchor="w", pady=(22, 0))

    def _build_notebook(self, parent: ttk.Frame) -> None:
        self.notebook = ttk.Notebook(parent, takefocus=False)
        self.notebook.pack(side="left", fill="both", expand=True)
        self.data_tab = ttk.Frame(self.notebook)
        self.preprocessing_tab = ttk.Frame(self.notebook)
        self.eda_tab = ttk.Frame(self.notebook)
        self.qa_tab = ttk.Frame(self.notebook)
        self.performance_tab = ttk.Frame(self.notebook)
        self.prediction_tab = ttk.Frame(self.notebook)
        for frame, title in [(self.data_tab, "데이터 입력"), (self.preprocessing_tab, "전처리"),
                             (self.eda_tab, "시각화(EDA)"), (self.qa_tab, "리터러시(Q→A)"),
                             (self.performance_tab, "모델 성능"),
                             (self.prediction_tab, "예측 결과")]:
            self.notebook.add(frame, text=title)
        self._build_data_tab()
        self._build_preprocessing_tab()
        self._build_eda_tab()
        self._build_qa_tab()
        self._build_performance_tab()
        self._build_prediction_tab()

    def _build_data_tab(self) -> None:
        top = ttk.Frame(self.data_tab)
        top.pack(fill="x", padx=16, pady=12)
        ttk.Label(top, text="이곳에서는 불러온 CSV의 행·열 개수와 실제 데이터를 확인합니다.").pack(anchor="w", pady=(0, 5))
        self.data_summary_label = ttk.Label(top, text="CSV를 불러오면 데이터 크기와 라벨 분포가 표시됩니다.", style="Section.TLabel")
        self.data_summary_label.pack(anchor="w")
        table_frame = ttk.Frame(self.data_tab, style="Card.TFrame")
        table_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.data_tree = ttk.Treeview(table_frame, show="headings")
        xbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.data_tree.xview, style="Modern.Horizontal.TScrollbar")
        ybar = ttk.Scrollbar(table_frame, orient="vertical", command=self.data_tree.yview, style="Modern.Vertical.TScrollbar")
        self.data_tree.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.data_tree.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

    def _build_preprocessing_tab(self) -> None:
        ttk.Label(self.preprocessing_tab, text="전처리란 분석 전에 빈 값과 필요 없는 열을 찾아 정리하는 과정입니다.").pack(anchor="w", padx=16, pady=(14, 0))
        self.preprocessing_text = tk.Text(self.preprocessing_tab, font=("Consolas", 11), bg="white", fg="#334155", relief="flat", borderwidth=0, padx=16, pady=14)
        self.preprocessing_text.pack(fill="both", expand=True, padx=16, pady=(8, 16))
        self._set_text(self.preprocessing_text, "CSV를 불러오면 전처리 점검 결과가 표시됩니다.")

    def _build_eda_tab(self) -> None:
        toolbar = ttk.Frame(self.eda_tab)
        toolbar.pack(fill="x", padx=16, pady=(12, 4))
        ttk.Label(toolbar, text="그래프로 양품과 불량의 차이 및 변수 간 관계를 살펴봅니다.   ").pack(side="left")
        ttk.Label(toolbar, text="상세 분석 변수:").pack(side="left")
        self.eda_combo = ttk.Combobox(toolbar, textvariable=self.eda_feature_var, state="readonly", width=30)
        self.eda_combo.pack(side="left", padx=8)
        ttk.Button(toolbar, text="그래프 새로고침", command=self.draw_eda).pack(side="left")
        self.eda_chart_frame = ttk.Frame(self.eda_tab, style="Card.TFrame")
        self.eda_chart_frame.pack(fill="both", expand=True, padx=16, pady=(4, 16))

    def _build_qa_tab(self) -> None:
        card = ttk.Frame(self.qa_tab, style="Card.TFrame")
        card.pack(fill="both", expand=True, padx=16, pady=16)
        ttk.Label(card, text="데이터 리터러시: 질문 → 해답", style="CardSection.TLabel").pack(anchor="w", padx=18, pady=(18, 10))
        self.question_combo = ttk.Combobox(card, textvariable=self.question_var, state="readonly", values=[
            "불량과 가장 관련이 큰 변수는 무엇인가요?", "데이터에서 주의할 품질 문제는 무엇인가요?",
            "정확도가 높으면 좋은 모델인가요?"])
        self.question_combo.pack(fill="x", padx=18, pady=6)
        ttk.Button(card, text="해답 보기", command=self.answer_question).pack(anchor="e", padx=18, pady=6)
        self.answer_label = ttk.Label(card, text="질문을 선택하고 '해답 보기'를 누르세요.", style="Card.TLabel", wraplength=900, justify="left")
        self.answer_label.pack(anchor="w", padx=18, pady=18)

    def _build_prediction_tab(self) -> None:
        outer = ttk.Frame(self.prediction_tab)
        outer.pack(fill="both", expand=True, padx=16, pady=16)
        ttk.Label(outer, text="공정값 입력", style="Section.TLabel").pack(anchor="w")
        ttk.Label(outer, text="CSV를 불러오면 각 변수의 중앙값이 자동 입력됩니다. 값을 바꾸고 예측 실행을 누르세요.").pack(anchor="w", pady=(2, 8))
        canvas = tk.Canvas(outer, bg="white", highlightthickness=1, highlightbackground="#d5deea")
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview, style="Modern.Vertical.TScrollbar")
        self.input_frame = ttk.Frame(canvas, style="Card.TFrame")
        self.input_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        self.input_window = canvas.create_window((0, 0), window=self.input_frame, anchor="nw")
        # 창 크기가 바뀌면 입력 프레임도 캔버스 폭에 맞춰져 오른쪽 값이 잘리지 않습니다.
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(self.input_window, width=event.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="left", fill="y")
        result_card = ttk.Frame(outer, style="Card.TFrame", width=270)
        result_card.pack(side="left", fill="y", padx=(12, 0))
        result_card.pack_propagate(False)
        ttk.Label(result_card, text="예측 결과", style="CardSection.TLabel").pack(pady=(30, 12))
        self.prediction_result_label = ttk.Label(result_card, text="예측 전", style="Card.TLabel", font=("Malgun Gothic", 18, "bold"), wraplength=230)
        self.prediction_result_label.pack(padx=16, pady=12)
        self.prediction_probability_label = ttk.Label(result_card, text="", style="Card.TLabel", wraplength=230)
        self.prediction_probability_label.pack(padx=16, pady=6)

    def _build_performance_tab(self) -> None:
        """두 저장 모델의 참고 성능을 한 표에서 비교합니다."""
        outer = ttk.Frame(self.performance_tab)
        outer.pack(fill="both", expand=True, padx=16, pady=16)
        ttk.Label(outer, text="저장 모델 성능 비교", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="재학습하지 않고, 현재 불러온 CSV 전체를 예측한 참고 성능입니다.",
        ).pack(anchor="w", pady=(3, 12))

        columns = ("model", "accuracy", "precision", "recall", "f1", "auc")
        self.performance_tree = ttk.Treeview(
            outer, columns=columns, show="headings", height=3, style="Treeview"
        )
        headings = {
            "model": "적용 모델",
            "accuracy": "정확도",
            "precision": "정밀도",
            "recall": "재현율",
            "f1": "F1 점수",
            "auc": "ROC-AUC",
        }
        widths = {"model": 210, "accuracy": 120, "precision": 120,
                  "recall": 120, "f1": 120, "auc": 120}
        for column in columns:
            self.performance_tree.heading(column, text=headings[column])
            self.performance_tree.column(
                column, width=widths[column], minwidth=90,
                anchor="center", stretch=True,
            )
        self.performance_tree.tag_configure("selected_model", background="#dbeafe", foreground="#1d4ed8")
        self.performance_tree.pack(fill="x")

        self.performance_note = ttk.Label(
            outer,
            text="CSV를 불러오면 두 모델의 성능이 표시됩니다.",
            style="Card.TLabel",
            wraplength=900,
        )
        self.performance_note.pack(anchor="w", fill="x", pady=(14, 0), ipadx=12, ipady=12)

    # 데이터 입력 및 전처리 -------------------------------------------------
    def open_csv(self) -> None:
        """파일 선택 창에서 CSV를 고르고 pandas DataFrame으로 읽습니다."""
        path = filedialog.askopenfilename(title="사출성형 CSV 선택", filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")])
        if not path:
            return
        try:
            df = self.analyzer.load_csv(path)
        except Exception as error:
            messagebox.showerror("파일 오류", f"CSV를 읽을 수 없습니다.\n{error}")
            return
        self.file_path.set(path)
        self.raw_df = df.copy()
        self.analysis_df = self.analyzer.analysis_df
        self.feature_columns = self.analyzer.feature_columns
        self.eda_columns = self.analyzer.numeric_columns
        self.eda_combo["values"] = self.eda_columns
        if self.eda_columns:
            self.eda_feature_var.set(self.eda_columns[0])
        self._show_data_preview()
        self._show_preprocessing_report()
        self._make_prediction_inputs()
        self._load_selected_model()
        self._show_model_performance()
        self.draw_eda()
        self.status_var.set(f"불러오기 완료: {os.path.basename(path)}")
        self.notebook.select(self.data_tab)

    def _prepare_data(self) -> None:
        """호환용 메서드: 실제 전처리는 DataAnalyzer가 담당합니다."""
        self.analyzer._prepare_data()
        self.analysis_df = self.analyzer.analysis_df
        self.feature_columns = self.analyzer.feature_columns
        self.eda_columns = self.analyzer.numeric_columns

    def _show_data_preview(self) -> None:
        """Treeview 표에 데이터 앞부분 100행을 보여줍니다."""
        assert self.raw_df is not None
        self.data_tree.delete(*self.data_tree.get_children())
        columns = self.raw_df.columns.tolist()
        self.data_tree["columns"] = columns
        for column in columns:
            self.data_tree.heading(column, text=friendly_name(column))
            # stretch=False로 설정해야 열이 창 너비에 맞춰 강제로 줄어들지 않습니다.
            self.data_tree.column(column, width=100, minwidth=55, stretch=False, anchor="center")
        self.data_tree.tag_configure("even", background="#ffffff")
        self.data_tree.tag_configure("odd", background="#f8fafc")
        for row_number, row in enumerate(self.raw_df.head(100).itertuples(index=False, name=None)):
            values = [f"{v:.{DISPLAY_DECIMALS}f}" if isinstance(v, float) else v for v in row]
            self.data_tree.insert("", "end", values=values, tags=("even" if row_number % 2 == 0 else "odd",))
        self._auto_size_table_columns(columns)
        counts = self.raw_df[TARGET_COLUMN].value_counts().to_dict()
        self.data_summary_label.configure(text=f"데이터: {len(self.raw_df):,}행 × {len(columns)}열   |   양품(0): {counts.get(0, 0):,}건   |   불량(1): {counts.get(1, 0):,}건")

    def _auto_size_table_columns(self, columns: list[str]) -> None:
        """열 제목과 숫자 길이를 측정해 Treeview 열 너비를 자동 조절합니다."""
        heading_font = tkfont.nametofont("TkHeadingFont")
        cell_font = tkfont.nametofont("TkDefaultFont")
        preview_rows = self.raw_df.head(100) if self.raw_df is not None else pd.DataFrame()

        for column in columns:
            # 제목과 해당 열의 표시값 중 가장 긴 픽셀 너비를 찾습니다.
            heading_width = heading_font.measure(friendly_name(column)) + 24
            value_width = 0
            if column in preview_rows:
                displayed_values = [
                    f"{value:.{DISPLAY_DECIMALS}f}" if isinstance(value, float) else str(value)
                    for value in preview_rows[column]
                ]
                value_width = max((cell_font.measure(value) for value in displayed_values), default=0) + 24

            # 지나치게 넓은 영문 제목은 280px까지만 허용합니다.
            width = max(65, min(max(heading_width, value_width), 280))
            self.data_tree.column(column, width=width, minwidth=55, stretch=False)

    def _show_preprocessing_report(self) -> None:
        """결측치와 상수 열처럼 모델링 전에 확인할 내용을 요약합니다."""
        assert self.raw_df is not None and self.analysis_df is not None
        summary = self.analyzer.preprocessing_summary()
        excluded = summary["excluded"]
        constants = summary["constants"]
        counts = summary["counts"]
        total = len(self.raw_df)
        good_count = int(counts.get(0, 0))
        fail_count = int(counts.get(1, 0))
        balance_gap = abs(good_count - fail_count) / total if total else 0.0
        if balance_gap <= 0.1:
            balance_message = "- 양품과 불량 비율이 균형에 가깝습니다."
        else:
            balance_message = "- 양품과 불량 비율 차이를 보완하도록 학습 가중치를 적용합니다."
        report = ["[전처리 점검 결과]",
                  f"원본 데이터 크기          : {self.raw_df.shape[0]:,}행 × {self.raw_df.shape[1]}열",
                  f"전체 결측값               : {summary['missing']:,}개",
                  f"제외한 인덱스 열          : {', '.join(excluded) or '없음'}",
                  f"제외한 상수 열            : {', '.join(constants) or '없음'}",
                  f"최종 독립변수             : {len(self.feature_columns)}개", "", "[타깃 분포]",
                  f"양품(0)                    : {good_count:,}건 ({good_count / total:.1%})",
                  f"불량(1)                    : {fail_count:,}건 ({fail_count / total:.1%})", "", "[해석]",
                  balance_message,
                  "- Accuracy, Precision, Recall, F1-score, ROC-AUC를 함께 확인합니다.",
                  "- 숫자형 공정 변수의 표준화는 모델 파이프라인 내부에서 수행합니다."]
        self._set_text(self.preprocessing_text, "\n".join(report))

    # EDA와 질문/답변 ------------------------------------------------------
    def draw_eda(self) -> None:
        """타깃 분포, 상관관계, 선택 변수 분포, 산점도를 그립니다."""
        if self.analysis_df is None or not self.eda_columns:
            return
        self._clear_frame(self.eda_chart_frame)
        selected = self.eda_feature_var.get() or self.eda_columns[0]
        corr = self.analysis_df.corr(numeric_only=True)[TARGET_COLUMN].drop(TARGET_COLUMN).abs().sort_values(ascending=False)
        second = corr.index[1] if len(corr) > 1 else corr.index[0]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        counts = self.analysis_df[TARGET_COLUMN].value_counts().sort_index()
        axes[0, 0].bar([LABEL_TEXT.get(i, str(i)) for i in counts.index], counts.values, color=["#4e9be8", "#e76f51"])
        axes[0, 0].set_title("타깃 분포")
        axes[0, 0].set_ylabel("건수")
        top = corr.head(10).index.tolist()
        sns.heatmap(self.analysis_df[top + [TARGET_COLUMN]].corr(), cmap="coolwarm", center=0, ax=axes[0, 1])
        axes[0, 1].set_title("상관관계 히트맵")
        sns.boxplot(data=self.analysis_df, x=TARGET_COLUMN, y=selected, ax=axes[1, 0], hue=TARGET_COLUMN, legend=False, palette=["#75b6e7", "#f39a7d"])
        axes[1, 0].set_title(f"{friendly_name(selected)}: 양품/불량 비교")
        sns.scatterplot(data=self.analysis_df, x=selected, y=second, hue=TARGET_COLUMN, palette={0: "#3182ce", 1: "#e53e3e"}, ax=axes[1, 1])
        axes[1, 1].set_title(f"{FEATURE_NAMES.get(selected, selected)} vs {FEATURE_NAMES.get(second, second)}")
        self._embed_figure(self.eda_chart_frame, fig)

    def answer_question(self) -> None:
        if self.analysis_df is None:
            messagebox.showwarning("데이터 없음", "먼저 CSV 파일을 불러오세요.")
            return
        question = self.question_var.get()
        if question.startswith("불량과 가장"):
            corr = self.analysis_df.corr(numeric_only=True)[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(key=abs, ascending=False)
            answer = f"품질 판정과 선형 상관관계가 가장 큰 변수는 {friendly_name(corr.index[0])}입니다 (상관계수 {corr.iloc[0]:.3f}). 상관관계만으로 실제 불량 원인이라고 단정할 수는 없습니다."
        elif question.startswith("데이터에서"):
            counts = self.analysis_df[TARGET_COLUMN].value_counts()
            answer = (f"양품 {int(counts.get(0, 0)):,}건, 불량 {int(counts.get(1, 0)):,}건으로 클래스 분포는 균형에 가깝습니다.")
        else:
            counts = self.analysis_df[TARGET_COLUMN].value_counts(normalize=True)
            majority_baseline = float(counts.max())
            answer = (f"아닙니다. 한 클래스만 예측하는 기준 정확도는 {majority_baseline:.1%}입니다. "
                      "현재 데이터는 양품과 불량이 균형에 가까우며, Accuracy와 함께 Recall, Precision, F1-score를 확인해야 합니다.")
        self.answer_label.configure(text=answer)

    # 한 건 예측 -----------------------------------------------------------
    def _show_model_performance(self) -> None:
        """Random Forest와 Gradient Boosting의 현재 CSV 성능을 표시합니다."""
        self.performance_tree.delete(*self.performance_tree.get_children())
        for model_name in ("Random Forest", "Gradient Boosting"):
            metrics = self.analyzer.evaluate_saved_model(model_name)
            values = (
                model_name,
                f"{metrics['Accuracy']:.3f}",
                f"{metrics['Precision']:.3f}",
                f"{metrics['Recall']:.3f}",
                f"{metrics['F1-Score']:.3f}",
                "-" if np.isnan(metrics["ROC-AUC"]) else f"{metrics['ROC-AUC']:.3f}",
            )
            tags = ("selected_model",) if model_name == self.model_var.get() else ()
            self.performance_tree.insert("", "end", iid=model_name, values=values, tags=tags)
        self.performance_note.configure(
            text=(f"현재 예측 적용 모델: {self.model_var.get()}\n"
                  "주의: add_data.csv는 모델 학습에 사용된 데이터이므로 실제 현장 성능보다 높게 표시될 수 있습니다.")
        )

    def _load_selected_model(self) -> None:
        """콤보박스에서 선택한 사전 학습 모델을 즉시 불러옵니다."""
        self.analyzer.load_model(self.model_var.get())
        self.model = self.analyzer.model
        self.last_model_name = self.model_var.get()

    def _on_model_selected(self, _event=None) -> None:
        """사용자가 모델을 바꾸면 해당 PKL로 교체합니다."""
        try:
            self._load_selected_model()
            self.prediction_result_label.configure(text="예측 전", foreground="#334155")
            self.prediction_probability_label.configure(text="")
            if self.analysis_df is not None:
                self._show_model_performance()
            self.status_var.set(f"{self.model_var.get()} 저장 모델 준비 완료")
        except Exception as error:
            messagebox.showerror("모델 오류", str(error))

    def _make_prediction_inputs(self) -> None:
        """각 독립변수의 중앙값을 기본값으로 넣은 입력칸을 만듭니다."""
        assert self.analysis_df is not None
        self._clear_frame(self.input_frame)
        self.feature_entries.clear()
        for i, column in enumerate(self.feature_columns):
            row, block = divmod(i, 2)
            base_col = block * 2
            ttk.Label(
                self.input_frame,
                text=friendly_name(column),
                style="Card.TLabel",
                wraplength=245,
            ).grid(row=row, column=base_col, sticky="w", padx=(16, 8), pady=6)
            if column in self.analyzer.categorical_columns:
                options = sorted(self.analysis_df[column].dropna().astype(str).unique().tolist())
                entry = ttk.Combobox(self.input_frame, values=options, state="readonly", width=12)
                if options:
                    entry.set(options[0])
            else:
                entry = ttk.Entry(self.input_frame, width=12)
                entry.insert(0, f"{self.analysis_df[column].median():.{DISPLAY_DECIMALS}f}")
            entry.grid(row=row, column=base_col + 1, sticky="ew", padx=(0, 22), pady=6)
            self.feature_entries[column] = entry
        # 이름 열은 내용만큼, 숫자 입력 열은 남은 공간을 균등하게 사용합니다.
        self.input_frame.columnconfigure(0, weight=0)
        self.input_frame.columnconfigure(1, weight=1, minsize=105)
        self.input_frame.columnconfigure(2, weight=0)
        self.input_frame.columnconfigure(3, weight=1, minsize=105)

    def predict_one(self) -> None:
        """입력한 한 행을 저장 모델에 넣어 불량 확률을 구합니다."""
        if self.analysis_df is None:
            messagebox.showwarning("데이터 없음", "먼저 add_data.csv를 불러오세요.")
            return
        try:
            self._load_selected_model()
            values = {
                name: (float(entry.get()) if name in self.analyzer.numeric_columns else entry.get())
                for name, entry in self.feature_entries.items()
            }
            result, fail_probability = self.analyzer.predict(values)
            text, color = ("불량(1) 예측", "#c53030") if result == 1 else ("양품(0) 예측", "#168a35")
            self.prediction_result_label.configure(text=text, foreground=color)
            self.prediction_probability_label.configure(text=f"불량일 확률: {fail_probability:.1%}\n\n이 결과는 참고용이며 실제 품질검사를 대체하지 않습니다.")
            self.notebook.select(self.prediction_tab)
        except ValueError as error:
            messagebox.showerror("입력 오류", str(error))
        except Exception as error:
            messagebox.showerror("예측 오류", str(error))

    # 반복되는 보조 기능 ----------------------------------------------------
    @staticmethod
    def _set_text(widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    @staticmethod
    def _clear_frame(frame: ttk.Frame) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _embed_figure(self, frame: ttk.Frame, figure: plt.Figure) -> None:
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvases.append(canvas)
        plt.close(figure)


def main() -> None:
    """이 파일을 직접 실행했을 때 Tkinter 프로그램을 시작합니다."""
    app = MoldingAnalysisApp()
    app.mainloop()
