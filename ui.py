"""
=====================================================================
ui.py
---------------------------------------------------------------------
화면(tkinter 위젯)만 담당하는 파일.
실제 데이터 처리/모델링 로직은 전혀 갖고 있지 않고,
data_analyzer.py의 DataAnalyzer 객체에게 모두 위임(호출)한다.

역할 분담:
    data_analyzer.py -> "무엇을 계산할지" (데이터/모델 로직)
    ui.py            -> "화면에 어떻게 보여줄지" (버튼, 표, 그래프 배치)
=====================================================================
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data_analyzer import DataAnalyzer, REGRESSION_MODELS, CLASSIFICATION_MODELS


class ManufacturingAnalyzerApp:
    """
    tkinter 창 하나(root) 안에 탭(Notebook)을 만들어
    파일선택 / 전처리 / 시각화 / 모델링 / 평가 / 예측 화면을 각각 배치하는 클래스.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("제조데이터 분석 프로그램")
        self.root.geometry("1200x800")

        # 데이터/모델 관련 로직은 전부 DataAnalyzer 인스턴스가 가지고 있는다.
        self.analyzer = DataAnalyzer()

        # 화면 표시용으로만 쓰는 tkinter 변수 (문제 유형 라디오버튼과 연결)
        self.problem_type = tk.StringVar(value="회귀")

        # 탭(Notebook) 위젯 생성
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.tab_file = ttk.Frame(self.notebook)
        self.tab_preprocess = ttk.Frame(self.notebook)
        self.tab_visualize = ttk.Frame(self.notebook)
        self.tab_model = ttk.Frame(self.notebook)
        self.tab_evaluate = ttk.Frame(self.notebook)
        self.tab_predict = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_file, text="1. 파일 선택")
        self.notebook.add(self.tab_preprocess, text="2. 전처리")
        self.notebook.add(self.tab_visualize, text="3. 시각화")
        self.notebook.add(self.tab_model, text="4. 모델링")
        self.notebook.add(self.tab_evaluate, text="5. 평가")
        self.notebook.add(self.tab_predict, text="6. 예측")

        # 각 탭 화면 구성
        self._build_file_tab()
        self._build_preprocess_tab()
        self._build_visualize_tab()
        self._build_model_tab()
        self._build_evaluate_tab()
        self._build_predict_tab()

    # =================================================================
    # 공통 유틸 함수 (표에 데이터 뿌려주기 등 순수 화면 처리)
    # =================================================================
    def _clear_treeview(self, tree):
        """Treeview(표) 위젯에 표시된 기존 데이터와 컬럼을 모두 지운다."""
        tree.delete(*tree.get_children())
        tree["columns"] = []

    def _show_df_in_treeview(self, tree, df, max_rows=100):
        """pandas DataFrame 내용을 tkinter Treeview(표)에 표시한다. (많으면 앞부분만)"""
        self._clear_treeview(tree)
        tree["columns"] = list(df.columns)
        tree["show"] = "headings"

        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")

        for _, row in df.head(max_rows).iterrows():
            tree.insert("", "end", values=list(row))

    def _draw_figure(self, container, fig):
        """matplotlib Figure를 지정된 컨테이너(프레임)에 그린다. (기존 그래프는 지우고 새로 그림)"""
        for widget in container.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return canvas

    def _check_data_loaded(self):
        """데이터가 필요한 동작 전에 파일이 로드됐는지 확인."""
        if self.analyzer.df is None:
            messagebox.showwarning("알림", "먼저 1번 탭에서 CSV 파일을 불러와주세요.")
            return False
        return True

    # =================================================================
    # 1) 파일 선택 탭
    # =================================================================
    def _build_file_tab(self):
        frame = self.tab_file

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Button(top, text="CSV 파일 열기", command=self._load_csv).pack(side="left")
        self.file_path_label = ttk.Label(top, text="선택된 파일 없음")
        self.file_path_label.pack(side="left", padx=10)

        preview_frame = ttk.LabelFrame(frame, text="데이터 미리보기 (상위 100행)")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.file_tree = ttk.Treeview(preview_frame)
        self.file_tree.pack(fill="both", expand=True, side="left")
        vsb = ttk.Scrollbar(preview_frame, orient="vertical", command=self.file_tree.yview)
        vsb.pack(side="right", fill="y")
        self.file_tree.configure(yscrollcommand=vsb.set)

        info_frame = ttk.LabelFrame(frame, text="데이터 요약 정보")
        info_frame.pack(fill="x", padx=10, pady=5)
        self.info_text = tk.Text(info_frame, height=8)
        self.info_text.pack(fill="both", expand=True)

    def _load_csv(self):
        """파일 대화상자를 열어 CSV를 선택하고, 실제 로딩은 DataAnalyzer에게 맡긴다."""
        file_path = filedialog.askopenfilename(
            title="CSV 파일 선택",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            df = self.analyzer.load_csv(file_path)
        except Exception as e:
            messagebox.showerror("오류", f"CSV 파일을 읽는 중 오류가 발생했습니다.\n{e}")
            return

        self.file_path_label.config(text=file_path)
        self._show_df_in_treeview(self.file_tree, df)
        self._update_file_info()
        self._refresh_column_choices()

    def _update_file_info(self):
        """데이터 요약 정보(DataAnalyzer가 계산)를 텍스트 위젯에 표시한다."""
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", self.analyzer.get_summary_text())

    # =================================================================
    # 2) 전처리 탭
    # =================================================================
    def _build_preprocess_tab(self):
        frame = self.tab_preprocess

        # ---- 결측치 처리 영역 ----
        missing_frame = ttk.LabelFrame(frame, text="① 결측치 처리")
        missing_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(missing_frame, text="처리 방법:").grid(row=0, column=0, padx=5, pady=5)
        self.missing_method = tk.StringVar(value="평균값으로 채우기")
        ttk.Combobox(
            missing_frame, textvariable=self.missing_method, state="readonly",
            values=["평균값으로 채우기", "중앙값으로 채우기", "최빈값으로 채우기", "결측치 행 삭제"]
        ).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(missing_frame, text="적용", command=self._apply_missing_handling).grid(row=0, column=2, padx=5)

        # ---- 범주형 인코딩 영역 ----
        encode_frame = ttk.LabelFrame(frame, text="② 범주형(문자) 컬럼 인코딩")
        encode_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(encode_frame, text="인코딩할 컬럼:").grid(row=0, column=0, padx=5, pady=5)
        self.encode_listbox = tk.Listbox(encode_frame, selectmode="multiple", height=5, exportselection=False)
        self.encode_listbox.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(encode_frame, text="라벨 인코딩 적용", command=self._apply_label_encoding).grid(row=0, column=2, padx=5)

        # ---- 스케일링 영역 ----
        scale_frame = ttk.LabelFrame(frame, text="③ 수치형 컬럼 스케일링")
        scale_frame.pack(fill="x", padx=10, pady=5)

        # 분류 문제의 타겟 컬럼처럼 스케일링하면 안 되는 컬럼이 섞여있을 수 있으므로 직접 선택
        ttk.Label(scale_frame, text="스케일링할 컬럼:").grid(row=0, column=0, padx=5, pady=5)
        self.scale_listbox = tk.Listbox(scale_frame, selectmode="multiple", height=5, exportselection=False)
        self.scale_listbox.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(scale_frame, text="스케일링 방법:").grid(row=0, column=2, padx=5, pady=5)
        self.scale_method = tk.StringVar(value="표준화 (StandardScaler)")
        ttk.Combobox(
            scale_frame, textvariable=self.scale_method, state="readonly",
            values=["표준화 (StandardScaler)", "정규화 (MinMaxScaler)"]
        ).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(scale_frame, text="적용", command=self._apply_scaling).grid(row=0, column=4, padx=5)

        # ---- 전처리 결과 미리보기 표 ----
        preview_frame = ttk.LabelFrame(frame, text="전처리 결과 미리보기")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.preprocess_tree = ttk.Treeview(preview_frame)
        self.preprocess_tree.pack(fill="both", expand=True, side="left")
        vsb = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preprocess_tree.yview)
        vsb.pack(side="right", fill="y")
        self.preprocess_tree.configure(yscrollcommand=vsb.set)

    def _apply_missing_handling(self):
        if not self._check_data_loaded():
            return
        df = self.analyzer.apply_missing_handling(self.missing_method.get())
        self._show_df_in_treeview(self.preprocess_tree, df)
        self._update_file_info()
        messagebox.showinfo("완료", "결측치 처리가 완료되었습니다.")

    def _apply_label_encoding(self):
        if not self._check_data_loaded():
            return

        selected_idx = self.encode_listbox.curselection()
        if not selected_idx:
            messagebox.showwarning("알림", "인코딩할 컬럼을 하나 이상 선택해주세요.")
            return
        columns = [self.encode_listbox.get(i) for i in selected_idx]

        df = self.analyzer.apply_label_encoding(columns)
        self._show_df_in_treeview(self.preprocess_tree, df)
        self._update_file_info()
        messagebox.showinfo("완료", f"{columns} 컬럼의 라벨 인코딩이 완료되었습니다.")

    def _apply_scaling(self):
        if not self._check_data_loaded():
            return

        selected_idx = self.scale_listbox.curselection()
        if not selected_idx:
            messagebox.showwarning("알림", "스케일링할 컬럼을 하나 이상 선택해주세요.")
            return
        columns = [self.scale_listbox.get(i) for i in selected_idx]

        df = self.analyzer.apply_scaling(columns, self.scale_method.get())
        self._show_df_in_treeview(self.preprocess_tree, df)
        messagebox.showinfo("완료", f"{columns} 컬럼의 스케일링이 완료되었습니다.\n"
                                    f"(분류 문제의 타겟 컬럼은 스케일링하지 마세요.)")

    def _refresh_column_choices(self):
        """CSV를 새로 불러왔을 때 전처리/모델링/시각화 탭의 컬럼 선택 목록들을 갱신한다."""
        columns = list(self.analyzer.df.columns)

        self.encode_listbox.delete(0, tk.END)
        for col in self.analyzer.get_object_columns():
            self.encode_listbox.insert(tk.END, col)

        self.scale_listbox.delete(0, tk.END)
        for col in self.analyzer.get_numeric_columns():
            self.scale_listbox.insert(tk.END, col)

        self.viz_x_combo["values"] = columns
        self.viz_y_combo["values"] = columns
        self.viz_hist_combo["values"] = columns

        self.target_combo["values"] = columns
        self.feature_listbox.delete(0, tk.END)
        for col in columns:
            self.feature_listbox.insert(tk.END, col)

    # =================================================================
    # 3) 시각화 탭
    # =================================================================
    def _build_visualize_tab(self):
        frame = self.tab_visualize

        control = ttk.Frame(frame)
        control.pack(fill="x", padx=10, pady=10)

        ttk.Label(control, text="분석 대상 컬럼:").grid(row=0, column=0, padx=5, pady=5)
        self.viz_hist_var = tk.StringVar()
        self.viz_hist_combo = ttk.Combobox(control, textvariable=self.viz_hist_var, state="readonly", width=15)
        self.viz_hist_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Button(control, text="히스토그램", command=self._plot_histogram).grid(row=0, column=2, padx=5)
        ttk.Button(control, text="박스플롯", command=self._plot_boxplot).grid(row=0, column=3, padx=5)

        ttk.Label(control, text="X축:").grid(row=1, column=0, padx=5, pady=5)
        self.viz_x_var = tk.StringVar()
        self.viz_x_combo = ttk.Combobox(control, textvariable=self.viz_x_var, state="readonly", width=15)
        self.viz_x_combo.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(control, text="Y축:").grid(row=1, column=2, padx=5, pady=5)
        self.viz_y_var = tk.StringVar()
        self.viz_y_combo = ttk.Combobox(control, textvariable=self.viz_y_var, state="readonly", width=15)
        self.viz_y_combo.grid(row=1, column=3, padx=5, pady=5)

        ttk.Button(control, text="산점도", command=self._plot_scatter).grid(row=1, column=4, padx=5)
        ttk.Button(control, text="상관관계 히트맵", command=self._plot_corr_heatmap).grid(row=1, column=5, padx=5)

        self.viz_canvas_frame = ttk.Frame(frame)
        self.viz_canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def _plot_histogram(self):
        if not self._check_data_loaded():
            return
        col = self.viz_hist_var.get()
        if not col:
            messagebox.showwarning("알림", "컬럼을 선택해주세요.")
            return
        fig = self.analyzer.make_histogram_figure(col)
        self._draw_figure(self.viz_canvas_frame, fig)

    def _plot_boxplot(self):
        if not self._check_data_loaded():
            return
        col = self.viz_hist_var.get()
        if not col:
            messagebox.showwarning("알림", "컬럼을 선택해주세요.")
            return
        fig = self.analyzer.make_boxplot_figure(col)
        self._draw_figure(self.viz_canvas_frame, fig)

    def _plot_scatter(self):
        if not self._check_data_loaded():
            return
        x_col, y_col = self.viz_x_var.get(), self.viz_y_var.get()
        if not x_col or not y_col:
            messagebox.showwarning("알림", "X축, Y축 컬럼을 모두 선택해주세요.")
            return
        fig = self.analyzer.make_scatter_figure(x_col, y_col)
        self._draw_figure(self.viz_canvas_frame, fig)

    def _plot_corr_heatmap(self):
        if not self._check_data_loaded():
            return
        try:
            fig = self.analyzer.make_corr_heatmap_figure()
        except ValueError as e:
            messagebox.showwarning("알림", str(e))
            return
        self._draw_figure(self.viz_canvas_frame, fig)

    # =================================================================
    # 4) 모델링 탭
    # =================================================================
    def _build_model_tab(self):
        frame = self.tab_model

        select_frame = ttk.LabelFrame(frame, text="① 예측 대상 및 입력 변수 선택")
        select_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(select_frame, text="예측할 컬럼 (Target):").grid(row=0, column=0, padx=5, pady=5, sticky="n")
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(select_frame, textvariable=self.target_var, state="readonly", width=20)
        self.target_combo.grid(row=0, column=1, padx=5, pady=5, sticky="n")

        ttk.Label(select_frame, text="입력 변수 (Feature, 다중선택):").grid(row=0, column=2, padx=5, pady=5, sticky="n")
        self.feature_listbox = tk.Listbox(select_frame, selectmode="multiple", height=6, exportselection=False)
        self.feature_listbox.grid(row=0, column=3, padx=5, pady=5)

        model_frame = ttk.LabelFrame(frame, text="② 문제 유형 및 모델 선택")
        model_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(model_frame, text="문제 유형:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Radiobutton(model_frame, text="회귀 (연속값 예측)", variable=self.problem_type,
                        value="회귀", command=self._update_model_choices).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(model_frame, text="분류 (범주값 예측)", variable=self.problem_type,
                        value="분류", command=self._update_model_choices).grid(row=0, column=2, padx=5)

        ttk.Label(model_frame, text="알고리즘 선택:").grid(row=1, column=0, padx=5, pady=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly", width=35)
        self.model_combo.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky="w")
        self._update_model_choices()

        ttk.Label(model_frame, text="테스트 데이터 비율:").grid(row=2, column=0, padx=5, pady=5)
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Spinbox(model_frame, from_=0.1, to=0.5, increment=0.05, textvariable=self.test_size_var,
                    width=8).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        ttk.Button(model_frame, text="모델 학습 시작", command=self._train_model).grid(row=2, column=2, padx=5)

        log_frame = ttk.LabelFrame(frame, text="학습 로그")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.model_log_text = tk.Text(log_frame)
        self.model_log_text.pack(fill="both", expand=True)

    def _update_model_choices(self):
        """회귀/분류 선택에 따라 알고리즘 콤보박스의 목록을 바꿔준다."""
        model_dict = REGRESSION_MODELS if self.problem_type.get() == "회귀" else CLASSIFICATION_MODELS
        self.model_combo["values"] = list(model_dict.keys())
        self.model_combo.current(0)

    def _train_model(self):
        if not self._check_data_loaded():
            return

        target = self.target_var.get()
        selected_idx = self.feature_listbox.curselection()
        features = [self.feature_listbox.get(i) for i in selected_idx]

        if not target:
            messagebox.showwarning("알림", "예측할 컬럼(Target)을 선택해주세요.")
            return
        if not features:
            messagebox.showwarning("알림", "입력 변수(Feature)를 하나 이상 선택해주세요.")
            return

        try:
            result = self.analyzer.train_model(
                target=target,
                features=features,
                problem_type=self.problem_type.get(),
                model_name=self.model_var.get(),
                test_size=self.test_size_var.get(),
            )
        except ValueError as e:
            messagebox.showerror("오류", str(e))
            return

        self.model_log_text.delete("1.0", tk.END)
        self.model_log_text.insert(
            "1.0",
            f"모델: {self.model_var.get()}\n"
            f"문제 유형: {self.problem_type.get()}\n"
            f"Target: {target}\n"
            f"Feature: {features}\n"
            f"학습 데이터 수: {result['train_count']} / 테스트 데이터 수: {result['test_count']}\n\n"
            f"학습이 완료되었습니다. '5. 평가' 탭에서 결과를 확인하세요."
        )

        self._build_predict_inputs()
        messagebox.showinfo("완료", "모델 학습이 완료되었습니다. 평가 탭을 확인해주세요.")

    # =================================================================
    # 5) 평가 탭
    # =================================================================
    def _build_evaluate_tab(self):
        frame = self.tab_evaluate

        top = ttk.Frame(frame)
        top.pack(fill="x", padx=10, pady=10)
        ttk.Button(top, text="평가 결과 보기", command=self._evaluate_model).pack(side="left")

        self.eval_text = tk.Text(frame, height=8)
        self.eval_text.pack(fill="x", padx=10, pady=5)

        self.eval_canvas_frame = ttk.Frame(frame)
        self.eval_canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def _evaluate_model(self):
        if self.analyzer.model is None:
            messagebox.showwarning("알림", "먼저 4번 탭에서 모델을 학습해주세요.")
            return

        if self.problem_type.get() == "회귀":
            text, fig = self.analyzer.evaluate_regression()
        else:
            text, fig = self.analyzer.evaluate_classification()

        self.eval_text.delete("1.0", tk.END)
        self.eval_text.insert("1.0", text)
        self._draw_figure(self.eval_canvas_frame, fig)

    # =================================================================
    # 6) 예측 탭
    # =================================================================
    def _build_predict_tab(self):
        frame = self.tab_predict

        self.predict_input_frame = ttk.LabelFrame(frame, text="입력 변수 값 입력")
        self.predict_input_frame.pack(fill="x", padx=10, pady=10)
        self.predict_entries = {}  # {컬럼명: Entry위젯}

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(btn_frame, text="예측 실행", command=self._run_prediction).pack(side="left")

        self.predict_result_label = ttk.Label(frame, text="예측 결과가 여기에 표시됩니다.", font=("맑은 고딕", 14))
        self.predict_result_label.pack(padx=10, pady=20)

    def _build_predict_inputs(self):
        """모델 학습이 끝난 뒤, 학습에 사용된 feature 컬럼 기준으로 입력폼을 새로 만든다."""
        for widget in self.predict_input_frame.winfo_children():
            widget.destroy()
        self.predict_entries = {}

        for i, col in enumerate(self.analyzer.feature_columns):
            ttk.Label(self.predict_input_frame, text=col).grid(row=i // 4, column=(i % 4) * 2, padx=5, pady=5, sticky="e")
            entry = ttk.Entry(self.predict_input_frame, width=15)
            entry.grid(row=i // 4, column=(i % 4) * 2 + 1, padx=5, pady=5)
            self.predict_entries[col] = entry

    def _run_prediction(self):
        if self.analyzer.model is None:
            messagebox.showwarning("알림", "먼저 4번 탭에서 모델을 학습해주세요.")
            return

        input_values = {col: entry.get() for col, entry in self.predict_entries.items()}

        try:
            prediction = self.analyzer.predict(input_values)
        except Exception as e:
            messagebox.showerror("오류", f"예측 중 오류가 발생했습니다.\n{e}")
            return

        target = self.analyzer.target_column
        if self.problem_type.get() == "회귀":
            self.predict_result_label.config(text=f"예측 결과 ({target}): {prediction:.4f}")
        else:
            self.predict_result_label.config(text=f"예측 결과 ({target}): {prediction}")
