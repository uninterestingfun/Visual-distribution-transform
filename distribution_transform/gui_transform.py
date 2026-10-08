import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass

import re
import numpy as np
from scipy import stats
from scipy.special import expit

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import font_manager
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize

# ---------- 中文字体 ----------
candidates = ['SimHei', 'Microsoft YaHei', 'PingFang SC',
              'Hiragino Sans GB', 'Arial Unicode MS',
              'Noto Sans CJK SC', 'Source Han Sans SC',
              'WenQuanYi Micro Hei']
avail = {f.name for f in font_manager.fontManager.ttflist}
chosen = next((f for f in candidates if f in avail), None)
if chosen:
    rcParams['font.sans-serif'] = [chosen]
rcParams['axes.unicode_minus'] = False


def smoothstep(t):
    return t * t * (3 - 2 * t)


@dataclass
class Transform:
    """任意函数变换 x' = f(x,y), y' = g(x,y)，用表达式定义（统一仿射与非线性）"""
    fx_expr: str  # x' 的表达式（含 x, y）
    fy_expr: str  # y' 的表达式（含 x, y）


try:
    import cupy as _cp
    HAS_GPU = True
except Exception:
    _cp = None
    HAS_GPU = False

GPU_MIN_POINTS = 5000  # 点数超过此阈值才启用 GPU 加速


# 表达式求值安全命名空间：支持的函数与常数（xp 为 numpy 或 cupy）
def _build_ns(xp):
    return {
        'pi': np.pi, 'e': np.e,
        'sin': xp.sin, 'cos': xp.cos, 'tan': xp.tan,
        'arcsin': xp.arcsin, 'arccos': xp.arccos, 'arctan': xp.arctan,
        'sinh': xp.sinh, 'cosh': xp.cosh, 'tanh': xp.tanh,
        'exp': xp.exp,
        'log': lambda u: xp.log(xp.maximum(u, 1e-9)),
        'log2': lambda u: xp.log2(xp.maximum(u, 1e-9)),
        'log10': lambda u: xp.log10(xp.maximum(u, 1e-9)),
        'sqrt': lambda u: xp.sqrt(xp.maximum(u, 0.0)),
        'abs': xp.abs, 'sign': xp.sign,
        'sigmoid': lambda u: 1.0 / (1.0 + xp.exp(-u)),
        'relu': lambda u: xp.maximum(0.0, u),
        'leaky_relu': lambda u, a=0.01: xp.where(u > 0, u, a * u),
        'softplus': lambda u: xp.log1p(xp.exp(u)),
        'floor': xp.floor, 'ceil': xp.ceil, 'round': xp.round,
    }


SAFE_NS = _build_ns(np)

FUNC_DOC = ('sin cos tan tanh exp log sqrt abs sigmoid relu '
            'leaky_relu softplus arcsin arccos arctan sinh cosh '
            'floor ceil round')


_EXPR_CACHE = {}


def _compile_expr(expr):
    code = _EXPR_CACHE.get(expr)
    if code is None:
        code = compile(expr, '<expr>', 'eval')
        _EXPR_CACHE[expr] = code
    return code


def eval_expr(expr, x, y):
    """在受限命名空间内求值表达式（x, y 可为 numpy 或 cupy 数组）"""
    xp = _cp if (HAS_GPU and isinstance(x, _cp.ndarray)) else np
    ns = _build_ns(xp)
    ns['x'] = x
    ns['y'] = y
    return eval(_compile_expr(expr), {'__builtins__': {'__import__': __import__}},
                ns)


def num_jacobian_det(fx_expr, fy_expr, x, y, s, h=1e-5):
    """数值计算部分变换 T_s 的 Jacobian 行列式绝对值（逐点）。
    T_s(x,y) = ((1-s)x + s·f(x,y), (1-s)y + s·g(x,y))"""
    xp = _cp if (HAS_GPU and isinstance(x, _cp.ndarray)) else np
    fx_dep_y = 'y' in fx_expr
    fy_dep_x = 'x' in fy_expr
    dfx_dx = (eval_expr(fx_expr, x + h, y) - eval_expr(fx_expr, x - h, y)) / (2 * h)
    dfy_dy = (eval_expr(fy_expr, x, y + h) - eval_expr(fy_expr, x, y - h)) / (2 * h)
    if fx_dep_y:
        dfx_dy = (eval_expr(fx_expr, x, y + h) - eval_expr(fx_expr, x, y - h)) / (2 * h)
    else:
        dfx_dy = 0.0
    if fy_dep_x:
        dfy_dx = (eval_expr(fy_expr, x + h, y) - eval_expr(fy_expr, x - h, y)) / (2 * h)
    else:
        dfy_dx = 0.0
    j11 = (1 - s) + s * dfx_dx
    j12 = s * dfx_dy
    j21 = s * dfy_dx
    j22 = (1 - s) + s * dfy_dy
    return xp.abs(j11 * j22 - j12 * j21)


def apply_transform(tr, xy, s):
    """对点集 xy (N,2) 应用部分变换（s=1 完全，s=0 恒等）。
    返回 (新坐标, 每点 Jacobian 行列式绝对值)。"""
    use_gpu = HAS_GPU and len(xy) >= GPU_MIN_POINTS
    if use_gpu:
        x = _cp.asarray(xy[:, 0])
        y = _cp.asarray(xy[:, 1])
        xp = _cp
    else:
        x = xy[:, 0]
        y = xy[:, 1]
        xp = np

    xn = (1 - s) * x + s * eval_expr(tr.fx_expr, x, y)
    yn = (1 - s) * y + s * eval_expr(tr.fy_expr, x, y)
    xy_new = xp.stack([xn, yn], axis=1)
    det = num_jacobian_det(tr.fx_expr, tr.fy_expr, x, y, s)
    det = xp.nan_to_num(det, nan=1e-3, posinf=1e-3, neginf=1e-3)
    xy_new = xp.nan_to_num(xy_new, nan=0.0, posinf=1e6, neginf=-1e6)
    det = xp.maximum(det, 1e-3)

    if use_gpu:
        return _cp.asnumpy(xy_new), _cp.asnumpy(det)
    return xy_new, det


# 表达式 -> matplotlib mathtext 书面公式
MATH_SYMBOLS = {
    'sin': r'\sin', 'cos': r'\cos', 'tan': r'\tan',
    'arcsin': r'\arcsin', 'arccos': r'\arccos', 'arctan': r'\arctan',
    'sinh': r'\sinh', 'cosh': r'\cosh', 'tanh': r'\tanh',
    'exp': r'\exp', 'log': r'\log',
    'log2': r'\log_2', 'log10': r'\log_{10}',
    'sigmoid': r'\sigma', 'pi': r'\pi',
}


def expr_to_math(expr):
    """把 Python 表达式转为 mathtext 书面公式（** -> ^，* -> ×，函数名规范化）"""
    s = expr.replace('**', '^')
    s = s.replace('*', r'\times\,')
    s = re.sub(r'sqrt\(([^()]*)\)',
               lambda m: r'\sqrt{' + m.group(1) + '}', s)
    s = re.sub(r'abs\(([^()]*)\)',
               lambda m: r'\left|' + m.group(1) + r'\right|', s)

    def repl(m):
        name = m.group(0)
        if name in MATH_SYMBOLS:
            return MATH_SYMBOLS[name]
        if name in ('relu', 'leaky_relu', 'softplus', 'floor', 'ceil',
                    'round', 'sign', 'e'):
            labels = {'relu': 'ReLU', 'leaky_relu': 'LeakyReLU',
                      'softplus': 'softplus', 'floor': 'floor',
                      'ceil': 'ceil', 'round': 'round', 'sign': 'sgn',
                      'e': 'e'}
            return r'\mathrm{' + labels[name] + '}'
        return name

    return re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', repl, s)


UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

ELEV, AZIM = 26, -58
FRAMES_PER_ROUND = 45

DIST_TYPES = ['均匀', '高斯', '二项', '泊松']

CMAPS = ['viridis', 'plasma', 'inferno', 'jet', 'coolwarm', 'turbo']

DIST_PARAM_LABELS = {
    '均匀': [],
    '高斯': ['μx', 'μy', 'σx', 'σy'],
    '二项': ['n', 'p'],
    '泊松': ['λ'],
}
DIST_PARAM_DEFAULTS = {
    '均匀': [],
    '高斯': ['0', '0', '1', '1'],
    '二项': ['10', '0.4'],
    '泊松': ['3'],
}


class TransformApp:
    def __init__(self, root):
        self.root = root
        root.title("二维分布变换可视化 - 多轮复合")
        root.configure(bg='white')

        self.transforms = []
        self.anim = None
        self.dist_var = tk.StringVar(value='均匀')
        self.dist_var.trace_add('write', self._on_dist_change)
        self._last_dist = '均匀'
        self.dist_param_store = {k: list(v) for k, v in DIST_PARAM_DEFAULTS.items()}
        self.paused = False
        self.point_size = 12.0
        self.current_frame = 0
        self._syncing_progress = False
        self.view_mode = 'animation'  # 'animation' 或 'final'
        self.density_weight = 0.0
        self.final_weight = 2.0
        self.user_limits = None   # 用户调整后的坐标轴范围

        self._build_ui()
        self.cmap_var.trace_add('write', self._on_cmap_change)
        self._build_figure()
        self._sync_param_ui()
        self.precompute()
        self._setup_colorbar()

    # ---------------- UI ----------------
    def _build_ui(self):
        control = ttk.Frame(self.root, padding=8)
        control.pack(side=tk.LEFT, fill=tk.Y)

        # 分布选择 + 采样点数
        dist_row = ttk.Frame(control)
        dist_row.pack(fill=tk.X)
        ttk.Label(dist_row, text="初始分布:").pack(side=tk.LEFT)
        self.dist_combo = ttk.Combobox(dist_row, textvariable=self.dist_var,
                                       values=DIST_TYPES, state='readonly',
                                       width=6)
        self.dist_combo.pack(side=tk.LEFT, padx=4)
        ttk.Label(dist_row, text="点数:").pack(side=tk.LEFT, padx=(8, 0))
        self.n_points_entry = ttk.Entry(dist_row, width=6)
        self.n_points_entry.insert(0, '3000')
        self.n_points_entry.pack(side=tk.LEFT, padx=2)

        # 点分布权重（0=均匀，越大越集中高密度）
        weight_row = ttk.Frame(control)
        weight_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(weight_row, text="点分布:").pack(side=tk.LEFT)
        self.weight_scale = ttk.Scale(weight_row, from_=0, to=4, value=0,
                                      orient='horizontal',
                                      command=self._on_weight_change)
        self.weight_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.weight_label = ttk.Label(weight_row, text="0.0(均匀)", width=10)
        self.weight_label.pack(side=tk.LEFT)

        # 最终分布权重
        final_weight_row = ttk.Frame(control)
        final_weight_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(final_weight_row, text="最终权重:").pack(side=tk.LEFT)
        self.final_weight_scale = ttk.Scale(
            final_weight_row, from_=0, to=6, value=2, orient='horizontal',
            command=self._on_final_weight_change)
        self.final_weight_scale.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                     padx=4)
        self.final_weight_label = ttk.Label(final_weight_row, text="2.0",
                                            width=6)
        self.final_weight_label.pack(side=tk.LEFT)

        # 点大小
        size_row = ttk.Frame(control)
        size_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(size_row, text="点大小:").pack(side=tk.LEFT)
        self.size_scale = ttk.Scale(size_row, from_=1, to=100, value=12,
                                    orient='horizontal',
                                    command=self._on_size_change)
        self.size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.size_label = ttk.Label(size_row, text="12", width=4)
        self.size_label.pack(side=tk.LEFT)

        # 颜色映射
        cmap_row = ttk.Frame(control)
        cmap_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(cmap_row, text="配色:").pack(side=tk.LEFT)
        self.cmap_var = tk.StringVar(value='viridis')
        self.cmap_combo = ttk.Combobox(cmap_row, textvariable=self.cmap_var,
                                       values=CMAPS, state='readonly', width=8)
        self.cmap_combo.pack(side=tk.LEFT, padx=4)

        # 分布参数
        self.param_frame = ttk.LabelFrame(control, text="分布参数", padding=6)
        self.param_frame.pack(fill=tk.X, pady=4)
        self.param_entries = {}
        self.param_labels = {}
        for i in range(4):
            row = ttk.Frame(self.param_frame)
            row.pack(fill=tk.X, pady=1)
            lbl = ttk.Label(row, text="", width=4)
            lbl.pack(side=tk.LEFT)
            e = ttk.Entry(row, width=8)
            e.pack(side=tk.LEFT, padx=2)
            self.param_labels[i] = lbl
            self.param_entries[i] = e

        # 变换列表
        ttk.Label(control, text="变换序列（自上而下依次复合）").pack(anchor='w')
        cols = ('#', '变换')
        self.tree = ttk.Treeview(control, columns=cols, show='headings',
                                 height=8)
        self.tree.heading('#', text='#')
        self.tree.column('#', width=30, anchor='center')
        self.tree.heading('变换', text='变换')
        self.tree.column('变换', width=230, anchor='center')
        self.tree.pack(fill=tk.X, pady=(2, 8))

        # 变换输入
        form = ttk.LabelFrame(control, text="新增变换", padding=6)
        form.pack(fill=tk.X)

        nl_x = ttk.Frame(form)
        nl_x.pack(anchor='w', pady=2)
        ttk.Label(nl_x, text="x' = f(x,y):").pack(side=tk.LEFT)
        self.fx_expr_entry = ttk.Entry(nl_x, width=30)
        self.fx_expr_entry.insert(0, '2*x + 1')
        self.fx_expr_entry.pack(side=tk.LEFT, padx=4)

        nl_y = ttk.Frame(form)
        nl_y.pack(anchor='w', pady=2)
        ttk.Label(nl_y, text="y' = g(x,y):").pack(side=tk.LEFT)
        self.fy_expr_entry = ttk.Entry(nl_y, width=30)
        self.fy_expr_entry.insert(0, '2*y + 1')
        self.fy_expr_entry.pack(side=tk.LEFT, padx=4)

        doc = ttk.Label(form, text=(
            '规范：变量 x、y；运算符 + - * / ** ^；'
            '可用函数 sin cos tan tanh exp log sqrt abs sigmoid relu '
            'leaky_relu softplus；常数 pi、e。\n'
            '示例：仿射 2*x+1；非线性 x**2+y**2, sin(x)*cos(y), '
            'exp(-(x**2+y**2)), relu(x)+0.5*y'),
            foreground='#666', justify='left', wraplength=360)
        doc.pack(anchor='w', pady=(2, 0))

        btns = ttk.Frame(control)
        btns.pack(fill=tk.X, pady=6)
        ttk.Button(btns, text="添加变换", command=self.add_transform).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(btns, text="删除选中", command=self.del_transform).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(btns, text="清空全部", command=self.clear_all).pack(
            side=tk.LEFT, padx=2)

        ctl = ttk.Frame(control)
        ctl.pack(fill=tk.X, pady=6)
        ttk.Button(ctl, text="生成并播放动画", command=self.play).pack(
            side=tk.LEFT, padx=2)
        self.pause_btn = ttk.Button(ctl, text="暂停", command=self.toggle_pause)
        self.pause_btn.pack(side=tk.LEFT, padx=2)
        self.final_btn = ttk.Button(ctl, text="查看最终分布",
                                    command=self.toggle_final_view)
        self.final_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(ctl, text="保存 GIF", command=self.save_gif).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(ctl, text="保存最终分布图",
                   command=self.generate_final_distribution).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(ctl, text="重置视角", command=self.reset_view).pack(
            side=tk.LEFT, padx=2)

        # 进度条
        prog_row = ttk.Frame(control)
        prog_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(prog_row, text="进度:").pack(side=tk.LEFT)
        self.progress_scale = ttk.Scale(prog_row, from_=0, to=1,
                                        orient='horizontal',
                                        command=self._on_progress_change)
        self.progress_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.progress_label = ttk.Label(prog_row, text="0/0", width=10)
        self.progress_label.pack(side=tk.LEFT)

        self.status = tk.Label(control, text="", bg='white', fg='#333',
                               anchor='w', justify='left')
        self.status.pack(fill=tk.X, pady=4)

    def _build_figure(self):
        self.fig = Figure(figsize=(8, 7), facecolor='white')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=ELEV, azim=AZIM)
        self.fig.subplots_adjust(top=0.86, bottom=0.04, left=0.02, right=0.94)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # 使用 matplotlib 自带的旋转/平移/缩放交互，仅在交互后捕获坐标轴范围
        self.canvas.mpl_connect('button_release_event', self._capture_view)
        self.canvas.mpl_connect('scroll_event', self._capture_view)

    def _capture_view(self, event):
        """matplotlib 默认交互（左键旋转/中键平移/右键缩放/滚轮）后，
        捕获用户调整后的坐标轴范围并同步刻度与网格。"""
        xlim, ylim, zlim = self._update_ticks(
            self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim())
        self.user_limits = (xlim, ylim, zlim)
        self.canvas.draw_idle()

    def _update_ticks(self, xlim, ylim, zlim):
        """根据当前坐标轴范围重新生成刻度，并把范围对齐到刻度端点，
        保证网格线完整覆盖整个视图。返回对齐后的 (xlim, ylim, zlim)。"""
        self.xticks = self._nice_ticks(xlim[0], xlim[1], 8)
        self.yticks = self._nice_ticks(ylim[0], ylim[1], 8)
        self.zticks = self._nice_ticks(zlim[0], zlim[1], 5)
        xlim = (self.xticks[0], self.xticks[-1])
        ylim = (self.yticks[0], self.yticks[-1])
        zlim = (self.zticks[0], self.zticks[-1])
        self.ax.set_xticks(self.xticks)
        self.ax.set_yticks(self.yticks)
        self.ax.set_zticks(self.zticks)
        self.ax.set_xticklabels([f'{v:g}' for v in self.xticks])
        self.ax.set_yticklabels([f'{v:g}' for v in self.yticks])
        self.ax.set_zticklabels([f'{v:g}' for v in self.zticks])
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_zlim(*zlim)
        return xlim, ylim, zlim

    def _setup_colorbar(self):
        from matplotlib import cm
        self.norm = Normalize(vmin=0, vmax=max(self.zmax, 1e-9))
        self.mappable = cm.ScalarMappable(norm=self.norm,
                                          cmap=self.cmap_var.get())
        self.mappable.set_array([])
        self.cbar = self.fig.colorbar(self.mappable, ax=self.ax,
                                      shrink=0.6, pad=0.12)
        self.cbar.set_label('概率密度')
        self._sync_colorbar_ticks()

    def _sync_colorbar_ticks(self):
        """按当前 norm 范围设置 colorbar 的刻度与标签"""
        ticks = self._nice_ticks(0.0, self.norm.vmax, 5)
        self.cbar.set_ticks(ticks)
        self.cbar.set_ticklabels([f'{v:g}' for v in ticks])

    # ---------------- 分布参数 ----------------
    def _on_dist_change(self, *args):
        self._store_params(self._last_dist)
        self._last_dist = self.dist_var.get()
        self._sync_param_ui()
        self.precompute()

    def _store_params(self, dist):
        labels = DIST_PARAM_LABELS[dist]
        vals = [self.param_entries[i].get() for i in range(len(labels))]
        self.dist_param_store[dist] = vals

    def _sync_param_ui(self):
        dist = self.dist_var.get()
        labels = DIST_PARAM_LABELS[dist]
        stored = self.dist_param_store.get(dist, DIST_PARAM_DEFAULTS[dist])
        for i in range(4):
            if i < len(labels):
                self.param_labels[i].config(text=labels[i])
                e = self.param_entries[i]
                e.config(state='normal')
                e.delete(0, tk.END)
                e.insert(0, stored[i])
            else:
                self.param_labels[i].config(text="")
                e = self.param_entries[i]
                e.config(state='disabled')
                e.delete(0, tk.END)

    def _get_dist_params(self):
        dist = self.dist_var.get()
        labels = DIST_PARAM_LABELS[dist]
        vals = []
        for i in range(len(labels)):
            try:
                vals.append(float(self.param_entries[i].get()))
            except ValueError:
                raise ValueError(f"参数 {labels[i]} 无效")
        return dist, vals

    def _get_n_points(self):
        try:
            n = int(self.n_points_entry.get())
            if n < 1:
                raise ValueError
            return n
        except ValueError:
            raise ValueError("采样点数无效")

    # ---------------- 采样：初始分布数据点 ----------------
    def _sample_points(self, n_points):
        dist, p = self._get_dist_params()
        rng = np.random.default_rng(42)
        weight = self.density_weight

        # 均匀采样候选点（初始位置 + 初始密度）
        n_cand = max(n_points * 10, 5000)
        xy_cand, f_cand = self._uniform_candidates(n_cand, rng)

        # 根据最终分布的概率密度选择：变换到最终分布，计算最终密度
        kind = 'discrete' if dist in ('二项', '泊松') else 'continuous'
        xy_p = xy_cand
        det_prod = np.ones(n_cand)
        for tr in self.transforms:
            xy_p, det = apply_transform(tr, xy_p, 1.0)
            det_prod = det_prod * det
        if kind == 'discrete':
            f_p = f_cand
        else:
            f_p = f_cand / np.maximum(det_prod, 1e-3)

        # 按最终密度选择：weight=0 时均匀采样（空间均匀覆盖），
        # weight>0 时按 f_p^weight 加权（越大越集中高密度区域）
        if weight == 0:
            idx = rng.integers(0, n_cand, size=n_points)
        else:
            w = f_p ** weight
            w = w / w.sum()
            idx = rng.choice(n_cand, size=n_points, p=w)

        xy = xy_cand[idx]
        f = f_cand[idx]
        z = rng.uniform(0.0, f)

        if dist == '均匀':
            bounds = UNIT_SQUARE.copy()
            peak = 1.0
        elif dist == '高斯':
            mux, muy, sigx, sigy = p
            bounds = np.array([[mux - 4 * sigx, muy - 4 * sigy],
                               [mux + 4 * sigx, muy - 4 * sigy],
                               [mux + 4 * sigx, muy + 4 * sigy],
                               [mux - 4 * sigx, muy + 4 * sigy]])
            peak = 1.0 / (2 * np.pi * sigx * sigy)
        elif dist == '二项':
            n = int(round(p[0]))
            prob = float(p[1])
            pk = stats.binom.pmf(np.arange(n + 1), n, prob)
            bounds = np.array([[0, 0], [n, 0], [n, n], [0, n]], dtype=float)
            peak = float(np.max(pk)) ** 2
        elif dist == '泊松':
            lam = float(p[0])
            kmax = int(round(lam + 6 * np.sqrt(max(lam, 0.01)) + 5))
            pk = stats.poisson.pmf(np.arange(kmax + 1), lam)
            bounds = np.array([[0, 0], [kmax, 0], [kmax, kmax], [0, kmax]],
                              dtype=float)
            peak = float(np.max(pk)) ** 2
        else:
            raise ValueError("未知分布")

        return kind, xy, f, z, bounds, peak

    # ---------------- 数据预计算 ----------------
    def precompute(self):
        try:
            n_points = self._get_n_points()
            kind, xy, f, z, bounds, peak = self._sample_points(n_points)
        except ValueError as e:
            self.status.config(text=str(e))
            kind = 'continuous'
            xy = UNIT_SQUARE.copy()
            f = np.ones(4)
            z = np.ones(4)
            bounds = UNIT_SQUARE.copy()
            peak = 1.0

        self.kind = kind
        self.sample_xy = xy
        self.sample_f = f
        self.sample_z = z
        self.peak = peak

        # 依次应用所有变换，缓存每轮终点的点集与累积密度缩放
        self.state_xy = [self.sample_xy.copy()]
        self.state_det = [np.ones(len(self.sample_xy))]
        densities = [self.sample_f]
        det_prod = np.ones(len(self.sample_xy))
        cur = self.sample_xy.copy()
        for tr in self.transforms:
            cur, det = apply_transform(tr, cur, 1.0)
            det_prod = det_prod * det
            self.state_xy.append(cur.copy())
            self.state_det.append(det_prod.copy())
            densities.append(self.sample_f / det_prod)

        states_xy = self.state_xy
        # 坐标轴范围（基于所有轮终点的实际点集，过滤 NaN/Inf）
        all_pts = np.vstack(states_xy)
        all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
        if len(all_pts) == 0:
            all_pts = UNIT_SQUARE.copy()
        span = max(np.nanmax(np.ptp(all_pts, axis=0)), 0.5)
        pad = 0.15 * span
        xlo = all_pts[:, 0].min() - pad
        xhi = all_pts[:, 0].max() + pad
        ylo = all_pts[:, 1].min() - pad
        yhi = all_pts[:, 1].max() + pad

        # 密度高度：用实际最大密度，使 colorbar 覆盖整个采样概率密度的值域
        if self.kind == 'discrete':
            max_h = self.peak
        else:
            dens = np.concatenate(densities)
            dens = dens[np.isfinite(dens)]
            if len(dens):
                max_h = float(np.max(dens))
            else:
                max_h = 1.0
            max_h = max(max_h, 1e-3)

        self.xticks = self._nice_ticks(xlo, xhi, 8)
        self.yticks = self._nice_ticks(ylo, yhi, 8)
        self.zticks = self._nice_ticks(0.0, max_h, 5)

        self.xlim = (self.xticks[0], self.xticks[-1])
        self.ylim = (self.yticks[0], self.yticks[-1])
        self.zmax = self.zticks[-1]

        # 保存默认刻度，供重置视角时恢复
        self._default_xticks = self.xticks.copy()
        self._default_yticks = self.yticks.copy()
        self._default_zticks = self.zticks.copy()

        if hasattr(self, 'norm'):
            self.norm.vmin = 0
            self.norm.vmax = self.zmax
            self.mappable.set_norm(self.norm)
            self._sync_colorbar_ticks()
            self.cbar.draw_all()

        self.total_frames = max(1, len(self.transforms) * FRAMES_PER_ROUND)
        if hasattr(self, 'progress_scale'):
            self._syncing_progress = True
            try:
                self.progress_scale.set(0)
                self.progress_scale.config(to=self.total_frames - 1)
                self.progress_label.config(text=f"0/{self.total_frames - 1}")
            finally:
                self._syncing_progress = False

    @staticmethod
    def _nice_ticks(lo, hi, n=5):
        span = hi - lo
        if span <= 0:
            return np.array([float(lo)])
        step = span / max(n, 1)
        mag = 10.0 ** np.floor(np.log10(step))
        for m in (1.0, 2.0, 2.5, 5.0, 10.0):
            if step <= m * mag:
                step = m * mag
                break
        k0 = int(np.ceil(lo / step - 1e-9))
        k1 = int(np.ceil(hi / step - 1e-9))
        return np.array([k * step for k in range(k0, k1 + 1)])

    # ---------------- 事件 ----------------
    def _validate_expr(self, expr):
        """验证表达式能否在样本点上求值，失败抛出异常"""
        test = np.array([0.0, 1.0])
        eval_expr(expr, test, test)

    def add_transform(self):
        fx_expr = self.fx_expr_entry.get().strip()
        fy_expr = self.fy_expr_entry.get().strip()
        if not fx_expr or not fy_expr:
            messagebox.showerror("表达式错误", "请输入 x' 和 y' 的表达式")
            return
        try:
            self._validate_expr(fx_expr)
            self._validate_expr(fy_expr)
        except Exception as e:
            messagebox.showerror("表达式错误", f"函数表达式无效：{e}")
            return
        tr = Transform(fx_expr, fy_expr)
        desc = f"x'={fx_expr}, y'={fy_expr}"
        self.transforms.append(tr)
        idx = len(self.transforms)
        self.tree.insert('', 'end', values=(idx, desc))
        self.precompute()

    def del_transform(self):
        sel = self.tree.selection()
        if not sel:
            return
        for item in sel:
            vals = self.tree.item(item, 'values')
            idx = int(vals[0]) - 1
            self.transforms.pop(idx)
            self.tree.delete(item)
        self._refresh_tree()

    def clear_all(self):
        self.transforms.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.precompute()

    def _refresh_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, tr in enumerate(self.transforms, start=1):
            desc = f"x'={tr.fx_expr}, y'={tr.fy_expr}"
            self.tree.insert('', 'end', values=(i, desc))
        self.precompute()

    # ---------------- 动画 ----------------
    def play(self):
        self.precompute()
        self.root.focus_set()
        total = self.total_frames
        if self.anim is not None:
            self.anim.event_source.stop()
        self.anim = FuncAnimation(self.fig, self.update, frames=total,
                                  interval=60, repeat=True)
        self.paused = False
        self.pause_btn.config(text="暂停")
        self.canvas.draw()

    def toggle_pause(self):
        if self.anim is None or self.anim.event_source is None:
            return
        if self.paused:
            self.anim.event_source.start()
            self.pause_btn.config(text="暂停")
            self.paused = False
        else:
            self.anim.event_source.stop()
            self.pause_btn.config(text="继续")
            self.paused = True

    def _on_size_change(self, val):
        self.point_size = float(val)
        self.size_label.config(text=f"{int(round(self.point_size))}")
        if self.paused or self.anim is None:
            self.update(self.current_frame)
            self.canvas.draw_idle()

    def _on_weight_change(self, val):
        self.density_weight = float(val)
        self.weight_label.config(
            text=f"{self.density_weight:.1f}" +
                 ("(均匀)" if self.density_weight == 0 else ""))
        self.precompute()
        if self.view_mode == 'final':
            self._draw_final_distribution()
        else:
            self.update(self.current_frame)
        self.canvas.draw_idle()

    def _on_final_weight_change(self, val):
        self.final_weight = float(val)
        self.final_weight_label.config(text=f"{self.final_weight:.1f}")
        if self.view_mode == 'final':
            self._draw_final_distribution()
            self.canvas.draw_idle()

    def _on_cmap_change(self, *args):
        if hasattr(self, 'mappable'):
            self.mappable.set_cmap(self.cmap_var.get())
            self.cbar.draw_all()
        self.update(self.current_frame)
        self.canvas.draw_idle()

    def _on_progress_change(self, val):
        if self._syncing_progress:
            return
        frame = int(round(float(val)))
        frame = max(0, min(frame, self.total_frames - 1))
        if self.anim is not None and not self.paused:
            self.anim.event_source.stop()
            self.paused = True
            self.pause_btn.config(text="继续")
        self.current_frame = frame
        self.update(frame)
        self.canvas.draw_idle()

    def save_gif(self):
        if not self.transforms:
            messagebox.showinfo("提示", "请先添加至少一个变换")
            return
        self.precompute()
        path = filedialog.asksaveasfilename(
            defaultextension='.gif', filetypes=[('GIF', '*.gif')],
            initialfile='transform_animation.gif')
        if not path:
            return
        total = self.total_frames
        anim = FuncAnimation(self.fig, self.update, frames=total, interval=60)
        self.status.config(text="正在渲染 GIF，请稍候...")
        self.root.update_idletasks()
        anim.save(path, writer=PillowWriter(fps=18), dpi=95,
                  savefig_kwargs={'facecolor': 'white'})
        self.status.config(text=f"已保存: {path}")

    def generate_final_distribution(self):
        if not self.transforms:
            messagebox.showinfo("提示", "请先添加至少一个变换")
            return
        self.precompute()
        path = filedialog.asksaveasfilename(
            defaultextension='.png', filetypes=[('PNG', '*.png')],
            initialfile='final_distribution.png')
        if not path:
            return

        # 与 UI 界面一致的最终分布散点云
        xy, f_p, z_p = self._final_points()
        fig = Figure(figsize=(8, 7), facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        sc = self._plot_final_points(ax, xy, f_p, z_p)
        fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.12, label='概率密度')
        fig.savefig(path, dpi=150, facecolor='white')
        self.status.config(text=f"已保存最终分布图: {path}")

    def _uniform_candidates(self, n_cand, rng):
        """在初始分布支撑区域均匀采样候选点，返回 (xy, 初始密度)"""
        dist, p = self._get_dist_params()
        if dist == '均匀':
            xy = rng.uniform(0.0, 1.0, size=(n_cand, 2))
            f = np.full(n_cand, 1.0)
        elif dist == '高斯':
            mux, muy, sigx, sigy = p
            xy = rng.uniform([mux - 4 * sigx, muy - 4 * sigy],
                             [mux + 4 * sigx, muy + 4 * sigy],
                             size=(n_cand, 2))
            f = stats.norm.pdf(xy[:, 0], mux, sigx) * \
                stats.norm.pdf(xy[:, 1], muy, sigy)
        elif dist == '二项':
            n = int(round(p[0]))
            prob = float(p[1])
            ks = np.arange(n + 1)
            pk = stats.binom.pmf(ks, n, prob)
            grid = np.array([[k, l] for k in ks for l in ks], dtype=float)
            f_grid = (pk[:, None] * pk[None, :]).ravel()
            idx = rng.integers(0, len(grid), size=n_cand)
            xy = grid[idx]
            f = f_grid[idx]
        elif dist == '泊松':
            lam = float(p[0])
            kmax = int(round(lam + 6 * np.sqrt(max(lam, 0.01)) + 5))
            ks = np.arange(kmax + 1)
            pk = stats.poisson.pmf(ks, lam)
            grid = np.array([[k, l] for k in ks for l in ks], dtype=float)
            f_grid = (pk[:, None] * pk[None, :]).ravel()
            idx = rng.integers(0, len(grid), size=n_cand)
            xy = grid[idx]
            f = f_grid[idx]
        else:
            raise ValueError("未知分布")
        return xy, f

    def _final_points(self):
        """对最终分布重新采样，返回点集 (xy, f_p, z_p)，
        点分布服从最终分布的概率密度（高密度区域点更多）。"""
        n = len(self.sample_xy)
        rng = np.random.default_rng(7)
        n_cand = max(n * 10, 5000)
        xy_cand, f_init = self._uniform_candidates(n_cand, rng)

        xy_p = xy_cand
        det_prod = np.ones(n_cand)
        for tr in self.transforms:
            xy_p, det = apply_transform(tr, xy_p, 1.0)
            det_prod = det_prod * det

        if self.kind == 'discrete':
            f_p = f_init
        else:
            f_p = f_init / np.maximum(det_prod, 1e-3)

        # 按最终密度的 final_weight 次方加权，权重越大高密度区域点越多
        w = f_p ** self.final_weight
        w = w / w.sum()
        idx = rng.choice(n_cand, size=n, p=w)
        xy = xy_p[idx]
        f = f_p[idx]
        z = rng.uniform(0.0, f)
        return xy, f, z

    def _plot_final_points(self, ax, xy, f_p, z_p):
        """在指定 ax 上绘制最终分布散点云（3D）"""
        f_valid = f_p[np.isfinite(f_p)]
        z_valid = z_p[np.isfinite(z_p)]
        z_max = (float(np.percentile(z_valid, 99)) if len(z_valid) else 1.0)
        norm = Normalize(vmin=0, vmax=max(z_max, 1e-9))
        sc = ax.scatter(xy[:, 0], xy[:, 1], z_p,
                        c=f_p, cmap=self.cmap_var.get(), norm=norm,
                        s=self.point_size, depthshade=True, alpha=0.85,
                        rasterized=True)

        xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
        ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
        span = max(xmax - xmin, ymax - ymin, 0.5)
        pad = 0.1 * span
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_zlim(0, max(z_max * 1.15, 1e-3))
        ax.set_xlabel('x', fontsize=12, labelpad=6, color='black')
        ax.set_ylabel('y', fontsize=12, labelpad=6, color='black')
        ax.set_zlabel('p(x, y)', fontsize=12, labelpad=8, color='black')
        ax.tick_params(colors='black')
        dist = self.dist_var.get()
        ax.set_title(f'{dist}分布 最终概率密度分布（3D 散点）',
                     fontsize=13, pad=12, color='black')

        # 在图上增加文本框展示变换过程（放顶部，不遮挡坐标轴内容）
        if self.transforms:
            steps = []
            for i, tr in enumerate(self.transforms):
                steps.append(
                    f"$x'={expr_to_math(tr.fx_expr)},"
                    f"\\,y'={expr_to_math(tr.fy_expr)}$")
            chain = r' $\rightarrow$ '.join(steps)
            ax.text2D(0.5, 0.98, f'变换过程：{chain}',
                      transform=ax.transAxes, fontsize=11, color='black',
                      va='top', ha='center',
                      bbox=dict(boxstyle='round,pad=0.4',
                                facecolor='white', edgecolor='#cccccc',
                                alpha=0.9))
        return sc

    def _draw_final_distribution(self):
        xy, f_p, z_p = self._final_points()
        self.ax.clear()
        self._white_background()
        self._plot_final_points(self.ax, xy, f_p, z_p)

    def toggle_final_view(self):
        if self.view_mode == 'animation':
            if self.anim is not None and not self.paused:
                self.anim.event_source.stop()
                self.paused = True
                self.pause_btn.config(text="继续")
            self.view_mode = 'final'
            self.final_btn.config(text="返回动画")
            self.precompute()
            self._draw_final_distribution()
            self.canvas.draw_idle()
        else:
            self.view_mode = 'animation'
            self.final_btn.config(text="查看最终分布")
            self.update(self.current_frame)
            self.canvas.draw_idle()

    def reset_view(self):
        self.user_limits = None
        self.ax.view_init(elev=ELEV, azim=AZIM)
        self.xticks = self._default_xticks.copy()
        self.yticks = self._default_yticks.copy()
        self.zticks = self._default_zticks.copy()
        if self.view_mode == 'final':
            self._draw_final_distribution()
        else:
            self.update(self.current_frame)
        self.canvas.draw_idle()

    def update(self, frame):
        if self.view_mode == 'final':
            self._draw_final_distribution()
            return
        self.current_frame = frame
        self._syncing_progress = True
        try:
            if hasattr(self, 'progress_scale'):
                self.progress_scale.set(frame)
                total = getattr(self, 'total_frames', 1)
                self.progress_label.config(text=f"{int(frame)}/{total - 1}")
        finally:
            self._syncing_progress = False

        self.ax.clear()
        self._white_background()

        n = len(self.transforms)
        if n == 0:
            round_idx = 0
            s = 0.0
        else:
            round_idx = min(frame // FRAMES_PER_ROUND, n - 1)
            s = smoothstep((frame % FRAMES_PER_ROUND) / (FRAMES_PER_ROUND - 1))

        # 用缓存的状态：第 round_idx 轮起点 = 前 round_idx 轮复合结果
        xy = self.state_xy[round_idx]
        det_prod = self.state_det[round_idx]
        if round_idx < n:
            xy, det = apply_transform(self.transforms[round_idx], xy, s)
            det_prod = det_prod * det

        z_p = self._draw_points(xy, det_prod)

        # Z 轴动态自适应：用 99% 分位数（避免发散点主导，颜色分布合理）
        z_p_valid = z_p[np.isfinite(z_p)]
        if len(z_p_valid):
            z_max = float(np.max(z_p_valid))
        else:
            z_max = 1.0
        self.zticks = self._nice_ticks(0.0, max(z_max, 1e-3), 5)

        # 坐标轴：优先使用用户缩放/平移后的范围，否则用默认
        if self.user_limits is not None:
            xlim, ylim, zlim = self.user_limits
        else:
            xlim = self.xlim
            ylim = self.ylim
            zlim = (0, self.zticks[-1])
        # 先设置刻度与标签（可能触发 autoscale），最后再设置范围覆盖
        self.ax.set_xticks(self.xticks)
        self.ax.set_yticks(self.yticks)
        self.ax.set_zticks(self.zticks)
        self.ax.set_xticklabels([f'{v:g}' for v in self.xticks])
        self.ax.set_yticklabels([f'{v:g}' for v in self.yticks])
        self.ax.set_zticklabels([f'{v:g}' for v in self.zticks])
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_zlim(*zlim)
        self.ax.set_xlabel('x', fontsize=12, labelpad=6, color='black')
        self.ax.set_ylabel('y', fontsize=12, labelpad=6, color='black')
        self.ax.set_zlabel('p(x, y)', fontsize=12, labelpad=8, color='black')
        self.ax.tick_params(colors='black')

        dist = self.dist_var.get()
        if n > 0:
            tr = self.transforms[round_idx]
            tr_desc = (f"$x'={expr_to_math(tr.fx_expr)},"
                       f"\\;y'={expr_to_math(tr.fy_expr)}$")
        else:
            tr_desc = '-'
        title = (f'{dist}分布   第 {round_idx + 1}/{n} 轮 {tr_desc}\n'
                 f'峰值密度 = {z_max:.4f}   总概率 = 1.0000')
        self.ax.set_title(title, fontsize=12, pad=12, color='black')

    # ---------------- 绘制散点 ----------------
    def _draw_points(self, xy_p, det_prod):
        f = self.sample_f
        z = self.sample_z
        if self.kind == 'discrete':
            f_p = f
            z_p = z
        else:
            f_p = f / det_prod
            z_p = z / det_prod
        f_p = np.nan_to_num(f_p, nan=0.0, posinf=1e6, neginf=0.0)
        z_p = np.nan_to_num(z_p, nan=0.0, posinf=1e6, neginf=0.0)

        # 归一化密度（0~1），使颜色柱固定，仅刻度数值随当前帧变化
        f_valid = f_p[np.isfinite(f_p)]
        f_max = float(np.max(f_valid)) if len(f_valid) else 1.0
        f_max = max(f_max, 1e-9)
        f_norm = f_p / f_max
        norm = Normalize(vmin=0.0, vmax=1.0)
        if hasattr(self, 'mappable'):
            self.mappable.set_norm(norm)
            tick_pos = np.linspace(0.0, 1.0, 6)
            tick_vals = tick_pos * f_max
            self.cbar.set_ticks(tick_pos)
            self.cbar.set_ticklabels([f'{v:g}' for v in tick_vals])

        sc = self.ax.scatter(xy_p[:, 0], xy_p[:, 1], z_p,
                             c=f_norm, cmap=self.cmap_var.get(),
                             norm=norm,
                             s=self.point_size, depthshade=True, alpha=0.85,
                             rasterized=True)
        return z_p

    def _white_background(self):
        self.fig.patch.set_facecolor('white')
        self.ax.set_facecolor('white')
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        for ax in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            ax._axinfo['grid']['color'] = (0.85, 0.85, 0.85, 1.0)
            ax._axinfo['axisline']['color'] = (0, 0, 0, 1)


def main():
    root = tk.Tk()
    TransformApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
