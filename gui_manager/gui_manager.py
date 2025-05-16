import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

from solution.solution import (
    make_grid, improved_euler_until_eps, runge_kutta4,
    rk4_until_eps, milne, milne_until_eps
)

FUNCTIONS = {
    "y' = y + x":      lambda x, y: y + x,
    "y' = y - x":      lambda x, y: y - x,
    "y' = x * y":      lambda x, y: x * y,
    "y' = y^2 + x":    lambda x, y: y ** 2 + x,
    "y' = sin(x) + y": lambda x, y: np.sin(x) + y,
}

EXACT = {
    "y' = y + x":      lambda x, y0, x0: np.exp(x - x0) * (y0 + x0 + 1) - x - 1,
    "y' = y - x":      lambda x, y0, x0: np.exp(x - x0) * (y0 - x0 + 1) + x - 1,
    "y' = x * y":      lambda x, y0, x0: y0 * np.exp((x ** 2 - x0 ** 2) / 2),
    "y' = y^2 + x":    lambda x, y0, x0: np.nan if (2*y0 + 1 - 2*x + 2*x0) < 0 else (np.sqrt(2*y0 + 1 - 2*x + 2*x0) - 1)/2,
    "y' = sin(x) + y": lambda x, y0, x0: (y0 + np.cos(x0) + np.sin(x0)) * np.exp(x - x0) - np.sin(x) - np.cos(x),
}


def solve_and_show():
    try:
        y0, x0, xn, h, eps = map(float, (e_y0.get(), e_x0.get(),
                                         e_xn.get(), e_h.get(), e_eps.get()))
    except ValueError:
        messagebox.showerror("Ошибка", "Неверный ввод")
        return

    fn = cmb.get()
    if fn not in FUNCTIONS or x0 >= xn or h <= 0 or eps <= 0:
        messagebox.showerror("Ошибка", "Проверьте входные данные")
        return

    f = FUNCTIONS[fn]
    exact_fn = EXACT[fn]

    xs = make_grid(x0, xn, h)
    y_exact = [exact_fn(x, y0, x0) for x in xs]

    xs_eul, y_eul, err_e = improved_euler_until_eps(f, x0, xn, y0, h, eps)
    xs_rk, y_rk, err_rk = rk4_until_eps(f, x0, xn, y0, h, eps)
    y_m = milne(f, xs, y0, eps)
    err_m = [abs(ye - ym) if np.isfinite(ye) else np.nan for ye, ym in zip(y_exact, y_m)]

    def safe(val): return float(val) if np.isfinite(val) else float('nan')

    txt.delete("1.0", tk.END)
    header = f"{'x':<7}{'y*':>11}{'Euler':>11}{'errE':>9}"\
             f"{'RK4':>11}{'errRK':>9}{'Milne':>11}{'errM':>9}\n"
    txt.insert(tk.END, header)
    for i, x in enumerate(xs):
        y_ex = safe(y_exact[i])
        yeul = safe(y_eul[i])
        e_eul = safe(err_e[i])
        yrk = safe(y_rk[i]) if i < len(y_rk) else np.nan
        e_rk = safe(err_rk[i]) if i < len(err_rk) else np.nan
        ym = safe(y_m[i])
        em = safe(err_m[i])
        txt.insert(tk.END, f"{x:<7.3f}{y_ex:>11.3f}{yeul:>11.3f}{e_eul:>9.3f}"
                           f"{yrk:>11.3f}{e_rk:>9.3f}{ym:>11.3f}{em:>9.3f}\n")

    for w in plot_frm.winfo_children():
        w.destroy()
    root.update_idletasks()

    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.grid(True)
    ax.plot(xs,     y_exact, 'k--', lw=2, label='Exact')
    ax.plot(xs_eul, y_eul,   'r-',  lw=2, label='Euler')
    ax.plot(xs_rk,  y_rk,    'g--', lw=1.5, label='RK4')
    ax.plot(xs,     y_m,     'b-.', lw=1.5, label='Milne')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend()
    fig.tight_layout()

    cvs = FigureCanvasTkAgg(fig, plot_frm)
    cvs.draw(); cvs.get_tk_widget().pack(expand=True, fill='both')
    NavigationToolbar2Tk(cvs, plot_frm).update()


root = tk.Tk()
root.title("Численные методы")
root.geometry("1100x760")

frm = tk.Frame(root); frm.pack(pady=8, fill='x')
tk.Label(frm, text="Уравнение:").grid(row=0, column=0, sticky='e')
cmb = ttk.Combobox(frm, width=26, state="readonly", values=list(FUNCTIONS.keys()))
cmb.grid(row=0, column=1, sticky='w'); cmb.current(0)

labels = ("y0", "x0", "xn", "h", "ε")
ents = []
for i, l in enumerate(labels, 1):
    tk.Label(frm, text=l).grid(row=i, column=0, sticky='e')
    e = tk.Entry(frm, width=10); e.grid(row=i, column=1, sticky='w')
    ents.append(e)

e_y0, e_x0, e_xn, e_h, e_eps = ents
for e, v in zip(ents, ("1", "0", "1", "0.1", "1e-4")):
    e.insert(0, v)

tk.Button(frm, text="Solve", command=solve_and_show).grid(row=6, column=1, pady=6, sticky='w')

txt = tk.Text(root, height=15, width=120, wrap='none'); txt.pack(fill='x')
plot_frm = tk.Frame(root); plot_frm.pack(expand=True, fill='both')


def start_gui_plot(xs, y_exact, xs_eul, y_eul, xs_rk, y_rk, xs_milne, y_milne):
    for w in plot_frm.winfo_children():
        w.destroy()

    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.grid(True)
    ax.plot(xs,      y_exact,  'k--', lw=2, label='Exact')
    ax.plot(xs_eul,  y_eul,    'r-',  lw=2, label='Euler')
    ax.plot(xs_rk,   y_rk,     'g--', lw=1.5, label='RK4')
    ax.plot(xs_milne, y_milne, 'b-.', lw=1.5, label='Milne')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend()
    fig.tight_layout()

    cvs = FigureCanvasTkAgg(fig, plot_frm)
    cvs.draw(); cvs.get_tk_widget().pack(expand=True, fill='both')
    NavigationToolbar2Tk(cvs, plot_frm).update()



def start_gui():
    root.mainloop()