import tkinter as tk
print("1: tkinter import edildi")
root = tk.Tk()
print("2: root yaratıldı")
root.withdraw()
print("3: root gizlendi")

from tkinter import filedialog
print("4: filedialog import edildi")

path = filedialog.askopenfilename(title="Test")
print(f"5: pencere kapatıldı, path={path}")

root.destroy()
print("6: bitti")