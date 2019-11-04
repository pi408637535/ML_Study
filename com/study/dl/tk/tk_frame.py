import tkinter as tk


def hit_me(var):
    global on_hit
    if on_hit == False:
        var.set("you hit me..")
        on_hit = True
    else:
        var.set("....")
        on_hit = False



if __name__ == '__main__':
    window = tk.Tk()  #

    on_hit = False

    window.geometry('500x300')
    frame_root = tk.Frame(window)
    frame_root.pack()
    frame_l = tk.Frame(frame_root)
    frame_r = tk.Frame(frame_root)


    frame_l.pack(side="left")
    frame_r.pack(side="right")

    tk.Label(frame_l, text="语音识别").grid(row=0,column=0)
    tk.Text(frame_l).grid(row=1,column=0)
    tk.Button(frame_l,text="选择文件").grid(row=1,column=1)

    tk.Label(frame_r, text="语音合成").grid(row=0, column=0)
    tk.Text(frame_r).grid(row=1, column=0)
    tk.Button(frame_r, text="开始合成").grid(row=1, column=1)


    window.mainloop()
