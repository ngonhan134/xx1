from pathlib import Path
from datetime import date
from tkinter import *
import tkinter as tk                # python 3
from tkinter import font as tkfont
from tkinter import messagebox,PhotoImage
from Predict import prediction
import threading
from ControlDoor import Detected_Object
from ControlDoor import on_led
from ControlDoor import off_led
import ControlDoor
import os
import time
from create_classifier import train_classifer
from create_dataset import start_capture
import glob
import shutil

# default_password="1111"

def save_password(password):
    with open("password.txt", "w") as file:
        file.write(password)

def load_password():
    try:
        with open("password.txt", "r") as file:
            return file.read()
    except FileNotFoundError:
        return ""


class MainUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # self.resizable(False, False)


        self.geometry("594x365")
        self.overrideredirect(True) 
        photo = PhotoImage(file = 'hand.png')
        self.iconphoto(False, photo)
        # self.title("PalmPrint Recognizer")
        # self.title_font = tkfont.Font(family='Cursive', size=18, weight="bold", slant="italic")
        # self.configure(bg="#FFFFFF")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, minsize=500, weight=1)
        container.grid_columnconfigure(0, minsize=866, weight=1)
        container.grid_columnconfigure(1, weight=1)

        self.frames = {}
        for F in (StartPage,EndPage,PutHand,Train,Password,Failed,ChangPassword):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")
        # password_manager = PasswordManager(parent=container, controller=self)  # Tạo đối tượng PasswordManager
        # password_frame = Password(parent=container, controller=self, password_manager=password_manager)
        # self.frames["Password"] = password_frame

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        if page_name == "StartPage":
            self.frames["StartPage"].show_buttons()
            self.frames["Train"].hide_buttons()
            self.frames["Password"].hide_buttons()
            self.frames["ChangPassword"].hide_buttons()
        elif page_name == "PutHand":
            self.frames["StartPage"].hide_buttons()
            self.frames["Password"].hide_buttons()
        elif page_name == "StartPage":
            self.frames["StartPage"].show_buttons()
            self.frames["Train"].hide_buttons()
            self.frames["Password"].hide_buttons()
        elif page_name == "EndPage":
            self.frames["Password"].hide_buttons()
            
        elif page_name == "Train":
            self.frames["StartPage"].hide_buttons()
            self.frames["Train"].show_buttons()
            self.frames["Password"].hide_buttons()
        elif page_name == "Password":
            self.frames["StartPage"].hide_buttons()
            self.frames["Password"].show_buttons()
            self.frames["Train"].hide_buttons()
            self.frames["ChangPassword"].hide_buttons()
        elif page_name == "ChangPassword":
            self.frames["StartPage"].hide_buttons()
            self.frames["Password"].hide_buttons()
            self.frames["Train"].hide_buttons()
            self.frames["ChangPassword"].show_buttons()



        frame = self.frames[page_name]
        frame.tkraise()

    
        # frame = self.frames[page_name]
        # frame.tkraise()

    def on_closing(self):
        #if messagebox.askokcancel("Quit", "Are you sure?"):
        self.destroy()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.canvas = Canvas(
            self,
            bg="#FFFFFF",
            height=372,
            width=594,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        self.image_image_1 = PhotoImage(file=self.relative_to_assets("image_1.png"))
        self.image_1 = self.canvas.create_image(304.0, 186.0, image=self.image_image_1)

        self.image_image_2 = PhotoImage(file=self.relative_to_assets("image_2.png"))
        self.image_2 = self.canvas.create_image(80.0, 306.0, image=self.image_image_2)

        self.button_image_1 = PhotoImage(file=self.relative_to_assets("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Password") ,
            relief="flat"
        )
        # self.button_1.place(x=291.0, y=183.0, width=140.428466796875, height=53.970947265625)

        self.button_login = PhotoImage(file=self.relative_to_assets("button_2.png"))
        self.button_2 = Button(
            image=self.button_login,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("PutHand") or  self.start_detection(),
            relief="flat"
        )
        # self.button_2.place(x=104.0, y=184.0, width=140.428466796875, height=53.970947265625)

        self.button_image_3 = PhotoImage(file=self.relative_to_assets("button_3.png"))
        self.button_3 = Button(
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda:self.delete_files_button_clicked(),
            relief="flat"
        )
        # self.button_3.place(x=291.0, y=81.0, width=140.428466796875, height=53.970947265625)

        self.button_image_4 = PhotoImage(file=self.relative_to_assets("button_4.png"))
        self.button_4 = Button(
            image=self.button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Train"),
            relief="flat"
        )
        # self.button_4.place(x=104.0, y=81.0, width=140.428466796875, height=53.970947265625) and self.button_4.config(state="disabled")

        self.button_image_5 = PhotoImage(file=self.relative_to_assets("button_5.png"))
        self.button_5 = Button(
            image=self.button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=self.on_closing,
            relief="flat"
        )
        # self.button_5.place(x=229.0, y=267.0, width=73.6923828125, height=32.0)
        self.buttons_hidden = False

        #thread
        self.thread1 = None
        self.thread2 = None
        self.thread3 = None
        self.stop_thread3 = False
        self.stop_thread1 = False
        self.stop_thread2=False
        self.last_detection_time=0
        # self.start_detection()
    def start_detection(self):
        self.thread1 = threading.Thread(target=self.run_detection)
        self.thread1.start()
        print("Thread 1 started")


    def stop_detection(self):
        if self.thread1 and self.thread1.is_alive():
            self.thread1.join()
        print("Thread 1 stopped")


    def run_detection(self):
        if Detected_Object():
            off_led()
            if prediction():
                self.controller.show_frame("EndPage")
                self.start_door_control()

            else:
                self.controller.show_frame("Failed")
                time.sleep(3)
                self.controller.show_frame("StartPage")
        else:
            self.controller.show_frame("StartPage")
        on_led()


    # def start_prediction(self):
    #     self.stop_thread2=False
    #     self.thread2 = threading.Thread(target=self.run_prediction)
    #     self.thread2.start()
    #     print("Thread 2 started")

    # def stop_prediction(self):
    #     # if self.thread2 and self.thread2.is_alive():
    #     #     self.thread2.join()
    #     self.stop_thread2=True
        # print("Thread 2 stopped")
    # def run_prediction(self):
    #     # self.stop_thread2=False
    #     if prediction():
    #         self.controller.show_frame("EndPage")
    #         self.start_door_control()

    #     else:
    #         self.controller.show_frame("Failed")
    #         time.sleep(3)
    #         self.controller.show_frame("StartPage")



    def start_door_control(self):
        self.stop_thread3 = False
        self.thread3 = threading.Thread(target=self.run_door_control)
        self.thread3.start()
        print("Thread 3 started")
    def stop_door_control(self):
        self.stop_thread3 = True
        print("Thread 3 stopped")

    def run_door_control(self):
        ControlDoor.open_door()
        while not self.stop_thread3:
            if  ControlDoor.GetDistance() > 35 :
                ControlDoor.close_door()
                self.stop_door_control()
                self.controller.show_frame("StartPage")
            time.sleep(1)  # Thực hiện kiểm tra lại sau một khoảng thời gian

    def start_threads(self):
        self.start_detection()
        # self.start_prediction()
        self.start_door_control()

    def stop_threads(self):
        self.stop_detection()
        # self.stop_prediction()
        self.stop_door_control()

    def hide_buttons(self):
        if not self.buttons_hidden:
            self.button_1.place_forget()
            self.button_2.place_forget()
            self.button_3.place_forget()
            self.button_4.place_forget()
            self.button_5.place_forget()
            self.buttons_hidden = True
            self.stop_threads()
        # else:
        #     data_path = './data1/user'
        #     subfolders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
        #     if subfolders:
        #         self.button_4.place_forget()
        #     # else:
        #     #     self.button_4.place(x=104.0, y=81.0, width=140.428466796875, height=53.970947265625)

    def show_buttons(self):
        # if self.buttons_hidden:
            self.button_1.place(x=291.0, y=183.0, width=140.428466796875, height=53.970947265625)
            
            self.button_3.place(x=291.0, y=81.0, width=140.428466796875, height=53.970947265625)
            data_path = './data1/user'
            subfolders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
            if subfolders:
                self.button_4.place(x=104.0, y=81.0, width=140.428466796875, height=53.970947265625) or  self.button_4.config(state="disabled") 
                self.button_3.place(x=291.0, y=81.0, width=140.428466796875, height=53.970947265625) or self.button_3.config(state="normal") 
                self.button_2.place(x=104.0, y=184.0, width=140.428466796875, height=53.970947265625) or self.button_2.config(state="normal") 
            else:
                self.button_4.place(x=104.0, y=81.0, width=140.428466796875, height=53.970947265625) or self.button_4.config(state="normal")
                self.button_3.place(x=291.0, y=81.0, width=140.428466796875, height=53.970947265625) or  self.button_3.config(state="disabled") 
                self.button_2.place(x=104.0, y=184.0, width=140.428466796875, height=53.970947265625) or self.button_2.config(state="disabled")

            self.button_5.place(x=229.0, y=267.0, width=73.6923828125, height=32.0)
            self.buttons_hidden = False



    def delete_files_button_clicked(self):

        confirm = messagebox.askyesno("Confirmation", "Are you sure delete all files?")
        if confirm:
            directory = "./data1/user"

            def delete_folders_in_directory(directory):
                folder_list = glob.glob(str(Path(directory) / "*"))
                for folder_path in folder_list:
                    if Path(folder_path).is_dir():
                        shutil.rmtree(folder_path)  # Xóa thư mục và nội dung bên trong

            delete_folders_in_directory(directory)
            messagebox.showinfo("Success", "All files deleted!")

            self.controller.show_frame("StartPage")

    def destroy(self):
        super().destroy()

    def on_closing(self):
        self.controller.on_closing()
    @staticmethod
    def relative_to_assets(path: str) -> Path:
        OUTPUT_PATH = Path(__file__).parent
        ASSETS_PATH = OUTPUT_PATH / Path("./assets/frame4")
        return ASSETS_PATH / Path(path)   
    
import tkinter as tk
from PIL import Image, ImageTk

class PutHand(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.canvas = tk.Canvas(
            self,
            bg="#FBFBFB",
            height=368,
            width=594,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        gif_file = "./assets/frame3/puthand.gif"
        gif = Image.open(gif_file)

        self.frames = []
        for frame in range(gif.n_frames):
                gif.seek(frame)
                frame_image = ImageTk.PhotoImage(gif.copy())
                self.frames.append(frame_image)

        self.current_frame = 0

        self.gif_animation = None

        self.update_gif_frame()

    def update_gif_frame(self):
        image = self.frames[self.current_frame]
        if self.gif_animation:
            self.canvas.delete(self.gif_animation)
        self.gif_animation = self.canvas.create_image(304.0, 184.0, image=image)

        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self.after(20, self.update_gif_frame)

    @staticmethod
    def relative_to_assets(path: str) -> Path:
        OUTPUT_PATH = Path(__file__).parent
        ASSETS_PATH = OUTPUT_PATH / Path("./assets/frame3")
        return ASSETS_PATH / Path(path)



class Train(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller  
        from pathlib import Path
        from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
        
        from tkinter import ttk
     
        self.canvas = Canvas(
            self,
            bg = "#FBFBFB",
            height = 368,
            width = 594,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )

        self.canvas.place(x = 0, y = 0)
        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets("image_1.png"))
        self.image_1 = self.canvas.create_image(
            304.0,
            184.0,
            image=self.image_image_1
        )

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=self.start_capture_images,
            relief="flat"
        )
        self.button_1.place(
            x=63.0,
            y=50.0,
            width=217.0,
            height=61.0
        )
        self.button_image_2 = PhotoImage(
            file=self.relative_to_assets("button_2.png"))
        self.button_2 = Button(
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=self.start_train,
            relief="flat"
        )
        self.button_2.place(
            x=69.0,
            y=140.0,
            width=205.0,
            height=56.0
        )

        self.button_image_3 = PhotoImage(
            file=self.relative_to_assets("button_3.png"))
        self.button_3 = Button(
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("StartPage"),
            relief="flat"
        )
        self.button_3.place(
            x=117.0,
            y=225.0,
            width=122.0,
            height=49.0
        )
        self.buttons_hidden = False
        self.thread4 = None
        self.thread5 = None
        self.pb = None  # Thanh progress bar
        self.value_label = None  # Nhãn tiến trình
        self.stop_thread4 = False
        self.stop_thread5 =False

    def start_scan(self):
        # ...
        self.pb = ttk.Progressbar(
            self,
            orient='horizontal',
            mode='determinate',
            length=280
        )
        # Đặt vị trí của thanh progress bar
        self.pb.place(    x=4.0,
                            y=298.0,
                            width=156.0,
                            height=22.0)

        # Nhãn tiến trình
        self.value_label = ttk.Label(self, text="Capture Progress: 0%")
        self.value_label.place(      x=41.0,
                                        y=330.0,
                                        # width=83.0,
                                        # height=11.0)
        )
        self.percentage = 0
        self.load_bar()
    
    def capture_images(self):
        messagebox.showinfo("INSTRUCTIONS", "We will Capture 100 pic of your Palm.")
        # data_path = './data1/user'
        # subfolders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
        # if subfolders:
        #     self.button_1.config(state="normal")
            
        #     root = self.winfo_toplevel()
        #     root.update()
        #     self.controller.show_frame("Train")
            
        # else:
        #     self.button_1.config(state="disabled")
            
        #     root = self.winfo_toplevel()
            
        #     root.update()
        #     self.controller.show_frame("Train")
        self.start_scan()
        
        num_of_images = start_capture()  # Gọi hàm start_capture() từ file khác
        self.load_bar()
        if num_of_images > 100:
            # Hiển thị thông báo hoàn thành khi num_of_images > 100
            messagebox.showinfo("Success", "Capture completed!")
            
        else:
            # Hiển thị thông báo lỗi nếu num_of_images không đạt yêu cầu
            messagebox.showerror("Error", "Capture at least 100 images!")
        # self.button_1.config(state="disabled")
        
        

    def load_bar(self):
        self.percentage += 5  # Giá trị tiến trình tăng 10% mỗi lần
        self.pb['value'] = self.percentage
        self.value_label['text'] = f"Scanning Progress: {self.percentage}%"
        if self.percentage < 100:
            # Tiếp tục cập nhật tiến trình nếu chưa hoàn thành
            self.after(1000, self.load_bar)

    def start_capture_images(self):
        self.stop_thread4 = False
        self.thread4 = threading.Thread(target=self.capture_images)
        self.thread4.start()
        print("Thread 4 started")
        self.update_progress_bar_thread = threading.Thread(target=self.update_progress_bar)
        self.update_progress_bar_thread.start()

    def update_progress_bar(self):
        while self.stop_thread4 :
            # Cập nhật thanh progress bar song song với quá trình start_capture()
            self.load_bar()

 
    def stop_capture_images(self):
        self.stop_thread4 = True
        print("Thread 4 stopped")

    def start_threads(self):
        self.start_capture_images()
        self.start_train()


    def stop_threads(self):
        self.stop_capture_images()
        self.stop_train()
        # self.stop_door_control()
    
    
    def start_scan1(self):
        # ...
        self.pb = ttk.Progressbar(
            self,
            orient='horizontal',
            mode='determinate',
            length=280
        )
        # Đặt vị trí của thanh progress bar
        self.pb.place(     x=209.0,
                        y=298.0,
                        width=156.0,
                        height=22.0)

        # Nhãn tiến trình
        self.value_label = ttk.Label(self, text="Train Progress: 0%")
        self.value_label.place(      x=258.0,
                            y=330.0,
                                        # width=83.0,
                                        # height=11.0)
                )
        self.percentage = 0
        self.load_bar1()
    def load_bar1(self):
        self.percentage += 0.2  # Giá trị tiến trình tăng 10% mỗi lần
        self.pb['value'] = self.percentage
        self.value_label['text'] = f"Train Progress: {self.percentage}%"
        if self.percentage < 100:
            # Tiếp tục cập nhật tiến trình nếu chưa hoàn thành
            self.after(1000, self.load_bar1)
    def update_progress_bar1(self):
        while self.stop_thread5 :
            # Cập nhật thanh progress bar song song với quá trình train()
            self.load_bar1()
    def trainmodel(self):
        data_path = './data1/user/1'
        num_of_images = len(glob.glob(os.path.join(data_path, '*.bmp')))
        if num_of_images < 101:
            messagebox.showerror("ERROR", "No enough Data, Capture at least 100 images!")
            return
        self.start_scan1()
        self.load_bar1()
        train_classifer()
        messagebox.showinfo("SUCCESS", "The modele has been successfully trained!")
        self.controller.show_frame("StartPage")

    def stop_train(self):
        self.stop_thread5 = True
        print("Thread 5 stopped")
    
    def start_train(self):
        self.stop_thread5 = False
        self.thread5 = threading.Thread(target=self.trainmodel)
        self.thread5.start()
        print("Thread 5 started")
        self.update_progress_bar_thread1 = threading.Thread(target=self.update_progress_bar1)
        self.update_progress_bar_thread1.start()

    def hide_buttons(self):
        if not self.buttons_hidden:
            self.button_1.place_forget()
            self.button_2.place_forget()
            self.button_3.place_forget()
            self.buttons_hidden = True
            # self.stop_threads()

    def show_buttons(self):
        if self.buttons_hidden:
            self.button_1.place(    x=63.0,
                                    y=50.0,
                                    width=217.0,
                                    height=61.0)
            self.button_2.place(    x=69.0,
                                    y=140.0,
                                    width=205.0,
                                    height=56.0)
            self.button_3.place(    x=117.0,
                                    y=225.0,
                                    width=122.0,
                                    height=49.0
                                )
            self.buttons_hidden = False




    @staticmethod
    def relative_to_assets(path: str) -> Path:
        OUTPUT_PATH = Path(__file__).parent
        ASSETS_PATH = OUTPUT_PATH / Path("./assets/frame1")
        return ASSETS_PATH / Path(path)

class EndPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.canvas = Canvas(
            self,
            bg="#FFFFFF",
            height=368,
            width=594,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        # self.image_image_1 = PhotoImage(file=self.relative_to_assets("image_1.png"))
        # self.image_1 = self.canvas.create_image(297.0, 184.0, image=self.image_image_1)
        gif_file = "./assets/frame0/verify2.gif"
        gif = Image.open(gif_file)

        self.frames = []
        for frame in range(gif.n_frames):
                gif.seek(frame)
                frame_image = ImageTk.PhotoImage(gif.copy())
                self.frames.append(frame_image)

        self.current_frame = 0

        self.gif_animation = None

        self.update_gif_frame()

    def update_gif_frame(self):
        image = self.frames[self.current_frame]
        if self.gif_animation:
            self.canvas.delete(self.gif_animation)
        self.gif_animation = self.canvas.create_image(304.0, 184.0, image=image)

        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self.after(10, self.update_gif_frame)




    @staticmethod
    def relative_to_assets(path: str) -> Path:
        OUTPUT_PATH = Path(__file__).parent
        ASSETS_PATH = OUTPUT_PATH / Path("./assets/frame0")
        return ASSETS_PATH / Path(path)
class Failed(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.canvas = Canvas(
            self,
            bg = "#FFFFFF",
            height = 372,
            width = 594,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )

        self.canvas.place(x = 0, y = 0)
        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets("image_1.png"))
        self.image_1 = self.canvas.create_image(
            309.0,
            190.0,
            image=self.image_image_1
)
    @staticmethod
    def relative_to_assets(path: str) -> Path:
        OUTPUT_PATH = Path(__file__).parent
        ASSETS_PATH = OUTPUT_PATH / Path("./assets/frame5")
        return ASSETS_PATH / Path(path)
    
class Password(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        self.canvas = Canvas(
            self,
            bg = "#FFFFFF",
            height = 368,
            width = 594,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )

        self.canvas.place(x = 0, y = 0)
        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets("image_1.png"))
        self.image_1 = self.canvas.create_image(
            297.0,
            184.0,
            image=self.image_image_1
        )

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_click(0),
            relief="flat"
        )
        self.button_1.place(
            x=56.0,
            y=271.0,
            width=62.0,
            height=55.0
        )

        self.button_image_2 = PhotoImage(
            file=self.relative_to_assets("button_2.png"))
        self.button_2 = Button(
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_click(1),
            relief="flat"
        )
        self.button_2.place(
            x=51.0,
            y=48.0,
            width=67.0,
            height=64.0
        )

        self.button_image_3 = PhotoImage(
            file=self.relative_to_assets("button_3.png"))
        self.button_3 = Button(
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_click(2),
            relief="flat"
        )
        self.button_3.place(
            x=127.0,
            y=49.0,
            width=67.0,
            height=64.0
        )

        self.button_image_4 = PhotoImage(
            file=self.relative_to_assets("button_4.png"))
        self.button_4 = Button(
            image=self.button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_click(3),
            relief="flat"
        )
        self.button_4.place(
            x=203.0,
            y=50.0,
            width=67.0,
            height=64.0
        )

        self.button_image_5 = PhotoImage(
            file=self.relative_to_assets("button_5.png"))
        self.button_5 = Button(
            image=self.button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_click(7),
            relief="flat"
        )
        self.button_5.place(
            x=51.0,
            y=194.0,
            width=67.0,
            height=64.0
        )

        self.button_image_6 = PhotoImage(
            file=self.relative_to_assets("button_6.png"))
        self.button_6 = Button(
            image=self.button_image_6,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_click(8),
            relief="flat"
        )
        self.button_6.place(
            x=127.0,
            y=194.0,
            width=67.0,
            height=64.0
        )

        self.button_image_7 = PhotoImage(
            file=self.relative_to_assets("button_7.png"))
        self.button_7 = Button(
            image=self.button_image_7,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_click(9),
            relief="flat"
        )
        self.button_7.place(
            x=203.0,
            y=195.0,
            width=67.0,
            height=64.0
        )

        self.button_image_8 = PhotoImage(
            file=self.relative_to_assets("button_8.png"))
        self.button_8 = Button(
            image=self.button_image_8,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_click(4),
            relief="flat"
        )
        self.button_8.place(
            x=51.0,
            y=121.0,
            width=67.0,
            height=64.0
        )

        self.button_image_9 = PhotoImage(
            file=self.relative_to_assets("button_9.png"))
        self.button_9 = Button(
            image=self.button_image_9,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_click(5),
            relief="flat"
        )
        self.button_9.place(
            x=127.0,
            y=121.0,
            width=67.0,
            height=64.0
        )

        self.button_image_10 = PhotoImage(
            file=self.relative_to_assets("button_10.png"))
        self.button_10 = Button(
            image=self.button_image_10,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_click(6),
            relief="flat"
        )
        self.button_10.place(
            x=203.0,
            y=122.0,
            width=67.0,
            height=64.0
        )

        self.button_image_11 = PhotoImage(
            file=self.relative_to_assets("button_11.png"))
        self.button_11 = Button(
            image=self.button_image_11,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.clear_password(),
            relief="flat"
        )
        self.button_11.place(
            x=131.0,
            y=271.0,
            width=134.0,
            height=50.0
        )

        self.button_image_12 = PhotoImage(
            file=self.relative_to_assets("button_12.png"))
        self.button_12 = Button(
            image=self.button_image_12,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.submit_password() or self.clear_password() ,
            relief="flat"
        )
        self.button_12.place(
            x=364.0,
            y=143.0,
            width=119.0,
            height=42.0
        )

        self.button_image_13 = PhotoImage(
            file=self.relative_to_assets("button_13.png"))
        self.button_13 = Button(
            image=self.button_image_13,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("ChangPassword") or self.clear_password(),
            relief="flat"
        )
        self.button_13.place(
            x=364.0,
            y=203.0,
            width=119.0,
            height=42.0
        )

        self.button_image_14 = PhotoImage(
            file=self.relative_to_assets("button_14.png"))
        self.button_14 = Button(
            image=self.button_image_14,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("StartPage") or self.clear_password() ,
            relief="flat"
        )
        self.button_14.place(
            x=364.0,
            y=256.0,
            width=119.0,
            height=42.0
        )
    
        self.buttons_hidden=False
        
        
        self.entered_password = ""
        self.stop_thread3 = False
        self.stop_thread1 = False
        
    def hide_buttons(self):
        if not self.buttons_hidden:
            self.button_1.place_forget()
            self.button_2.place_forget()
            self.button_3.place_forget()
            self.button_4.place_forget()
            self.button_5.place_forget()
            self.button_6.place_forget()
            self.button_7.place_forget()
            self.button_8.place_forget()
            self.button_9.place_forget()
            self.button_10.place_forget()
            self.button_11.place_forget()
            self.button_12.place_forget()
            self.button_13.place_forget()
            self.button_14.place_forget()
            # self.entry_1.place_forget()

            self.buttons_hidden = True
            # self.stop_threads()

    def show_buttons(self):
        if self.buttons_hidden:
            self.button_1.place(           x=56.0,
                                        y=271.0,
                                        width=62.0,
                                        height=55.0)
            self.button_2.place(          x=51.0,
                                        y=48.0,
                                        width=67.0,
                                        height=64.0)
            self.button_3.place(        x=127.0,
                                        y=49.0,
                                        width=67.0,
                                        height=64.0)
            self.button_4.place(         x=203.0,
                                        y=50.0,
                                        width=67.0,
                                        height=64.0)
            self.button_5.place(      x=51.0,
                                        y=194.0,
                                        width=67.0,
                                        height=64.0)
            self.button_6.place(      x=127.0,
                                    y=194.0,
                                    width=67.0,
                                    height=64.0)
            self.button_7.place(       x=203.0,
                                    y=195.0,
                                    width=67.0,
                                    height=64.0)
            self.button_8.place(          x=51.0,
                                    y=121.0,
                                    width=67.0,
                                    height=64.0)
            self.button_9.place(
            x=127.0,
            y=121.0,
            width=67.0,
            height=64.0
        )
            self.button_10.place(
            x=203.0,
            y=122.0,
            width=67.0,
            height=64.0
        )
            self.button_11.place(
            x=131.0,
            y=271.0,
            width=134.0,
            height=50.0
        )

            self.button_12.place(
            x=364.0,
            y=143.0,
            width=119.0,
            height=42.0
        )
            self.button_13.place(
            x=364.0,
            y=203.0,
            width=119.0,
            height=42.0
        )
            self.button_14.place(
            x=364.0,
            y=256.0,
            width=119.0,
            height=42.0
        )
        #     self.entry_1.place(
        #     x=328.0,
        #     y=61.0,
        #     width=180.0,
        #     height=51.0
        # )


            self.buttons_hidden = False
   

    def start_door_control(self):
        self.stop_thread3 = False
        self.thread3 = threading.Thread(target=self.run_door_control)
        self.thread3.start()
        print("Thread 3 started")
    def stop_door_control(self):
        self.stop_thread3 = True
        print("Thread 3 stopped")

    def run_door_control(self):
        ControlDoor.open_door()
        while not self.stop_thread3:
            if  ControlDoor.GetDistance() > 35 :
                ControlDoor.close_door()
                self.stop_door_control()
                self.controller.show_frame("StartPage")
            time.sleep(1)  # Thực hiện kiểm tra lại sau một khoảng thời gian

    def start_threads(self):
        # self.start_detection()
        # self.start_prediction()
        self.start_door_control()

    def stop_threads(self):
        # self.stop_detection()
        # self.stop_prediction()
        self.stop_door_control()

    def submit_password(self):
        if self.entered_password == load_password():
            print("Password correct!")
            self.controller.show_frame("EndPage")
            self.start_door_control()
            # self.controller.after(3000, lambda: self.controller.show_frame("StartPage"))
        else:
            # messagebox.showinfo("Error", "Incorrect password!")
            self.controller.show_frame("StartPage")
           

    def button_click(self, button_number):
        self.entered_password += str(button_number)
        print("Entered password:", self.entered_password)
    def clear_password(self):
        self.entered_password = ""
        print("Password cleared!")


    @staticmethod
    def relative_to_assets(path: str) -> Path:
        OUTPUT_PATH = Path(__file__).parent
        ASSETS_PATH = OUTPUT_PATH / Path("./assets/frame6")
        return ASSETS_PATH / Path(path)


from tkinter import ttk

class CustomProgressBar(ttk.Progressbar):
    def __init__(self, parent, length):
        super().__init__(parent, orient='horizontal', mode='determinate', length=length)
        self.value = 0

    def update_value(self, new_value):
        self.value = new_value
        self['value'] = self.value

    def reset(self):
        self.value = 0
        self['value'] = self.value

class ChangPassword(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller


        self.canvas = Canvas(
            self,
            bg = "#FFFFFF",
            height = 368,
            width = 594,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )

        self.canvas.place(x = 0, y = 0)
        self.image_image_1 = PhotoImage(
            file=self.relative_to_assets("image_1.png"))
        self.image_1 = self.canvas.create_image(
            297.0,
            184.0,
            image=self.image_image_1
        )

        self.button_image_1 = PhotoImage(
            file=self.relative_to_assets("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_1_click(),
            relief="flat"
        )
        self.button_1.place(
            x=56.0,
            y=271.0,
            width=62.0,
            height=55.0
        )

        self.button_image_2 = PhotoImage(
            file=self.relative_to_assets("button_2.png"))
        self.button_2 = Button(
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_2_click(),
            relief="flat"
        )
        self.button_2.place(
            x=51.0,
            y=48.0,
            width=67.0,
            height=64.0
        )

        self.button_image_3 = PhotoImage(
            file=self.relative_to_assets("button_3.png"))
        self.button_3 = Button(
            image=self.button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_3_click(),
            relief="flat"
        )
        self.button_3.place(
            x=127.0,
            y=49.0,
            width=67.0,
            height=64.0
        )

        self.button_image_4 = PhotoImage(
            file=self.relative_to_assets("button_4.png"))
        self.button_4 = Button(
            image=self.button_image_4,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_4_click(),
            relief="flat"
        )
        self.button_4.place(
            x=203.0,
            y=50.0,
            width=67.0,
            height=64.0
        )

        self.button_image_5 = PhotoImage(
            file=self.relative_to_assets("button_5.png"))
        self.button_5 = Button(
            image=self.button_image_5,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_5_click(),
            relief="flat"
        )
        self.button_5.place(
            x=51.0,
            y=194.0,
            width=67.0,
            height=64.0
        )

        self.button_image_6 = PhotoImage(
            file=self.relative_to_assets("button_6.png"))
        self.button_6 = Button(
            image=self.button_image_6,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_6_click(),
            relief="flat"
        )
        self.button_6.place(
            x=127.0,
            y=194.0,
            width=67.0,
            height=64.0
        )

        self.button_image_7 = PhotoImage(
            file=self.relative_to_assets("button_7.png"))
        self.button_7 = Button(
            image=self.button_image_7,
            borderwidth=0,
            highlightthickness=0,
            command=lambda : self.button_7_click(),
            relief="flat"
        )
        self.button_7.place(
            x=203.0,
            y=195.0,
            width=67.0,
            height=64.0
        )

        self.button_image_8 = PhotoImage(
            file=self.relative_to_assets("button_8.png"))
        self.button_8 = Button(
            image=self.button_image_8,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_8_click(),
            relief="flat"
        )
        self.button_8.place(
            x=51.0,
            y=121.0,
            width=67.0,
            height=64.0
        )

        self.button_image_9 = PhotoImage(
            file=self.relative_to_assets("button_9.png"))
        self.button_9 = Button(
            image=self.button_image_9,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_9_click(),
            relief="flat"
        )
        self.button_9.place(
            x=127.0,
            y=121.0,
            width=67.0,
            height=64.0
        )

        self.button_image_10 = PhotoImage(
            file=self.relative_to_assets("button_10.png"))
        self.button_10 = Button(
            image=self.button_image_10,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.button_10_click(),
            relief="flat"
        )
        self.button_10.place(
            x=203.0,
            y=122.0,
            width=67.0,
            height=64.0
        )

        self.button_image_11 = PhotoImage(
            file=self.relative_to_assets("button_11.png"))
        self.button_11 = Button(
            image=self.button_image_11,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.clear_entry(),
            relief="flat"
        )
        self.button_11.place(
            x=131.0,
            y=271.0,
            width=134.0,
            height=50.0
        )

        self.button_image_12 = PhotoImage(
            file=self.relative_to_assets("button_12.png"))
        self.button_12 = Button(
            image=self.button_image_12,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.change_password_entry(),
            relief="flat"
        )
        self.button_12.place(
            x=358.0,
            y=217.0,
            width=119.0,
            height=42.0
        )

        self.button_image_13 = PhotoImage(
            file=self.relative_to_assets("button_13.png"))
        self.button_13 = Button(
            image=self.button_image_13,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: self.controller.show_frame("Password"),
            relief="flat"
        )
        self.button_13.place(
            x=358.0,
            y=284.0,
            width=119.0,
            height=42.0
        )

        self.entry_image_1 = PhotoImage(
            file=self.relative_to_assets("entry_1.png"))
        self.entry_bg_1 = self.canvas.create_image(
            418.0,
            87.5,
            image=self.entry_image_1
        )
        self.entry_1 = Entry(
            bd=0,
            bg="#FFFFFF",
            fg="#000716",
            highlightthickness=0,
            show="*"
        )
        self.entry_1.place(
            x=328.0,
            y=61.0,
            width=180.0,
            height=51.0
        )

        self.entry_image_2 = PhotoImage(
            file=self.relative_to_assets("entry_2.png"))
        self.entry_bg_2 = self.canvas.create_image(
            418.0,
            167.5,
            image=self.entry_image_2
        )
        self.entry_2 = Entry(
            bd=0,
            bg="#FFFFFF",
            fg="#000716",
            highlightthickness=0,
            show="*"
        )
        self.entry_2.place(
            x=328.0,
            y=141.0,
            width=180.0,
            height=51.0
            
        )

        self.canvas.create_text(
            316.0,
            41.0,
            anchor="nw",
            text="Old Password",
            fill="#000000",
            font=("Inter Black", 16 * -1)
        )

        self.canvas.create_text(
            316.0,
            118.0,
            anchor="nw",
            text="New Password",
            fill="#000000",
            font=("Inter Black", 16 * -1)
        )
        self.buttons_hidden=False
    def insert_number(self,number,entry):
            global selected_entry
            selected_entry = entry
            if selected_entry == self.entry_1:
                selected_entry.insert(ttk.END, str(number))
            elif selected_entry == self.entry_2:
                selected_entry.insert(ttk.END, str(number))
    def button_1_click(self):
            selected_entry = self.focus_get()

            if selected_entry == self.entry_1:
                self.entry_1.insert("end", "0")
            elif selected_entry == self.entry_2:
                self.entry_2.insert("end", "0")
    def button_2_click(self):
            selected_entry = self.focus_get()

            if selected_entry == self.entry_1:
                self.entry_1.insert("end", "1")
            elif selected_entry == self.entry_2:
                self.entry_2.insert("end", "1")
    def button_3_click(self):
            selected_entry = self.focus_get()

            if selected_entry == self.entry_1:
                self.entry_1.insert("end", "2")
            elif selected_entry == self.entry_2:
                self.entry_2.insert("end", "2")
    def button_4_click(self):
            selected_entry = self.focus_get()

            if selected_entry == self.entry_1:
                self.entry_1.insert("end", "3")
            elif selected_entry == self.entry_2:
                self.entry_2.insert("end", "3")
    def button_5_click(self):
            selected_entry = self.focus_get()

            if selected_entry == self.entry_1:
                self.entry_1.insert("end", "7")
            elif selected_entry == self.entry_2:
                self.entry_2.insert("end", "7")
    def button_6_click(self):
            selected_entry = self.focus_get()

            if selected_entry == self.entry_1:
                self.entry_1.insert("end", "8")
            elif selected_entry == self.entry_2:
                self.entry_2.insert("end", "8")
    def button_7_click(self):
            selected_entry = self.focus_get()

            if selected_entry == self.entry_1:
                self.entry_1.insert("end", "9")
            elif selected_entry == self.entry_2:
                self.entry_2.insert("end", "9")
    def button_8_click(self):
            selected_entry = self.focus_get()

            if selected_entry == self.entry_1:
                self.entry_1.insert("end", "4")
            elif selected_entry == self.entry_2:
                self.entry_2.insert("end", "4")
    def button_9_click(self):
            selected_entry = self.focus_get()

            if selected_entry == self.entry_1:
                self.entry_1.insert("end", "5")
            elif selected_entry == self.entry_2:
                self.entry_2.insert("end", "5")
    def button_10_click(self):
            selected_entry = self.focus_get()

            if selected_entry == self.entry_1:
                self.entry_1.insert("end", "6")
            elif selected_entry == self.entry_2:
                self.entry_2.insert("end", "6")
    def change_password_entry(self):
        entered_old_password = self.entry_1.get()
        entered_new_password = self.entry_2.get()

        # Kiểm tra tính hợp lệ của mật khẩu cũ
        
        if entered_old_password == load_password():
            # Thực hiện thay đổi mật khẩu
            save_password(entered_new_password)
            self.controller.show_frame("Password")
            # messagebox.showinfo( "Notice" ,"Change password successfully!")
            
        else:
            self.controller.show_frame("StartPage")
            # messagebox.showerror("Error", "Old password is incorrect!")

        # Xóa nội dung trong entry_1 và entry_2
        self.entry_1.delete(0, "end")
        self.entry_2.delete(0, "end")
    def clear_entry(self):
        selected_entry = self.focus_get()

        if selected_entry == self.entry_1:
            self.entry_1.delete(0, "end")
        elif selected_entry == self.entry_2:
            self.entry_2.delete(0, "end")



    def hide_buttons(self):
        if not self.buttons_hidden:
            self.button_1.place_forget()
            self.button_2.place_forget()
            self.button_3.place_forget()
            self.button_4.place_forget()
            self.button_5.place_forget()
            self.button_6.place_forget()
            self.button_7.place_forget()
            self.button_8.place_forget()
            self.button_9.place_forget()
            self.button_10.place_forget()
            self.button_11.place_forget()
            self.button_12.place_forget()
            self.button_13.place_forget()
            self.entry_1.place_forget()
            self.entry_2.place_forget()

            self.buttons_hidden = True
            # self.stop_threads()

    def show_buttons(self):
        if self.buttons_hidden:
            self.button_1.place(           x=56.0,
                                        y=271.0,
                                        width=62.0,
                                        height=55.0)
            self.button_2.place(          x=51.0,
                                        y=48.0,
                                        width=67.0,
                                        height=64.0)
            self.button_3.place(        x=127.0,
                                        y=49.0,
                                        width=67.0,
                                        height=64.0)
            self.button_4.place(         x=203.0,
                                        y=50.0,
                                        width=67.0,
                                        height=64.0)
            self.button_5.place(      x=51.0,
                                        y=194.0,
                                        width=67.0,
                                        height=64.0)
            self.button_6.place(      x=127.0,
                                    y=194.0,
                                    width=67.0,
                                    height=64.0)
            self.button_7.place(       x=203.0,
                                    y=195.0,
                                    width=67.0,
                                    height=64.0)
            self.button_8.place(          x=51.0,
                                    y=121.0,
                                    width=67.0,
                                    height=64.0)
            self.button_9.place(
            x=127.0,
            y=121.0,
            width=67.0,
            height=64.0
        )
            self.button_10.place(
            x=203.0,
            y=122.0,
            width=67.0,
            height=64.0
        )
            self.button_11.place(
            x=131.0,
            y=271.0,
            width=134.0,
            height=50.0
        )

            self.button_12.place(
                        x=358.0,
                y=217.0,
                width=119.0,
                height=42.0
        )
            self.button_13.place(
            x=358.0,
            y=284.0,
            width=119.0,
            height=42.0
        )

            self.entry_1.place(
                        x=328.0,
    y=61.0,
    width=180.0,
    height=51.0
    
        )
            self.entry_2.place(
                x=328.0,
                y=141.0,
                width=180.0,
                height=51.0
            )


            self.buttons_hidden = False
   
    @staticmethod
    def relative_to_assets(path: str) -> Path:
        OUTPUT_PATH = Path(__file__).parent
        ASSETS_PATH = OUTPUT_PATH / Path("./assets/frame7")
        return ASSETS_PATH / Path(path)

app = MainUI()
app.mainloop()

