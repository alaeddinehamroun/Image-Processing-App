from tkinter import *
from tkinter import filedialog as fd
from tkinter import simpledialog

from utils import *
import customtkinter
import numpy as np
from PIL import ImageTk, Image
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")
root  = customtkinter.CTk()
root.title("Image processing")
root.geometry("800x600") 

def apply_dilation(d):
  global dilated_image
  global filter_data
  size = simpledialog.askstring(
            "Kernel size",
            "Enter kernel size",
            parent=root)
  
  d_data = dilation(d, int(size))
  top = customtkinter.CTkToplevel()
  top.title('Dilation Applied')

  dilated_image = ImageTk.PhotoImage(Image.fromarray(d_data))
  dilated_image_label = Label(top, image=dilated_image)
  dilated_image_label.pack()
  filter_data = d_data
  apply_btn = customtkinter.CTkButton(top, text="Apply Dilation", command=apply_image_change)
  apply_btn.pack()
def apply_erosion(d):
  global eros_image
  global filter_data
  size = simpledialog.askstring(
            "Kernel size",
            "Enter kernel size",
            parent=root)
  e_data = erosion(d, int(size))
  top = customtkinter.CTkToplevel()
  top.title('Erosion Applied')

  eros_image = ImageTk.PhotoImage(Image.fromarray(e_data))
  eros_image_label = Label(top, image=eros_image)
  eros_image_label.pack()
  filter_data = e_data
  apply_btn = customtkinter.CTkButton(top, text="Apply Dilation", command=apply_image_change)
  apply_btn.pack()

def apply_opening(d):
  global opening_image
  global filter_data
  size = simpledialog.askstring(
            "Kernel size",
            "Enter kernel size",
            parent=root)
  o_data = opening(d, int(size))
  top = customtkinter.CTkToplevel()
  top.title('Erosion Applied')

  opening_image = ImageTk.PhotoImage(Image.fromarray(o_data))
  opening_image_label = Label(top, image=opening_image)
  opening_image_label.pack()
  filter_data = o_data
  apply_btn = customtkinter.CTkButton(top, text="Apply Dilation", command=apply_image_change)
  apply_btn.pack()

def apply_closing(d):
  global closing_image
  global filter_data
  size = simpledialog.askstring(
            "Kernel size",
            "Enter kernel size",
            parent=root)
  c_data = closing(d, int(size))
  top = customtkinter.CTkToplevel()
  top.title('Erosion Applied')

  closing_image = ImageTk.PhotoImage(Image.fromarray(c_data))
  closing_image_label = Label(top, image=closing_image)
  closing_image_label.pack()
  filter_data = c_data
  apply_btn = customtkinter.CTkButton(top, text="Apply Dilation", command=apply_image_change)
  apply_btn.pack()
def apply_binarize(d):
  if root.filename.endswith(".ppm"):
    d = rgb2gray(d)
  threshold = simpledialog.askstring(
            "Threshold",
            "Enter threshold",
            parent=root)
  data = binarize(d, int(threshold))
  for widgets in root.winfo_children():
    widgets.destroy()
  show_image(data)
def add_noise(data):
  data = apply_noise(data)
  for widgets in root.winfo_children():
    widgets.destroy()
  show_image(data)
def apply_rgb2gray(data):
  data = rgb2gray(data)
  for widgets in root.winfo_children():
    widgets.destroy()
  show_image(data)
def apply_thresholding(thresholding_type):
  global thresholded_image

  if (thresholding_type) == "indep":
    s = simpledialog.askstring(
            "Thresholds",
            "Provide threshold for the 3 channels in this order: [R] [G] [B]",
            parent=root)
    sR, sG, sB = [int(i) for i in s.split() if i.isdigit()]
    
    s_data = seuillage_indep(data, sR, sG, sB)
  elif (thresholding_type) == "et":
    s = simpledialog.askstring(
            "Thresholds",
            "Provide threshold for the 3 channels in this order: [R] [G] [B]",
            parent=root)
    sR, sG, sB = [int(i) for i in s.split() if i.isdigit()]
    
    s_data = seuillage_ET(data, sR, sG, sB)

  elif (thresholding_type) == "ou":
    s = simpledialog.askstring(
            "Thresholds",
            "Provide threshold for the 3 channels in this order: [R] [G] [B]",
            parent=root)
    sR, sG, sB = [int(i) for i in s.split() if i.isdigit()]
    
    s_data = seuillage_OU(data, sR, sG, sB)
  elif (thresholding_type) == "auto_indep":
    s_data, sR, sG, sB = seuillage_auto_indep(data)
  elif (thresholding_type) == "auto_et":
    s_data, sR, sG, sB = seuillage_autp_ET(data)
  elif (thresholding_type) == "auto_ou":
    s_data, sR, sG, sB = seuillage_auto_OU(data)
  top = customtkinter.CTkToplevel()
  top.title('Thresholding Applied')

  thresholded_image = ImageTk.PhotoImage(Image.fromarray(s_data))
  thresholded_image_label = Label(top, image=thresholded_image)
  thresholded_image_label.pack()


  sR_label = customtkinter.CTkLabel(top)
  sR_label.configure(text="Threshold for R channel: " + str(sR))
  sR_label.pack()
  sG_label = customtkinter.CTkLabel(top)
  sG_label.configure(text="Threshold for G channel: " + str(sG))
  sG_label.pack()
  sB_label = customtkinter.CTkLabel(top)
  sB_label.configure(text="Threshold for B channel: " + str(sB))
  sB_label.pack()

  close_btn = customtkinter.CTkButton(top, text="close window", command=top.destroy)
  close_btn.pack()

    
def open_new_image():
  for widgets in root.winfo_children():
    widgets.destroy()
  read_image()
  

def save_image():

  answer = simpledialog.askstring(
            "Filename",
            "Provide a name to the saved image",
            parent=root)
  write_pgm(data, format, comment, n_cols, n_raws, p_max, str(answer))


def read_image():
    
    
    global data, format, comment, n_cols, n_raws, p_max
    image_types = (
        ('pgm images', '*.pgm'),
        ('ppm images', '*.ppm'),
        ('pgn images', '*.png')
    )
    root.filename = fd.askopenfilename(
        title='Open an image',
        initialdir='~/Desktop/image_processing_project/assets/demo_imgs',
        filetypes=image_types
    )
    

    if root.filename.endswith(".pgm"):
      data, format, comment, n_cols, n_raws, p_max = read_pgm(filename=root.filename)
    elif root.filename.endswith(".ppm"):
      data, format, comment, n_cols, n_raws, p_max = read_ppm(filename=root.filename)
    
    data = data.astype(np.uint8)
    #data = apply_noise(data)
    
    show_image(data)
def open_filter_window(filtername):
      global filter_image
      global filter_data
      top = customtkinter.CTkToplevel()
      top.title(filtername+' Filter Applied')
      
      if len(data.shape)>2:
        filter_data = apply_filter_rgb(data, filtername)
      else:
        
        filter_data= apply_filter(data, filtername)

      filter_image = ImageTk.PhotoImage(Image.fromarray(filter_data))
      filter_image_label = Label(top, image=filter_image)
      filter_image_label.grid(row=0, column=1)


      s_label = customtkinter.CTkLabel(top)
      s_label.configure(text="SNR: " + str(signal_to_noise_ratio(data, filter_data)))
      s_label.grid(row=1, column=1)
      
      apply_btn = customtkinter.CTkButton(top, text="Apply Filter", command=apply_image_change)
      apply_btn.grid(row=2, column = 2)
      close_btn = customtkinter.CTkButton(top, text="close window", command=top.destroy)
      close_btn.grid(row=2, column=0)
def apply_image_change():
        global data
        data = filter_data
        data = data.astype(np.uint8)
        for widgets in root.winfo_children():
          widgets.destroy()
        show_image(data)
def show_image(data):
    global  my_image

    open_button.pack_forget()

    menubar = Menu(root)
    root.config(menu=menubar)
    file_menu = Menu(menubar)
    file_menu.add_command(
      label = 'Open New image',
      command= open_new_image
    )
    file_menu.add_command(
      label = "save image",
      command =  save_image
    )
    
    thresholding_menu = Menu(menubar)
    thresholding_menu.add_command(
      label = "Independant thresholding",
      command = lambda: apply_thresholding("indep")
    )
    thresholding_menu.add_command(
      label = "AND thresholding",
      command = lambda: apply_thresholding("et")
    )
    thresholding_menu.add_command(
      label = "OR thresholding",
      command = lambda: apply_thresholding("ou")
    )
    thresholding_menu.add_command(
      label = "Independant thresholding(otsu)",
      command = lambda: apply_thresholding("auto_indep")
    )
    thresholding_menu.add_command(
      label = "AND thresholding(otsu)",
      command = lambda: apply_thresholding("auto_et")
    )
    thresholding_menu.add_command(
      label = "OR thresholding(otsu)",
      command = lambda: apply_thresholding("auto_ou")
    )

    Histogram_menu = Menu(menubar)
    Histogram_menu.add_command(
      label = "Grayscale Histogram",
      command = lambda: create_grayscale_hist(data)
    )
    Histogram_menu.add_command(
      label = "Cumulative Grayscale Histogram",
      command = lambda: create_cumulative_hist(data)
    )

    Filter_menu = Menu(menubar)
    Filter_menu.add_command(
      label = "Mean Filter",
      command = lambda: open_filter_window("Mean")
    )
    Filter_menu.add_command(
      label = "Median Filter",
      command = lambda: open_filter_window("Median")
    )
    Filter_menu.add_command(
      label = "Gaussian Filter (3x3)",
      command = lambda: open_filter_window("Gaussian")
    )

    utils_menu = Menu(menubar)
    utils_menu.add_command(
      label = 'Add noise',
      command= lambda: add_noise(data)
    )
    utils_menu.add_command(
      label="Binarize",
      command= lambda: apply_binarize(data)
    )
    if (root.filename.endswith(".ppm")):
      utils_menu.add_command(
        label="RGB to gray",
        command= lambda: apply_rgb2gray(data)
      )

    morph_menu = Menu(menubar)
    morph_menu.add_command(
      label="Dilation",
      command= lambda: apply_dilation(data)
    )
    morph_menu.add_command(
      label="Erosion",
      command= lambda: apply_erosion(data)
    )
    morph_menu.add_command(
      label="Opening",
      command= lambda: apply_opening(data)
    )
    morph_menu.add_command(
      label="Closing",
      command= lambda: apply_closing(data)
    )
    
    menubar.add_cascade(label="File", menu=file_menu)
    menubar.add_cascade(label='Histograms', menu=Histogram_menu)
    menubar.add_cascade(label='Filters', menu=Filter_menu)
    if (root.filename.endswith(".ppm")):
      menubar.add_cascade(label='Thresholding', menu=thresholding_menu)
    menubar.add_cascade(label='Utils', menu=utils_menu)
    menubar.add_cascade(label= "Morphological ops", menu=morph_menu)
    
    # main image viewer
    my_image = ImageTk.PhotoImage(Image.fromarray(data))
    
    
    main_image_frame = customtkinter.CTkFrame(root)
    main_image_frame.pack(pady=10, padx=10, fill="both", expand=True)
    my_image_label = Label(main_image_frame, image=my_image)
    my_image_label.pack()

    calcul_frame = customtkinter.CTkFrame(root)
    calcul_frame.pack(fill="x", side = BOTTOM)
    mean_label = customtkinter.CTkLabel(calcul_frame)
    mean_label.configure(text = "Mean: " + str(img_mean(data)))
    mean_label.pack(anchor = W, side=BOTTOM)

    sd_label = customtkinter.CTkLabel(calcul_frame)
    sd_label.configure(text="Standard deviation: " + str(img_ecart_type(data)))
    sd_label.pack(anchor = W, side=BOTTOM)

    

    
    
    # mean_filter_button =  customtkinter.CTkButton(root, text = "Apply mean filter", command= lambda: open_filter_window("Mean"))
    # mean_filter_button.pack()
    # mean_filter_button =  customtkinter.CTkButton(root, text = "Apply median filter", command= lambda: open_filter_window("Median"))
    # mean_filter_button.pack()
    # mean_filter_button =  customtkinter.CTkButton(root, text = "Apply gaussian filter", command= lambda: open_filter_window("Gaussian"))
    # mean_filter_button.pack()

open_button = customtkinter.CTkButton(
    root,
    text = 'Import an image',
    command = lambda: read_image()
)
open_button.pack(pady=200)



root.mainloop()