import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox,filedialog
from PIL import Image, ImageTk
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# global variables
point1 = ()
point2 = ()
drawn = False
done = False
drawing = False
img = None
rect_id = None  # for rectangle id

def resize_img(image, scale):
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def draw_rectangle(event):
    global point1, point2, drawing, img, drawn, done, canvas, rect_id

    if event.widget == canvas:
        if not drawn:
            if event.type == tk.EventType.ButtonPress:
                if not drawing:
                    drawing = True
                    point1 = (event.x, event.y)
            elif event.type == tk.EventType.ButtonRelease:
                drawing = False
                point2 = (event.x, event.y)
                drawn = True
                if rect_id is not None:
                    canvas.delete(rect_id)
                rect_id = canvas.create_rectangle(point1[0], point1[1], point2[0], point2[1], outline='green')
            elif event.type == tk.EventType.Motion:
                if drawing:
                    point2 = (event.x, event.y)
                    if rect_id is not None:
                        canvas.delete(rect_id)
                    rect_id = canvas.create_rectangle(point1[0], point1[1], point2[0], point2[1], outline='green')

def main():
    # Set up SAM
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint = sam_checkpoint)
    sam.to(device = "cuda")
    segmentator = SamPredictor(sam)

    root = tk.Tk()

    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if not img_path:
        return

    global img, canvas, rect_id
    img = cv2.imread(img_path)
    segmentator.set_image(img)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    root.geometry(f"{int(screen_width*0.8)}x{int(screen_height*0.8)}+{int(screen_width*0.1)}+{int(screen_height*0.1)}")

    canvas = tk.Canvas(root, width=img.shape[1], height=img.shape[0])
    canvas.pack()
    img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    x = (img.shape[1] - img.shape[1]) // 2
    y = (img.shape[0] - img.shape[0]) // 2
    canvas.create_image(x, y, anchor='nw', image=img_tk)

    canvas.bind("<ButtonPress-1>", draw_rectangle)
    canvas.bind("<B1-Motion>", draw_rectangle)
    canvas.bind("<ButtonRelease-1>", draw_rectangle)

    def check_drawn():
        global drawn, done, rect_id
        if drawn and not done:
            if messagebox.askyesno("Confirmation", "Do you want to select again?"):
                drawn = False
                if rect_id is not None:
                    canvas.delete(rect_id)
            else:
                done = True
                root.quit()
        root.after(100, check_drawn)

    check_drawn()

    root.mainloop()

    if done:
        input_box = np.array([point1[0], point1[1], point2[0], point2[1]])

        mask, _, _ = segmentator.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        segmentation_mask = mask[0]
        binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
        alpha_channel = (binary_mask * 255).astype(np.uint8)
        object = np.dstack((img, alpha_channel))

        output_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if output_path:
            cv2.imwrite(output_path, object)

            # Show the output image
            output_image = Image.open(output_path)
            output_image.show()

            # Show a message box to let the user know the process has finished
            messagebox.showinfo("Information","The process has finished successfully!")

if __name__ == "__main__":
    main()
