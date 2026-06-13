import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import datetime
import random

# Optional: TensorFlow prediction (if model available)
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class BrainTumorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Brain Tumor Detection System")
        self.root.geometry("900x650")
        self.root.configure(bg="#f8fafc")

        self.current_image_path = None
        self.current_image_photo = None
        self.model = None

        # Notebook Tabs
        self.notebook = ttk.Notebook(self.root)
        self.home_tab = tk.Frame(self.notebook, bg="#f8fafc")
        self.result_tab = tk.Frame(self.notebook, bg="#f8fafc")
        self.notebook.add(self.home_tab, text="Patient Entry")
        self.notebook.pack(fill="both", expand=True)

        self.create_home_tab()

    # ---------------- HOME TAB ----------------
    def create_home_tab(self):
        tk.Label(self.home_tab, text="🧠 Brain Tumor Detection",
                 font=("Segoe UI", 22, "bold"),
                 fg="#1e3a8a", bg="#f8fafc").pack(pady=15)

        tk.Label(self.home_tab, text="Enter Patient Details",
                 font=("Segoe UI", 13), fg="#334155", bg="#f8fafc").pack()

        form_frame = tk.Frame(self.home_tab, bg="#f8fafc")
        form_frame.pack(pady=15)

        self.name_var = tk.StringVar()
        self.age_var = tk.StringVar()
        self.gender_var = tk.StringVar()
        self.mobile_var = tk.StringVar()

        # Name
        self.create_label_entry(form_frame, "Full Name", self.name_var, 0, 0)
        # Age
        self.create_label_entry(form_frame, "Age", self.age_var, 0, 1)
        # Gender
        tk.Label(form_frame, text="Gender", bg="#f8fafc", fg="#334155", font=("Segoe UI", 11)).grid(row=1, column=0, padx=10, pady=10, sticky="w")
        ttk.Combobox(form_frame, textvariable=self.gender_var, values=["Male", "Female", "Other"], width=25).grid(row=1, column=1, padx=10, pady=10)
        # Mobile
        self.create_label_entry(form_frame, "Patient ID / Mobile", self.mobile_var, 1, 2)

        # Upload Button
        upload_btn = tk.Button(self.home_tab, text="📁 Upload MRI Image", font=("Segoe UI", 12, "bold"),
                               bg="#2563eb", fg="white", padx=20, pady=8, relief="flat",
                               command=self.upload_image)
        upload_btn.pack(pady=20)

        # Image Preview
        self.preview_label = tk.Label(self.home_tab, bg="#f8fafc")
        self.preview_label.pack()

        # Detect Button
        self.detect_button = tk.Button(self.home_tab, text="🔍 Analyze MRI",
                                       font=("Segoe UI", 13, "bold"),
                                       bg="#16a34a", fg="white",
                                       padx=30, pady=10, relief="flat",
                                       state="disabled",
                                       command=self.detect_tumor)
        self.detect_button.pack(pady=25)

    def create_label_entry(self, frame, text, var, row, col):
        tk.Label(frame, text=text, bg="#f8fafc", fg="#334155", font=("Segoe UI", 11)).grid(row=row, column=col*2, padx=10, pady=10, sticky="w")
        tk.Entry(frame, textvariable=var, width=28, font=("Segoe UI", 10), relief="solid", bd=1).grid(row=row, column=col*2+1, padx=10, pady=10)

    # ---------------- IMAGE UPLOAD ----------------
    def upload_image(self):
        path = filedialog.askopenfilename(title="Select MRI Image",
                                          filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if path:
            self.current_image_path = path
            img = Image.open(path).resize((300, 300))
            self.current_image_photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self.current_image_photo)
            self.detect_button.config(state="normal")

    # ---------------- DETECTION LOGIC ----------------
    def detect_tumor(self):
        if not all([self.name_var.get(), self.age_var.get(), self.gender_var.get(), self.mobile_var.get(), self.current_image_path]):
            messagebox.showwarning("Missing Info", "Please fill all fields and upload an image.")
            return

        try:
            self.detect_button.config(state="disabled", text="Analyzing...")
            self.root.update()

            # Simulated prediction (replace with actual model)
            if TENSORFLOW_AVAILABLE and self.model:
                img = self.preprocess_image(self.current_image_path)
                prediction = self.model.predict(img, verbose=0)[0][0]
            else:
                prediction = random.uniform(0.1, 0.9)

            threshold = 0.5
            if prediction >= threshold:
                result_text = "Tumor Detected"
                color = "#dc2626"  # red
                recommendation = (
                    "Consult a neurologist or oncologist immediately.\n"
                    "Carry this MRI scan report for detailed evaluation.\n"
                    "Further advanced imaging (e.g., contrast MRI or biopsy) may be required."
                )
            else:
                result_text = "Tumor Not Detected"
                color = "#16a34a"  # green
                recommendation = (
                    "No signs of tumor detected.\n"
                    "Maintain a healthy lifestyle and attend periodic check-ups."
                )


            result_data = {
                "name": self.name_var.get(),
                "age": self.age_var.get(),
                "gender": self.gender_var.get(),
                "mobile": self.mobile_var.get(),
                "confidence": f"{prediction:.2f}",
                "result": result_text,
                "color": color,
                "recommendation": recommendation
            }

            self.show_result_tab(result_data)
            self.detect_button.config(state="normal", text="🔍 Analyze MRI")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.detect_button.config(state="normal", text="🔍 Analyze MRI")

    # ---------------- RESULT TAB ----------------
    def show_result_tab(self, data):
        for widget in self.result_tab.winfo_children():
            widget.destroy()

        if "Result" not in [self.notebook.tab(i, "text") for i in range(self.notebook.index("end"))]:
            self.notebook.add(self.result_tab, text="Result")

        tk.Label(self.result_tab, text="🧠 Detection Result",
                 font=("Segoe UI", 20, "bold"),
                 fg="#1e3a8a", bg="#f8fafc").pack(pady=15)

        if self.current_image_photo:
            tk.Label(self.result_tab, image=self.current_image_photo, bg="#f8fafc").pack(pady=10)

        tk.Label(self.result_tab, text=data["result"],
                 font=("Segoe UI", 22, "bold"),
                 fg=data["color"], bg="#f8fafc").pack(pady=10)

        tk.Label(self.result_tab, text=f"Confidence: {data['confidence']}",
                 font=("Segoe UI", 13),
                 fg="#334155", bg="#f8fafc").pack()

        tk.Label(self.result_tab, text=data["recommendation"],
                 font=("Segoe UI", 11, "italic"),
                 fg="#475569", bg="#f8fafc", wraplength=700, justify="center").pack(pady=15)

        tk.Label(self.result_tab,
                 text=f"Patient: {data['name']} | Age: {data['age']} | Gender: {data['gender']} | ID: {data['mobile']}",
                 font=("Segoe UI", 10),
                 fg="#475569", bg="#f8fafc").pack(pady=5)

        # Buttons
        btn_frame = tk.Frame(self.result_tab, bg="#f8fafc")
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="🆕 New Test", bg="#22c55e", fg="white",
                  font=("Segoe UI", 11, "bold"), padx=20, pady=8,
                  relief="flat", command=self.new_test).pack(side="left", padx=10)

        tk.Button(btn_frame, text="💾 Save Report", bg="#64748b", fg="white",
                  font=("Segoe UI", 11, "bold"), padx=20, pady=8,
                  relief="flat", command=lambda: self.save_report(data)).pack(side="left", padx=10)

        self.notebook.select(self.result_tab)

    # ---------------- UTILITIES ----------------
    def new_test(self):
        self.name_var.set("")
        self.age_var.set("")
        self.gender_var.set("")
        self.mobile_var.set("")
        self.current_image_path = None
        self.preview_label.config(image="")
        self.detect_button.config(state="disabled")
        self.notebook.select(self.home_tab)

    def save_report(self, data):
        filename = f"BrainTumorReport_{data['mobile']}_{datetime.date.today()}.txt"
        with open(filename, "w") as f:
            f.write("Brain Tumor Detection Report\n")
            f.write("============================\n\n")
            f.write(f"Patient Name : {data['name']}\n")
            f.write(f"Age          : {data['age']}\n")
            f.write(f"Gender       : {data['gender']}\n")
            f.write(f"Patient ID   : {data['mobile']}\n\n")
            f.write(f"Result       : {data['result']}\n")
            f.write(f"Confidence   : {data['confidence']}\n\n")
            f.write("Next Steps:\n")
            f.write(f"{data['recommendation']}\n")
        messagebox.showinfo("Saved", f"Report saved as {filename}")

    def preprocess_image(self, path):
        img = Image.open(path).resize((150, 150))
        img = tf.keras.utils.img_to_array(img)
        img = img / 255.0
        return img.reshape(1, 150, 150, 3)


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = BrainTumorApp(root)
    root.mainloop()
