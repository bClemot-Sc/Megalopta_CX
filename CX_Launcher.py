#### GUI for CX modeling
## Autor: Bastien Clémot

## ----- Import packages
import customtkinter as ctk


## ----- Command functions

# Condition for Logic activation function
def show_threshold(*args):
    selected_option = activation_var.get()

    if selected_option == "Linear":
        threshold_label.grid_forget()
        threshold_entry.grid_forget()
        i_row = 0

    elif selected_option == "Logic":
        threshold_label.grid(row=3, column=0, pady=20, sticky="w", padx=200)
        threshold_entry.grid(row=3, column=1, padx=150)
        i_row = 1

    # Adjust other widgets position
    run_button.grid(row=3+i_row, column=0, columnspan=2, pady=20)
    

# Check for missing inputs and run simulation
def run_simulation():
    k=0
    error=0

    # Remove previous error
    error1_label.grid_forget()
    error2_label.grid_forget()

    # Check for inputs and issues
    try:
        time = int(time_entry.get())
        if time <=0:
                error1_label.grid(row=5+i_row+k, column=0, columnspan=2,padx=50, pady=10)
                error=1
                k+=1 
    except ValueError:
        error1_label.grid(row=5+i_row+k, column=0, columnspan=2,padx=50, pady=10)
        error=1
        k+=1

    activation = activation_menu.get()
    if activation == "Logic":
        try:
            threshold = float(threshold_entry.get())
            if not 0.0 <= threshold <= 1.0:
                error2_label.grid(row=5+i_row+k, column=0, columnspan=2,padx=50, pady=10)
                error=1
                k+=1 
        except ValueError:
            error2_label.grid(row=5+i_row+k, column=0, columnspan=2,padx=50, pady=10)
            error=1
            k+=1

    # Run the simulation
    if error==0:
        root.quit()


## ----- Configure GUI

if __name__ == "__main__":
    # Configure theme
    ctk.set_default_color_theme("green")
    # Call window
    root = ctk.CTk()
    width = root.winfo_screenwidth() 
    height = root.winfo_screenheight()
    root.geometry("{}x{}+{}+{}".format(width-50,height-100,20,20))
    root.title("CX simulation")
    
    # Configure tabs
    tabs = ctk.CTkTabview(root, height=height-125, width=width-100, corner_radius=10)
    tabs.pack()
    tab_1 = tabs.add("control panel")
    tab_2 = tabs.add("Heatmap")
    tab_1.columnconfigure(0, weight=1)
    tab_2.columnconfigure(1, weight=2)
    
    # Title label
    title_label = ctk.CTkLabel(tab_1, text="Central Complex Control Panel", font=ctk.CTkFont(size=40, weight="bold"))
    title_label.grid(row=0, column=0, columnspan=2, pady=20)
    
    # Simulation time label
    time_label = ctk.CTkLabel(tab_1, text="• Simulation time:", font=ctk.CTkFont(size=30))
    time_label.grid(row=1, column=0, pady=20, sticky="w", padx=150)
    # Simulation time entry
    time_entry = ctk.CTkEntry(tab_1, placeholder_text=500, font=ctk.CTkFont(size=20))
    time_entry.grid(row=1, column=1, padx=150)
    
    # Activation function label
    activation_label = ctk.CTkLabel(tab_1, text="• Activation function:", font=ctk.CTkFont(size=30))
    activation_label.grid(row=2, column=0, pady=20, sticky="w", padx=150)
    # Activation function menu
    activation_var = ctk.StringVar(tab_1)
    activation_var.set("Linear")
    activation_menu = ctk.CTkOptionMenu(tab_1, variable=activation_var, values=["Linear", "Logic"], font=ctk.CTkFont(size=20))
    activation_menu.grid(row=2, column=1, padx=150)
    
    # Threshold condition
    i_row=0
    activation_var.trace_add("write", show_threshold)
    # Threshold label
    threshold_label = ctk.CTkLabel(tab_1, text="→ Threshold:", font=ctk.CTkFont(size=30))
    threshold_label.grid_forget()
    # Threshold entry
    threshold_entry = ctk.CTkEntry(tab_1, placeholder_text=0.5, font=ctk.CTkFont(size=20))
    threshold_entry.grid_forget()
    
    # Initialize error labels
    # Initialize error
    error1_label = ctk.CTkLabel(tab_1, text="Simulation time value should be a positive integer.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))
    error2_label = ctk.CTkLabel(tab_1, text="Threshold value should be a float between 0 and 1.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))
    
    # Run button
    run_button = ctk.CTkButton(tab_1, text="Run simulation", command=run_simulation, font=ctk.CTkFont(size=20))
    run_button.grid(row=3, column=0, columnspan=2, pady=20)
    
    # Main loop
    root.mainloop()