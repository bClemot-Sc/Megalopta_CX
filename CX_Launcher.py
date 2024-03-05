#### GUI for CX modeling
## Autor: Bastien Clémot


## ----- Import packages
import customtkinter as ctk
import CX_Script


## ----- Command functions

# Condition for paradigm parameters
def show_parameters(*args):
    selected_option = paradigm_var.get()

    if selected_option == "Timed exploration":
        radius_label.grid_forget()
        radius_entry.grid_forget()
        food_label.grid_forget()
        food_entry.grid_forget()
        i_row = 0

    elif selected_option == "Till border exploration":
        food_label.grid_forget()
        food_entry.grid_forget()
        radius_label.grid(row=5, column=0, pady=20, sticky="w", padx=70)
        radius_entry.grid(row=5, column=1, sticky="ew", padx=30)
        i_row = 1

    elif selected_option == "Food seeking":
        radius_label.grid_forget()
        radius_entry.grid_forget()
        food_label.grid(row=5, column=0, pady=20, sticky="w", padx=70)
        food_entry.grid(row=5, column=1, sticky="ew", padx=30)
        i_row = 1

    # Adjust other widgets position
    run_button.grid(row=5+i_row, column=0, columnspan=2, pady=20)

# Check for missing inputs and run simulation
def run_simulation():
    k=0
    error=0

    # Remove previous error
    error1_label.grid_forget()
    error2_label.grid_forget()
    error3_label.grid_forget()
    error4_label.grid_forget()

    # Check for inputs and issues
    try:
        TIME = int(time_entry.get())
        if TIME <=0:
                error1_label.grid(row=7+i_row+k, column=0, columnspan=2,padx=50, pady=10)
                error=1
                k+=1
    except ValueError:
        error1_label.grid(row=7+i_row+k, column=0, columnspan=2,padx=50, pady=10)
        error=1
        k+=1

    try:
        NOISE = float(noise_entry.get())
        if not 0.0 <= NOISE <= 90.0:
                error2_label.grid(row=7+i_row+k, column=0, columnspan=2,padx=50, pady=10)
                error=1
                k+=1 
    except ValueError:
        error2_label.grid(row=7+i_row+k, column=0, columnspan=2,padx=50, pady=10)
        error=1
        k+=1

    RADIUS = 200
    PARADIGM = paradigm_menu.get()
    if PARADIGM == "Till border exploration":
        try:
            RADIUS = float(radius_entry.get())
            if not 0.0 <= RADIUS:
                error3_label.grid(row=7+i_row+k, column=0, columnspan=2,padx=50, pady=10)
                error=1
                k+=1 
        except ValueError:
            error3_label.grid(row=7+i_row+k, column=0, columnspan=2,padx=50, pady=10)
            error=1
            k+=1 
    
    FOOD = 5
    if PARADIGM == "Food seeking":
        try:
            FOOD = int(food_entry.get())
            if not 0.0 <= FOOD:
                error4_label.grid(row=7+i_row+k, column=0, columnspan=2,padx=50, pady=10)
                error=1
                k+=1 
        except ValueError:
            error4_label.grid(row=7+i_row+k, column=0, columnspan=2,padx=50, pady=10)
            error=1
            k+=1         

    PERIOD = period_menu.get()

    # Run the simulation
    if error==0:
        CX_Script.run_function(CX_Script.CON_MAT, TIME, PERIOD, NOISE)


## ----- Configure GUI
if __name__ == "__main__":
    # Configure theme
    ctk.set_default_color_theme("green")
    # Call window
    root = ctk.CTk()
    root.geometry("750x650")
    root.title("CX simulation")
    # Configure columns
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=2)

    # Title label
    title_label = ctk.CTkLabel(root, text="Central Complex Control Panel", font=ctk.CTkFont(size=30, weight="bold"))
    title_label.grid(row=0, column=0, columnspan=2, pady=20)

    # Simulation time label
    time_label = ctk.CTkLabel(root, text="• Maximum simulation time:", font=ctk.CTkFont(size=20))
    time_label.grid(row=1, column=0, pady=20, sticky="w", padx=50)
    # Simulation time entry
    time_entry = ctk.CTkEntry(root, placeholder_text=500)
    time_entry.grid(row=1, column=1, sticky="ew", padx=30)

    # Time period label
    period_label = ctk.CTkLabel(root, text="• Time period:", font=ctk.CTkFont(size=20))
    period_label.grid(row=2, column=0, pady=20, sticky="w", padx=50)
    # Time period menu
    period_menu = ctk.CTkOptionMenu(root, values=["Day", "Night"])
    period_menu.grid(row=2, column=1, sticky="ew", padx=30)

    # Noise range label
    noise_label = ctk.CTkLabel(root, text="• Noise range:", font=ctk.CTkFont(size=20))
    noise_label.grid(row=3, column=0, pady=20, sticky="w", padx=50)
    # Noise range entry
    noise_entry = ctk.CTkEntry(root, placeholder_text=10)
    noise_entry.grid(row=3, column=1, sticky="ew", padx=30)

    # Paradigm label
    paradigm_label = ctk.CTkLabel(root, text="• Paradigm:", font=ctk.CTkFont(size=20))
    paradigm_label.grid(row=4, column=0, pady=20, sticky="w", padx=50)
    # Paradigm menu
    paradigm_var = ctk.StringVar(root)
    paradigm_var.set("Timed exploration")
    paradigm_menu = ctk.CTkOptionMenu(root, variable=paradigm_var, values=["Timed exploration", "Till border exploration", "Food seeking"])
    paradigm_menu.grid(row=4, column=1, sticky="ew", padx=30)

    # Paradigm parameters
    i_row=0
    paradigm_var.trace_add("write", show_parameters)
    # Radius label
    radius_label = ctk.CTkLabel(root, text="→ Border radius:", font=ctk.CTkFont(size=20))
    radius_label.grid_forget()
    # Radius entry
    radius_entry = ctk.CTkEntry(root, placeholder_text=200)
    radius_entry.grid_forget()
    # Food source label
    food_label = ctk.CTkLabel(root, text="→ Food source number:", font=ctk.CTkFont(size=20))
    food_label.grid_forget()
    # Food source entry
    food_entry = ctk.CTkEntry(root, placeholder_text=5)
    food_entry.grid_forget()

    # Initialize error labels
    # Initialize error
    error1_label = ctk.CTkLabel(root, text="Maximum simulation time value should be a positive integer.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))
    error2_label = ctk.CTkLabel(root, text="Noise range value should be a float between 0 and 90.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))
    error3_label = ctk.CTkLabel(root, text="Radius value should be a a positive float.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))
    error4_label = ctk.CTkLabel(root, text="Number of food source should be a a positive integer.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))

    # Run button
    run_button = ctk.CTkButton(root, text="Run simulation", command=run_simulation)
    run_button.grid(row=5, column=0, columnspan=2, pady=20)

    # Main loop
    root.mainloop()