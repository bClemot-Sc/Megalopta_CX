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
        ratio_label.grid_forget()
        ratio_entry.grid_forget()
        timer_label.grid(row=2, column=0, pady=20, sticky="w", padx=70)
        timer_entry.grid(row=2, column=1, sticky="ew", padx=30)
        nest_label.grid(row=3, column=0, pady=20, sticky="w", padx=70)
        nest_entry.grid(row=3, column=1, sticky="ew", padx=30)

    elif selected_option == "Till border exploration":
        food_label.grid_forget()
        food_entry.grid_forget()
        timer_label.grid_forget()
        timer_entry.grid_forget()
        ratio_label.grid_forget()
        ratio_entry.grid_forget()
        radius_label.grid(row=2, column=0, pady=20, sticky="w", padx=70)
        radius_entry.grid(row=2, column=1, sticky="ew", padx=30)
        nest_label.grid(row=3, column=0, pady=20, sticky="w", padx=70)
        nest_entry.grid(row=3, column=1, sticky="ew", padx=30)

    elif selected_option == "Food seeking":
        radius_label.grid_forget()
        radius_entry.grid_forget()
        timer_label.grid_forget()
        timer_entry.grid_forget()
        ratio_label.grid_forget()
        ratio_entry.grid_forget()
        food_label.grid(row=2, column=0, pady=20, sticky="w", padx=70)
        food_entry.grid(row=2, column=1, sticky="ew", padx=30)
        nest_label.grid(row=3, column=0, pady=20, sticky="w", padx=70)
        nest_entry.grid(row=3, column=1, sticky="ew", padx=30)

    elif selected_option == "Debug rotation":
        timer_label.grid_forget()
        timer_entry.grid_forget()
        radius_label.grid_forget()
        radius_entry.grid_forget()
        food_label.grid_forget()
        food_entry.grid_forget()
        ratio_label.grid_forget()
        ratio_entry.grid_forget()
        nest_label.grid_forget()
        nest_entry.grid_forget()

    elif selected_option == "Simple double goals":
        timer_label.grid_forget()
        timer_entry.grid_forget()
        food_label.grid_forget()
        food_entry.grid_forget()
        nest_label.grid_forget()
        nest_entry.grid_forget()
        radius_label.grid(row=2, column=0, pady=20, sticky="w", padx=70)
        radius_entry.grid(row=2, column=1, sticky="ew", padx=30)
        ratio_label.grid(row=3, column=0, pady=20, sticky="w", padx=70)
        ratio_entry.grid(row=3, column=1, sticky="ew", padx=30)

# Check for missing inputs and run simulation
def run_simulation():
    k=0
    error=0

    # Remove previous error
    error1_label.grid_forget()
    error2_label.grid_forget()
    error3_label.grid_forget()
    error4_label.grid_forget()
    error5_label.grid_forget()
    error6_label.grid_forget()
    error7_label.grid_forget()
    error8_label.grid_forget()

    # Check for inputs and issues
    try:
        TIME = int(time_entry.get())
        if TIME <=0:
                error1_label.grid(row=8+k, column=0, columnspan=4,padx=50, pady=10)
                error=1
                k+=1
    except:
        error1_label.grid(row=8+k, column=0, columnspan=4,padx=50, pady=10)
        error=1
        k+=1

    try:
        NOISE = float(noise_entry.get())
        if not 0.0 <= NOISE <= 90.0:
                error2_label.grid(row=8+k, column=4, columnspan=2,padx=50, pady=10)
                error=1
                k+=1 
    except:
        error2_label.grid(row=8+k, column=0, columnspan=4,padx=50, pady=10)
        error=1
        k+=1

    try:
        TRIAL = int(trial_entry.get())
        if not 1 <= TRIAL:
                error8_label.grid(row=8+k, column=4, columnspan=2,padx=50, pady=10)
                error=1
                k+=1 
    except:
        error8_label.grid(row=8+k, column=0, columnspan=4,padx=50, pady=10)
        error=1
        k+=1

    PARADIGM = paradigm_menu.get()

    TIMER = 250
    if PARADIGM == "Timed exploration":
        try:
            TIMER = int(timer_entry.get())
            if not 0.0 <= TIMER <= TIME:
                error5_label.grid(row=8+k, column=0, columnspan=4,padx=50, pady=10)
                error=1
                k+=1 
        except:
            error5_label.grid(row=8+k, column=0, columnspan=4,padx=50, pady=10)
            error=1
            k+=1

    RADIUS = 200
    if PARADIGM == "Till border exploration":
        try:
            RADIUS = float(radius_entry.get())
            if not 0.0 <= RADIUS:
                error3_label.grid(row=8+k, column=0, columnspan=4,padx=50, pady=10)
                error=1
                k+=1 
        except:
            error3_label.grid(row=8+k, column=0, columnspan=4,padx=50, pady=10)
            error=1
            k+=1 
    
    FOOD = 5
    if PARADIGM == "Food seeking":
        try:
            FOOD = int(food_entry.get())
            if not 0.0 <= FOOD:
                error4_label.grid(row=8+k, column=0, columnspan=4,padx=50, pady=10)
                error=1
                k+=1 
        except:
            error4_label.grid(row=8+k, column=0, columnspan=4,padx=50, pady=10)
            error=1
            k+=1       

    RATIO = 0.5
    if PARADIGM == "Simple double goals":
        try:
            RATIO = float(ratio_entry.get())
            if not 0.0 <= RATIO <= 1.0:
                error7_label.grid(row=8+k, column=0, columnspan=4,padx=50, pady=10)
                error=1
                k+=1 
        except:
            error7_label.grid(row=8+k, column=0, columnspan=4,padx=50, pady=10)
            error=1
            k+=1 

    NEST = 0
    if PARADIGM in ["Timed exploration", "Till border exploration", "Food seeking"]:
        try:
            NEST = float(nest_entry.get())
            if not 0.0 <= NEST:
                    error6_label.grid(row=8+k, column=0, columnspan=4,padx=50, pady=10)
                    error=1
                    k+=1 
        except:
            error6_label.grid(row=8+k, column=0, columnspan=4,padx=50, pady=10)
            error=1
            k+=1

    PERIOD = period_menu.get()

    GRAPHIC = [activity_button.get(), pathway_button.get(), circle_button.get(), sinusoid_button.get()]

    # Run the simulation
    if error==0:
        CX_Script.run_function(TIME, PERIOD, NOISE, NEST, PARADIGM, TIMER, RADIUS, FOOD, RATIO, TRIAL, GRAPHIC)


## ----- Configure GUI
if __name__ == "__main__":
    # Configure theme
    ctk.set_default_color_theme("green")
    # Call window
    root = ctk.CTk()
    root.after(0, lambda:root.state('zoomed'))
    root.title("CX simulation")
    # Get window size
    win_width = root.winfo_width()
    win_height = root.winfo_height()  
    # Configure columns
    root.columnconfigure(0, weight=1, uniform="column")
    root.columnconfigure(1, weight=1, uniform="column")
    root.columnconfigure(2, weight=1, uniform="column")
    root.columnconfigure(3, weight=1, uniform="column")

    # Title label
    title_label = ctk.CTkLabel(root, text="- Central Complex Control Panel -", font=ctk.CTkFont(size=25, weight="bold"))
    title_label.grid(row=0, column=0, columnspan=4, pady=20)

    # Frames
    # Left
    left_frame = ctk.CTkFrame(root, height=350)
    left_frame.grid_columnconfigure(0, weight=1, uniform=["column"])
    left_frame.columnconfigure(1, weight=1, uniform=["column"])
    left_frame.grid(row=1, column=0, rowspan=4, columnspan=2, sticky="ew", padx=10, pady=10)
    left_frame.grid_propagate(0)
    # Right
    right_frame = ctk.CTkFrame(root, height=350)
    right_frame.grid_columnconfigure(0, weight=1, uniform=["column"])
    right_frame.grid_columnconfigure(1, weight=1, uniform=["column"])
    right_frame.grid(row=1, column=2, rowspan=4, columnspan=2, sticky="ew", padx=10, pady=10)
    right_frame.grid_propagate(0)
    # Below
    below_frame = ctk.CTkFrame(root, height=125)
    below_frame.grid_columnconfigure(0, weight=1, uniform=["column"])
    below_frame.grid_columnconfigure(1, weight=1, uniform=["column"])
    below_frame.grid_columnconfigure(2, weight=1, uniform=["column"])
    below_frame.grid_columnconfigure(3, weight=1, uniform=["column"])
    below_frame.grid(row=5, column=0, rowspan=2, columnspan=4, sticky="ew", padx=10, pady=10)
    below_frame.grid_propagate(0)

    # Basic parameters title label
    basic_label = ctk.CTkLabel(left_frame, text="General parameters:", font=ctk.CTkFont(size=20, weight="bold"))
    basic_label.grid(row=0, column=0, columnspan=2, pady=20)

    # Simulation time label
    time_label = ctk.CTkLabel(left_frame, text="• Maximum simulation time:", font=ctk.CTkFont(size=20))
    time_label.grid(row=1, column=0, pady=20, sticky="w", padx=50)
    # Simulation time entry
    time_entry = ctk.CTkEntry(left_frame, placeholder_text=500)
    time_entry.grid(row=1, column=1, sticky="ew", padx=30)

    # Time period label
    period_label = ctk.CTkLabel(left_frame, text="• Time period:", font=ctk.CTkFont(size=20))
    period_label.grid(row=2, column=0, pady=20, sticky="w", padx=50)
    # Time period menu
    period_menu = ctk.CTkOptionMenu(left_frame, values=["Day", "Night"])
    period_menu.grid(row=2, column=1, sticky="ew", padx=30)

    # Noise range label
    noise_label = ctk.CTkLabel(left_frame, text="• Noise range:", font=ctk.CTkFont(size=20))
    noise_label.grid(row=3, column=0, pady=20, sticky="w", padx=50)
    # Noise range entry
    noise_entry = ctk.CTkEntry(left_frame, placeholder_text=10)
    noise_entry.grid(row=3, column=1, sticky="ew", padx=30)

    # Number trial label
    trial_label = ctk.CTkLabel(left_frame, text="• Number of trials:", font=ctk.CTkFont(size=20))
    trial_label.grid(row=4, column=0, pady=20, sticky="w", padx=50)
    # Noise range entry
    trial_entry = ctk.CTkEntry(left_frame, placeholder_text=1)
    trial_entry.grid(row=4, column=1, sticky="ew", padx=30)

    # Paradigm parameters title label
    paradigm_title_label = ctk.CTkLabel(right_frame, text="Paradigm parameters:", font=ctk.CTkFont(size=20, weight="bold"))
    paradigm_title_label.grid(row=0, column=0, columnspan=2, pady=20)

    # Paradigm label
    paradigm_label = ctk.CTkLabel(right_frame, text="• Paradigm:", font=ctk.CTkFont(size=20))
    paradigm_label.grid(row=1, column=0, pady=20, sticky="w", padx=50)
    # Paradigm menu
    paradigm_var = ctk.StringVar(right_frame)
    paradigm_var.set("Timed exploration")
    paradigm_menu = ctk.CTkOptionMenu(right_frame, variable=paradigm_var, values=["Timed exploration", "Till border exploration", "Food seeking", "Debug rotation", "Simple double goals"])
    paradigm_menu.grid(row=1, column=1, sticky="ew", padx=30)

    # Paradigm parameters
    paradigm_var.trace_add("write", show_parameters)
    # Timer label
    timer_label = ctk.CTkLabel(right_frame, text="→ Time before homing:", font=ctk.CTkFont(size=20))
    timer_label.grid(row=2, column=0, pady=20, sticky="w", padx=70)
    # Timer entry
    timer_entry = ctk.CTkEntry(right_frame, placeholder_text=250)
    timer_entry.grid(row=2, column=1, sticky="ew", padx=30)
    # Radius label
    radius_label = ctk.CTkLabel(right_frame, text="→ Border radius:", font=ctk.CTkFont(size=20))
    radius_label.grid_forget()
    # Radius entry
    radius_entry = ctk.CTkEntry(right_frame, placeholder_text=200)
    radius_entry.grid_forget()
    # Food source label
    food_label = ctk.CTkLabel(right_frame, text="→ Food source number:", font=ctk.CTkFont(size=20))
    food_label.grid_forget()
    # Food source entry
    food_entry = ctk.CTkEntry(right_frame, placeholder_text=5)
    food_entry.grid_forget()
    # Nest size label
    nest_label = ctk.CTkLabel(right_frame, text="→ Nest size:", font=ctk.CTkFont(size=20))
    nest_label.grid(row=3, column=0, pady=20, sticky="w", padx=70)
    # Nest size entry
    nest_entry = ctk.CTkEntry(right_frame, placeholder_text=20)
    nest_entry.grid(row=3, column=1, sticky="ew", padx=30)
    # Goal ratio label
    ratio_label = ctk.CTkLabel(right_frame, text="→ Goal direction ratio:", font=ctk.CTkFont(size=20))
    ratio_label.grid_forget()
    # Goal ratio entry
    ratio_entry = ctk.CTkEntry(right_frame, placeholder_text=0.5)
    ratio_entry.grid_forget()

    # Graphical output title
    graphic_title__label = ctk.CTkLabel(below_frame, text="Graphical outputs:", font=ctk.CTkFont(size=20, weight="bold"))
    graphic_title__label.grid(row=0, column=0, columnspan=4, pady=20)

    # Activity graphical checkbutton
    activity_button = ctk.CTkCheckBox(below_frame, text="Neuron activity", font=ctk.CTkFont(size=15))
    activity_button.grid(row=1, column=0, padx=30)
    activity_button.select()

    # Pathway graphical checkbutton
    pathway_button = ctk.CTkCheckBox(below_frame, text="Agent journey", font=ctk.CTkFont(size=15))
    pathway_button.grid(row=1, column=1, padx=30)
    pathway_button.select()

    # Circle graphical checkbutton
    circle_button = ctk.CTkCheckBox(below_frame, text="Direction circle", font=ctk.CTkFont(size=15))
    circle_button.grid(row=1, column=2, padx=30)

    # Sinusoid graphical checkbutton
    sinusoid_button = ctk.CTkCheckBox(below_frame, text="Sinusoid shape", font=ctk.CTkFont(size=15))
    sinusoid_button.grid(row=1, column=3, padx=30)

    # Initialize error labels
    # Initialize error
    error1_label = ctk.CTkLabel(root, text="Maximum simulation time value should be a positive integer.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))
    error2_label = ctk.CTkLabel(root, text="Noise range value should be a float between 0 and 90.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))
    error3_label = ctk.CTkLabel(root, text="Radius value should be a a positive float.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))
    error4_label = ctk.CTkLabel(root, text="Number of food source should be a a positive integer.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))
    error5_label = ctk.CTkLabel(root, text="Timer before homing behaviour should be a positive integer lower or equal to maximum simulation time.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))
    error6_label = ctk.CTkLabel(root, text="Nest size should be a positive float value.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))
    error7_label = ctk.CTkLabel(root, text="Ratio value should be a float between 0 and 1.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))
    error8_label = ctk.CTkLabel(root, text="Number of trials hould be a positive integer", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))

    # Run button
    run_button = ctk.CTkButton(root, text="Run simulation", command=run_simulation)
    run_button.grid(row=7, column=0, columnspan=4, pady=20)

    # Main loop
    root.mainloop()