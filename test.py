#### GUI for CX modeling
## Autor: Bastien Clémot


## ----- Import packages
import customtkinter as ctk
import CX_Script


## ----- Command functions
# Check for missing inputs and run simulation
def run_simulation():
    k=0
    error=0

    # Remove previous error
    error1_label.grid_forget()
    error2_label.grid_forget()

    # Check for inputs and issues
    try:
        TIME = int(time_entry.get())
        if TIME <=0:
                error1_label.grid(row=4+k, column=0, columnspan=2,padx=50, pady=10)
                error=1
                k+=1
    except ValueError:
        error1_label.grid(row=4+k, column=0, columnspan=2,padx=50, pady=10)
        error=1
        k+=1
    
    try:
        NOISE = float(noise_entry.get())
        if not 0.0 <= NOISE <= 90.0:
                error2_label.grid(row=4+k, column=0, columnspan=2,padx=50, pady=10)
                error=1
                k+=1 
    except ValueError:
        error2_label.grid(row=4+k, column=0, columnspan=2,padx=50, pady=10)
        error=1
        k+=1

    PERIOD = period_menu.get()

    # Adjust run button position
    run_button.grid(row=4+k, column=0, columnspan=2, pady=20)

    # Run the simulation
    if error==0:
        CX_Script.run_function(CX_Script.CON_MAT, TIME, PERIOD, NOISE, THRESHOLD)


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
    time_label = ctk.CTkLabel(root, text="• Simulation time:", font=ctk.CTkFont(size=20))
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
    noise_entry = ctk.CTkEntry(root, placeholder_text=0.1)
    noise_entry.grid(row=3, column=1, sticky="ew", padx=30)

    # Initialize error labels
    # Initialize error
    error1_label = ctk.CTkLabel(root, text="Simulation time value should be a positive integer.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))
    error2_label = ctk.CTkLabel(root, text="Noise range value should be a float between 0 and 90.", text_color="red", font=ctk.CTkFont(size=15, weight="bold"))

    # Run button
    run_button = ctk.CTkButton(root, text="Run simulation", command=run_simulation)
    run_button.grid(row=4, column=0, columnspan=2, pady=20)

    # Main loop
    root.mainloop()